# coding=utf-8
# Copyright 2025 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the SLA optimizer."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
from init2winit.optimizer_lib import sla
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _build_sgd():
  return optax.sgd(1.0)


class OptimizerTestState(NamedTuple):
  """Fast optimizer state for the lookahead tests."""

  aggregate_grads: optax.Params
  # Include a variable with non-zero initial value to check that it is reset
  # correctly by the lookahead optimizer.
  is_reset: bool = True


def _test_optimizer(step_size: float) -> optax.GradientTransformation:
  """Fast optimizer for the lookahead tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behavior of lookahead can be tested.
  def init_fn(params):
    aggregate_grads = jax.tree.map(jnp.zeros_like, params)
    return OptimizerTestState(aggregate_grads, is_reset=True)

  def update_fn(updates, state, params):
    # The test optimizer does not use the parameters, but we check that they
    # have been passed correctly.
    chex.assert_trees_all_equal_shapes(updates, params)
    aggregate_grads = optax.apply_updates(state.aggregate_grads, updates)
    updates = jax.tree.map(lambda u: step_size * u, updates)
    return updates, OptimizerTestState(aggregate_grads, is_reset=False)

  return optax.GradientTransformation(init_fn, update_fn)


class SLATest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.grads = {'x': np.array(2.0), 'y': np.array(-2.0)}
    self.initial_params = {'x': np.array(3.0), 'y': np.array(-3.0)}

  def loop(self, optimizer, num_steps, params):
    """Performs a given number of optimizer steps."""
    init_fn, update_fn = optimizer
    # Use the chex variant to check various function versions (jit, pmap, etc).
    step = self.variant(update_fn)
    opt_state = self.variant(init_fn)(params)

    # A no-op change, to verify that tree map works.
    opt_state = optax.tree_map_params(init_fn, lambda v: v, opt_state)

    for _ in range(num_steps):
      updates, opt_state = step(self.grads, opt_state, params)
      params = optax.apply_updates(params, updates)

    return params, opt_state

  @chex.all_variants
  def test_lookahead(self):
    """Tests the lookahead optimizer in an analytically tractable setting."""
    # IE, SLA with outer momentum = 0.0.
    sync_period = 3
    optimizer = sla.super_lookahead(
        _test_optimizer(-0.5),
        outer_momentum=0.0,
        sync_period=sync_period,
        slow_step_size=1 / 3,
    )

    final_params, final_state = self.loop(
        optimizer, 2 * sync_period, self.initial_params
    )
    # x steps must be: 3 -> 2 -> 1 -> 2 (sync) -> 1 -> 0 -> 1 (sync).
    # Similarly for y (with sign flipped).
    correct_final_params = {'x': 1, 'y': -1}
    chex.assert_trees_all_close(final_state.slow_params, correct_final_params)
    chex.assert_trees_all_close(final_params, correct_final_params)

  @chex.all_variants
  @parameterized.parameters([False], [True])
  def test_lookahead_state_reset(self, reset_state):
    """Checks that lookahead resets the fast optimizer state correctly."""
    num_steps = sync_period = 3
    fast_optimizer = _test_optimizer(-0.5)
    optimizer = sla.super_lookahead(
        fast_optimizer,
        outer_momentum=0.0,
        sync_period=sync_period,
        slow_step_size=0.5,
        reset_state=reset_state,
    )

    _, opt_state = self.loop(optimizer, num_steps, self.initial_params)

    # A no-op change, to verify that this does not break anything
    opt_state = optax.tree_map_params(optimizer, lambda v: v, opt_state)

    fast_state = opt_state.fast_state
    if reset_state:
      correct_state = fast_optimizer.init(self.initial_params)
    else:
      _, correct_state = self.loop(
          fast_optimizer, num_steps, self.initial_params
      )

    chex.assert_trees_all_close(fast_state, correct_state)

  @chex.all_variants
  @parameterized.parameters(
      [1, 0.5, {'x': np.array(1.), 'y': np.array(-1.)}],
      [1, 0, {'x': np.array(3.), 'y': np.array(-3.)}],
      [1, 1, {'x': np.array(-1.), 'y': np.array(1.)}],
      [2, 1, {'x': np.array(-1.), 'y': np.array(1.)}])  # pyformat: disable
  def test_lookahead_edge_cases(
      self, sync_period, slow_step_size, correct_result
  ):
    """Checks special cases of the lookahed optimizer parameters."""
    # These edge cases are important to check since users might use them as
    # simple ways of disabling lookahead in experiments.
    optimizer = sla.super_lookahead(
        _test_optimizer(-1),
        sync_period,
        slow_step_size,
        outer_momentum=0.0,
    )
    _, final_state = self.loop(
        optimizer,
        num_steps=2,
        params=self.initial_params,
    )
    chex.assert_trees_all_close(final_state.slow_params, correct_result)

  @chex.all_variants
  def test_sla(self):
    """Tests SLA in an analytically tractable setting."""
    sync_period = 3
    outer_momentum = 0.9
    optimizer = sla.super_lookahead(
        _test_optimizer(-0.5),
        outer_momentum=outer_momentum,
        sync_period=sync_period,
        slow_step_size=1 / 3,
        # Leaving nesterov=False because I find it easier to write the
        # calculations down.
        nesterov=False,
    )

    fast_params, final_state = self.loop(
        optimizer, 2 * sync_period, self.initial_params
    )
    # Inner steps will be the same, as will the first outer step (as accumulator
    # starts at 0). But the final step will be different. Without momentum, the
    # steps would be like those above:
    # x steps must be: 3 -> 2 -> 1 -> 2 (sync) -> 1 -> 0 -> 1 (sync).
    # Similarly for y (with sign flipped).

    # But now we have an outer momentum at our sync steps. This momentum buffer
    # gets updated with the pseudo-gradient, which in our case is 3. So we are
    # taking a step of size 1 / 3 in with momentum buffer (3 + outer_momentum *
    # 3), thus the expression for expected_x. expected_y is the same, but agains
    # with the sign flipped.
    expected_x = 2 - (3 + outer_momentum * 3) / 3
    expected_y = -expected_x
    correct_final_params = {'x': expected_x, 'y': expected_y}
    chex.assert_trees_all_close(final_state.slow_params, correct_final_params)
    chex.assert_trees_all_close(fast_params, correct_final_params)


if __name__ == '__main__':
  absltest.main()
