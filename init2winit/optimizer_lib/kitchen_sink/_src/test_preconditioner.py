# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Tests for search_subspace."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from init2winit.optimizer_lib.kitchen_sink._src import preconditioner
from init2winit.optimizer_lib.kitchen_sink._src import transform
import jax
import jax.numpy as jnp
import optax


class GradTransformTest(parameterized.TestCase):
  """Test grad transforms."""

  @parameterized.product(
      power=[1, 2, 4, [1, 2], [0, 3]],
      seed=[0, 1],
  )
  def test_nth_power(self, power, seed):
    """Test nth_power."""
    grads = jax.random.uniform(jax.random.PRNGKey(seed), (10, 10))
    grads = {'updates': grads, 'variables': {}, 'moments': {}, 'output': None}

    actual_power = power
    if not isinstance(power, list):
      actual_power = [actual_power]

    actual_grads = {
        str(int(o)): jax.tree.map(lambda x: x**o, grads['updates'])  # pylint: disable=cell-var-from-loop
        for o in actual_power
    }
    nth_grads, _ = preconditioner.nth_power(power).update(grads, None)

    chex.assert_trees_all_close(actual_grads, nth_grads['variables'])


class AccumulatorTest(parameterized.TestCase):
  """Test accumulators."""

  @parameterized.product(
      decay=[0.1, 0.5, 0.9, 0.999],
      debias=[True, False],
  )
  def test_ema_accumulator(self, decay, debias):
    """Test ema_accumulator."""
    grads = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))
    grads = {'updates': grads, 'variables': {}, 'moments': {}, 'output': None}

    nth_grads = preconditioner.nth_power(power=[1, 2])
    accumulator = preconditioner.ema_accumulator(decay, debias)

    grads, _ = nth_grads.update(grads, None)
    state = accumulator.init(grads['variables'])

    for i in range(5):
      moments, count = state
      grads = jax.random.uniform(jax.random.PRNGKey(i + 1), (10, 10))
      grads = {'updates': grads, 'variables': {}, 'moments': {}, 'output': None}
      grads, _ = nth_grads.update(grads, None)
      updates, state = accumulator.update(grads, state)

      actual = jax.tree.map(lambda g, t: (1 - decay) * g + decay * t,
                            grads['variables'], moments)
      if debias:
        count += jnp.array(1, dtype=jnp.int32)
        beta = jnp.array(1, dtype=jnp.int32) - decay ** count
        actual = jax.tree.map(lambda t: t / beta.astype(t.dtype), actual)  # pylint: disable=cell-var-from-loop

      chex.assert_trees_all_close(updates['moments'], actual)


class PreconditionerTest(parameterized.TestCase):
  """Test preconditioner decompositions."""

  @chex.variants(with_jit=True, with_pmap=True, without_jit=True)
  @parameterized.product(
      decay=[0.999, 0.9, 0.1],
      eps=[1e-8, 1e-4, 1e-1],
      eps_root=[1e-8, 1e-4, 0.0],
      debias=[True, False],
  )
  def test_precondition_by_rms(self, decay, eps, eps_root, debias):
    """Test precondition_by_rms."""

    actual_rms = transform.precondition_by_rms(
        decay=decay, eps=eps, eps_root=eps_root, debias=debias)

    decon_rms = preconditioner.preconditioner(preconditioner.nth_power,
                                              preconditioner.ema_accumulator,
                                              preconditioner.rexp_updater,
                                              {'power': 2}, {
                                                  'decay': decay,
                                                  'debias': debias
                                              }, {
                                                  'eps': eps,
                                                  'eps_root': eps_root,
                                              })

    params = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))

    init = self.variant(decon_rms.init)
    update = self.variant(decon_rms.update)

    actual_state = actual_rms.init(params)
    decon_state = init(params)

    for i in range(5):
      grads = jax.random.uniform(jax.random.PRNGKey(i + 1), (10, 10))

      actual_updates, actual_state = actual_rms.update(grads, actual_state)
      decon_updates, decon_state = update(grads, decon_state)

      actual_params = optax.apply_updates(params, actual_updates)
      decon_params = optax.apply_updates(params, decon_updates)

      chex.assert_trees_all_close(actual_params, decon_params, atol=1e-4)

  @chex.variants(with_jit=True, with_pmap=True, without_jit=True)
  @parameterized.product(
      b2=[0.999, 0.9, 0.1],
      eps=[1e-8, 1e-4, 1e-1],
      eps_root=[1e-8, 1e-4, 0.0],
      initial_accumulator_value=[1e-8, 1e-6, 1e-1],
      debias=[True, False],
  )
  def test_precondition_by_yogi(self, b2, eps, eps_root,
                                initial_accumulator_value, debias):
    """Test precondition_by_yogi."""

    actual_yogi = transform.precondition_by_yogi(
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        initial_accumulator_value=initial_accumulator_value,
        debias=debias)

    decon_yogi = preconditioner.preconditioner(
        preconditioner.nth_power, preconditioner.yogi_accumulator,
        preconditioner.rexp_updater, {'power': 2}, {
            'b2': b2,
            'initial_accumulator_value': initial_accumulator_value,
            'debias': debias
        }, {
            'eps': eps,
            'eps_root': eps_root,
        })

    params = jax.random.uniform(jax.random.PRNGKey(0), (10, 10))

    init = self.variant(decon_yogi.init)
    update = self.variant(decon_yogi.update)

    actual_state = actual_yogi.init(params)
    decon_state = init(params)

    for i in range(5):
      grads = jax.random.uniform(jax.random.PRNGKey(i + 1), (10, 10))

      actual_updates, actual_state = actual_yogi.update(grads, actual_state)
      decon_updates, decon_state = update(grads, decon_state)

      actual_params = optax.apply_updates(params, actual_updates)
      decon_params = optax.apply_updates(params, decon_updates)

      chex.assert_trees_all_close(actual_params, decon_params, atol=1e-4)

if __name__ == '__main__':
  absltest.main()
