# coding=utf-8
# Copyright 2026 The init2winit Authors.
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
"""Super Lookahead."""

from typing import NamedTuple, Union

from absl import logging
import jax
import jax.numpy as jnp
import optax


class SLAState(NamedTuple):
  """State of the `GradientTransformation` returned by `super_lookahead`.

  Attributes:
    fast_state: Optimizer state of the fast optimizer.
    slow_opt_state: Optimizer state of the slow optimizer.
    steps_since_sync: Number of fast optimizer steps taken since slow and fast
      parameters were synchronized.
    slow_params: The 'global' or 'slow' parameters tracked by SLA.
  """

  fast_state: optax.OptState
  slow_opt_state: optax.OptState
  slow_params: optax.Params
  steps_since_sync: jnp.ndarray


def generic_super_lookahead(
    fast_optimizer: optax.GradientTransformation,
    slow_optimizer: optax.GradientTransformation,
    sync_period: int,
    reset_state: bool = False,
) -> optax.GradientTransformation:
  """Generic version of super lookahead."""

  if sync_period < 1:
    raise ValueError('Synchronization period must be >= 1.')

  def init_fn(params: optax.Params) -> SLAState:
    fast_params = getattr(params, 'fast', None)
    if fast_params is None:
      # Allowing init_fn to be called with fast parameters reduces the
      # modifications necessary to adapt code to use lookahead in some cases.
      logging.warning(
          '`params` has no attribute `fast`. Continuing by assuming that '
          'only fast parameters were passed to lookahead init.'
      )
      fast_params = params

    return SLAState(
        fast_state=fast_optimizer.init(fast_params),
        slow_opt_state=slow_optimizer.init(params),
        slow_params=params,
        steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
    )

  def update_fn(
      updates: optax.Updates, state: SLAState, params: optax.Params
  ) -> tuple[optax.Params, SLAState]:
    updates, fast_state = fast_optimizer.update(
        updates,
        state.fast_state,
        params,
    )

    sync_next = state.steps_since_sync == (sync_period - 1)
    updates, slow_params, slow_opt_state = _lookahead_update(
        updates,
        sync_next,
        params,
        state.slow_params,
        state.slow_opt_state,
        slow_optimizer,
    )
    if reset_state:
      # Jittable way of resetting the fast optimizer state if parameters will be
      # synchronized after this update step.
      initial_state = fast_optimizer.init(params)
      fast_state = jax.tree.map(
          lambda current, init: (1 - sync_next) * current + sync_next * init,
          fast_state,
          initial_state,
      )

    steps_since_sync = (state.steps_since_sync + 1) % sync_period
    return updates, SLAState(
        fast_state=fast_state,
        slow_opt_state=slow_opt_state,
        slow_params=slow_params,
        steps_since_sync=steps_since_sync,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def super_lookahead(
    fast_optimizer: optax.GradientTransformation,
    sync_period: int,
    slow_step_size: float,
    outer_momentum: float,
    reset_state: bool = False,
    nesterov: bool = True,
) -> optax.GradientTransformation:
  """Super Lookahead optimizer.

  Performs steps with a fast optimizer and periodically updates a set of slow
  parameters, with momentum. Optionally resets the fast optimizer
  state after synchronization by calling the init function of the fast
  optimizer.

  Updates returned by the lookahead optimizer should not be modified before they
  are applied, otherwise fast and slow parameters are not synchronized
  correctly.

  Args:
    fast_optimizer: The optimizer to use in the inner loop of lookahead.
    sync_period: Number of fast optimizer steps to take before synchronizing
      parameters. Must be >= 1.
    slow_step_size: Step size of the slow parameter updates.
    outer_momentum: Momentum to use for slow param updates.
    reset_state: Whether to reset the optimizer state of the fast optimizer
      after each synchronization.
    nesterov: Whether or not to use nesterov momentum.

  Returns:
    A :class:`optax.GradientTransformation` with init and update functions. The
    updates passed to the update function should be calculated using the fast
    lookahead parameters only.

  References:
    Zhang et al, `Lookahead Optimizer: k steps forward, 1 step back
    <https://arxiv.org/abs/1907.08610>`_, 2019
    Diloco: https://arxiv.org/abs/2311.08105.
    Scaling laws for diloco: https://arxiv.org/abs/2503.09799.
  """

  slow_optimizer = optax.sgd(
      learning_rate=slow_step_size, momentum=outer_momentum, nesterov=nesterov
  )
  return generic_super_lookahead(
      fast_optimizer=fast_optimizer,
      slow_optimizer=slow_optimizer,
      sync_period=sync_period,
      reset_state=reset_state,
  )


def _lookahead_update(
    updates: optax.Updates,
    sync_next: Union[bool, jax.Array],
    params: optax.Params,
    slow_params: optax.Params,
    slow_opt_state: optax.OptState,
    outer_opt: optax.GradientTransformation,
) -> tuple[optax.Params, optax.Params, optax.OptState]:
  """Returns the updates corresponding to one lookahead step.

  Args:
    updates: Updates returned by the fast optimizer.
    sync_next: Wether fast and slow parameters should be synchronized after the
      fast optimizer step.
    params: Current fast parameters.
    slow_params: Current slow parameters.
    slow_opt_state: Current state of the slow optimizer.
    outer_opt: Outer optimizer to use.

  Returns:
    The updates for the lookahead parameters.

  References:
    Zhang et al, `Lookahead Optimizer: k steps forward, 1 step back
    <https://arxiv.org/abs/1907.08610>`_, 2019
  """

  def _update_all_fn(slow_opt_state):
    # Working in update space is forcing us to do a little extra work, but it
    # should all be cheap.
    updated_fast_params = optax.apply_updates(params=params, updates=updates)
    # This is the analog of the _negative_ pseudogradient, so we need to negate
    # it before we pass to the outer optimizer
    last_difference = jax.tree.map(
        lambda f, s: f - s, updated_fast_params, slow_params
    )
    negative_last_difference = jax.tree.map(lambda x: -x, last_difference)
    slow_updates, slow_opt_state = outer_opt.update(
        negative_last_difference,
        slow_opt_state,
        slow_params,
    )
    updated_slow_params = optax.apply_updates(
        params=slow_params, updates=slow_updates
    )
    # We want the fast updates to solve the equation fast_params =
    # updated_slow_params = old fast_params + fast updates. So taking
    # fast_updates = updated_slow_params - old_fast_params solves this.
    fast_updates = jax.tree.map(lambda x, y: x - y, updated_slow_params, params)
    return fast_updates, updated_slow_params, slow_opt_state

  def _only_update_fast_fn(slow_opt_state):
    # Leave the slow weights where they are, let the fast weights take the step.
    return updates, slow_params, slow_opt_state

  return jax.lax.cond(
      sync_next, _update_all_fn, _only_update_fast_fn, slow_opt_state
  )
