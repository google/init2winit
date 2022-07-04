# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""Modularizing optimizer preconditioners.

TL;DR: `optax.GradientTransformation`s, but initializing a bit differently and
abusing the `updates` that get passed from one transformation to the next.

We split up preconditioners into three "stages":
1. Produce the variables to be accumulated.
2. Accumulate the variables.
3. Use `updates` and outputs from above two stages to compute final update.

Each stage is an `optax.GradientTransformation`, but must follow certain rules:

1. `updates` passed into `tx.update` must be of the form

    ```
    {
      'updates': updates,  # The original `updates` passed from training loop.
      'variables': {},     # A dictionary of variables to accumulate.
      'moments': {},       # Moments computed from variables.
      'output': None,      # Final list of outputs.
    }
    ```

2. After stage 1, `variables` is populated, after stage 2, `moments` is
   populated, after stage 3, `output` is populated.

NOTE: `preconditioner` is a wrapper `optax.GradientTransformation` that takes
three `optax.GradientTransformation`s adhering to above rules and correctly
initializes each and forms the `updates` dictionary.
"""

from typing import Optional
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import optax


def preconditioner(
    variable_creator,
    accumulator,
    updater,
    variable_creator_args,
    accumulator_args,
    updater_args,
) -> optax.GradientTransformation:
  """Generic precondition update function."""

  variable_creator = variable_creator(**variable_creator_args)
  accumulator = accumulator(**accumulator_args)
  updater = updater(**updater_args)

  def init(params: optax.Params) -> optax.OptState:
    """`init` function."""
    grads_state = variable_creator.init(params)

    # NOTE(dsuo): assumes params and updates have the same shape.
    updates = {'updates': params, 'variables': {}}
    grads, _ = variable_creator.update(updates, grads_state, params)

    # NOTE(dsuo): assume accumulator only needs `gradients`.
    accumulator_state = accumulator.init(grads['variables'])

    updater_state = updater.init(params)
    return (grads_state, accumulator_state, updater_state)

  def update(updates, state, params=None):
    """`update` function."""
    updates = {
        'updates': updates,
        'variables': {},
        'moments': {},
        'output': None
    }
    new_state = []
    for s, transform in zip(state, [variable_creator, accumulator, updater]):
      updates, new_s = transform.update(updates, s, params)
      new_state.append(new_s)

    return updates['output'], tuple(new_state)

  return optax.GradientTransformation(init, update)


def nth_power(
    power: Union[int, Tuple[int]] = 2) -> optax.GradientTransformation:
  """Create nth power(s) from gradients."""

  if not hasattr(power, '__iter__'):
    power = [power]

  for p in power:
    if p != int(p):
      raise ValueError(f'Currently we only support integer orders; got {p}.')

  def init(params: optax.Params) -> optax.OptState:
    del params
    return None

  def update(
      updates: optax.Updates,
      state: optax.OptState,
      params: Optional[optax.Params] = None
  ) -> Tuple[optax.Updates, optax.OptState]:
    del params

    for p in power:
      if p == 1:
        updates['variables'][str(int(p))] = updates['updates']
      else:
        gradients = jax.tree_map(lambda x: x**p, updates['updates'])  # pylint: disable=cell-var-from-loop
        updates['variables'][str(int(p))] = gradients

    return updates, state

  return optax.GradientTransformation(init, update)


def ema_accumulator(decay: float = 0.999,
                    debias: bool = False) -> optax.GradientTransformation:
  """Create accumulator that computes EMA on all updates."""

  def init(params: optax.Params) -> optax.OptState:
    return (jax.tree_map(jnp.zeros_like, params), jnp.array(0, dtype=jnp.int32))

  def update(
      updates: optax.Updates,
      state: optax.OptState,
      params: Optional[optax.Params] = None
  ) -> Tuple[optax.Updates, optax.OptState]:
    del params

    moments, count = state
    update_fn = lambda g, t: (1 - decay) * g + decay * t
    moments = jax.tree_map(update_fn, updates['variables'], moments)

    count = count + jnp.array(1, dtype=jnp.int32)
    beta = jnp.array(1, dtype=jnp.int32) - decay**count
    updates['moments'] = moments if not debias else jax.tree_map(
        lambda t: t / beta.astype(t.dtype), moments)

    return updates, (moments, count)

  return optax.GradientTransformation(init, update)


# TODO(dsuo): from namanagarwal@: revisit `initial_accumulator_value`.
#             `tensorflow` defaults to this value, but perhaps should consider
#             0 or 1e-8 to match rms-type accumulators.
def yogi_accumulator(b2: float = 0.999,
                     initial_accumulator_value: float = 1e-6,
                     debias: bool = False) -> optax.GradientTransformation:
  """Create yogi accumulator."""

  def init(params: optax.Params) -> optax.OptState:
    return (jax.tree_map(lambda p: jnp.full_like(p, initial_accumulator_value),
                         params), jnp.zeros([], dtype=jnp.int32))

  def update(updates, state, params=None):
    del params
    moments, count = state

    update_fn = lambda g, v: v - (1 - b2) * jnp.sign(v - g) * g
    moments = jax.tree_map(update_fn, updates['variables'], moments)

    count = count + jnp.array(1, dtype=jnp.int32)
    beta = jnp.array(1, dtype=jnp.float32) - b2**count
    updates['moments'] = moments if not debias else jax.tree_map(
        lambda t: t / beta.astype(t.dtype), moments)

    return updates, (moments, count)

  return optax.GradientTransformation(init, update)


def rexp_updater(
    exponent: float = 0.5,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment: int = 2,
    use_accumulated_gradient: bool = False,
) -> optax.GradientTransformation:
  """Apply an update function."""

  def init(params: optax.Params) -> optax.OptState:
    del params
    return None

  def update(
      updates: optax.Updates,
      state: optax.OptState,
      params: Optional[optax.Params] = None
  ) -> Tuple[optax.Updates, optax.OptState]:
    del params
    grads = updates['updates'] if not use_accumulated_gradient else updates[
        'moments']['1']

    updates['output'] = jax.tree_map(
        lambda u, v: u / (jnp.power(v + eps_root, exponent) + eps), grads,
        updates['moments'][str(moment)])

    return updates, state

  return optax.GradientTransformation(init, update)
