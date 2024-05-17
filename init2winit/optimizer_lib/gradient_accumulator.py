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

"""Gradient accumulator optax transformation.

Note that we should not expect superlinear training speedups with increased
batch size; that is, doubling the batch size will only halve the number of
training steps, and because simulating a 2x larger batch size with this
accumulator would take 2x longer, we would not expect using this accumulator to
simulate larger batches would ever result in superior training speeds.
Therefore, you should always just use the largest batch size that fits into your
hardware (https://arxiv.org/abs/1811.03600). That said, this accumulator can be
an extremely useful transformation when studying the effects of scaling batch
sizes beyond existing hardware constraints.

Note that this accumulator does not support using virtual batch sizes that are
larger than the per-step batch size, as this would require multiple forward
steps *up until each batch norm layer* in order to properly calculate the batch
statistics necessary to properly simulate a larger batch size. However, with a
virtual batch size <= per-step batch size, we can avoid this slowdown.

Note that this accumulator should be used before any statistics of the gradients
are calculated. For example, this should be used before computing momentum or
second-moment estimates, so that these are computed using the gradients
calculated on the total batch size.

Note that this computes the *average* accumulated gradient, because we take the
mean across the batch dimension in our loss functions.

Note that we only sync gradients when we are about to update the model, in
order to avoid unnecessary cross replica communications.
"""
from typing import NamedTuple, Optional

from init2winit.optimizer_lib import utils as optimizer_utils
import jax
import jax.numpy as jnp
import optax


class GradientAccumulatorState(NamedTuple):
  """State for the gradient accumulator."""
  base_state: NamedTuple  # The state of the base optimizer.
  hyperparams: dict[str, jnp.ndarray]
  num_per_step_batches: jnp.ndarray  # shape=(), dtype=jnp.int32.
  accumulations: optax.Updates  # Gradient accumulators for each parameter.


def accumulate_gradients(
    per_step_batch_size: int,
    total_batch_size: int,
    virtual_batch_size: Optional[int],
    base_opt_init_fn: optax.TransformInitFn,
    base_opt_update_fn: optax.TransformUpdateFn,
    batch_axis_name: Optional[str] = None,
) -> optax.GradientTransformationExtraArgs:
  """Accumulate gradients.

  Note that we only sync gradients when we are about to update the model, in
  order to avoid unnecessary cross replica communications.

  Args:
    per_step_batch_size: The batch sized used for each individual
      forward/backward step.
    total_batch_size: The total batch size to simulate via accumulation.
    virtual_batch_size: The virtual batch sized used for batch normalization.
    base_opt_init_fn: The initialization function for the base optimizer used to
      generate updates given the total gradient.
    base_opt_update_fn: The update function for the base optimizer used to
      generate updates given the total gradient.
    batch_axis_name: the name of the axis to pmap over. Used to run a pmean
      before applying the optimizer update.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  if (virtual_batch_size is not None and
      virtual_batch_size > per_step_batch_size):
    raise ValueError(
        'Gradient accumulation does not currently support using a virtual '
        'batch size ({}) that is larger than the per-step batch size ({}), as '
        'this would require multiple forward steps *up to each batch norm '
        'layer* in order to properly calculate the batch statistics necessary '
        'to simulate a larger batch size.'.format(
            virtual_batch_size, per_step_batch_size))

  if total_batch_size % per_step_batch_size != 0:
    raise ValueError(
        'Need to step a per-step batch size ({}) that evenly divides the total '
        'batch size ({}).'.format(per_step_batch_size, total_batch_size))

  steps_per_update = total_batch_size // per_step_batch_size

  def init_fn(params):
    base_state = base_opt_init_fn(params)
    return GradientAccumulatorState(
        base_state=base_state,
        hyperparams=base_state.hyperparams,
        num_per_step_batches=jnp.zeros([], jnp.int32),
        accumulations=jax.tree.map(jnp.zeros_like, params))

  @optimizer_utils.no_cross_device_gradient_aggregation
  def update_fn(updates, state, params=None, **extra_args):
    zeros_params = jax.tree.map(jnp.zeros_like, state.accumulations)

    def total_batch_update(total_gradients, params, state):
      # Enough example gradients have been accumulated to represent the total
      # batch size.
      # Note that this is only accurate for losses that take the mean, which is
      # the case for our default cross entropy. This also does not take into
      # account any example weighting, which is rarely used for training
      # batches.
      total_gradients = jax.tree.map(
          lambda x: x / steps_per_update, total_gradients)
      if batch_axis_name:
        # We only sync gradients when we are about to update the model, in order
        # to avoid unnecessary cross replica communications.
        total_gradients = jax.lax.pmean(
            total_gradients, axis_name=batch_axis_name)

      updates, updated_base_state = base_opt_update_fn(
          total_gradients, state.base_state, params=params, **extra_args)
      reset_state = GradientAccumulatorState(
          base_state=updated_base_state,
          hyperparams=updated_base_state.hyperparams,
          num_per_step_batches=0,
          accumulations=zeros_params)
      return updates, reset_state

    def accumulation_continuation(updated_accumulations, _, state):
      updated_state = GradientAccumulatorState(
          base_state=state.base_state,
          hyperparams=state.base_state.hyperparams,
          num_per_step_batches=state.num_per_step_batches + 1,
          accumulations=updated_accumulations)
      return zeros_params, updated_state

    updated_accumulations = jax.tree.map(lambda g, acc: g + acc, updates,
                                         state.accumulations)
    updates, state = jax.lax.cond(
        state.num_per_step_batches == steps_per_update - 1,
        total_batch_update,
        accumulation_continuation,
        updated_accumulations,
        params,
        state)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
