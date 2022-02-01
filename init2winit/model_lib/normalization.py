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

"""Virtual Batch Normalization Flax module."""
from typing import Any, Callable, Iterable, Optional

from flax import linen as nn

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


def _get_batch_axis(
    data_format,
    x,
    virtual_batch_size,
    use_running_average,
    axis_index_groups):
  """Get the batch axis of input x, and check for a valid virtual batch size."""
  if data_format:
    if 'N' not in data_format:
      raise ValueError(
          'Could not locate batch axis "N" in `data_format={}`.'.format(
              data_format))
    batch_axis = data_format.index('N')
  else:
    batch_axis = 0

  # Do not run checks for `virtual_batch_size` if we are evaluating.
  if virtual_batch_size is not None and not use_running_average:
    if data_format is None:
      raise ValueError(
          'Must provide `data_format` when providing `virtual_batch_size` '
          'to a VirtualBatchNorm layer.')
    if axis_index_groups is not None:
      raise ValueError(
          'Only one of `virtual_batch_size` or `axis_index_groups` can '
          'be provided to a VirtualBatchNorm layer.')
    if virtual_batch_size < 1:
      raise ValueError(
          'Must have a `virtual_batch_size` > 1, received {}.'.format(
              virtual_batch_size))
    if x.shape[batch_axis] % virtual_batch_size != 0:
      raise ValueError(
          '`virtual_batch_size={}` must evenly divide '
          '`x.shape[batch_axis]={}`. You are likely running with a per-core '
          'batch size < hps.virtual_batch_size; either decrease the number of '
          'cores, increase the total batch size, or decrease '
          'hps.virtual_batch_size.'.format(
              virtual_batch_size, x.shape[batch_axis]))
  return batch_axis


Initializer = Callable[[Any, Iterable[int], Any], Any]


class VirtualBatchNorm(nn.Module):
  """VirtualBatchNorm Module. Normalizes the input using batch statistics.

  Forked from the original flax nn.BatchNorm layer, this allows users to have
  multiple virtual batches per device. For example, if the per-device batch size
  is 128 and the user specifies `virtual_batch_size=32`, each 1/4 of the batch
  will be normalized with statistics only from that 1/4. The running averages
  computed here are averaged at every step, which is equivalent to keeping
  separate running averages and then averaging them at the end of training. Note
  that users must still sync running averages across hosts before checkpointing,
  as is also done with nn.BatchNorm.

  Attributes:
    x: the input to be normalized.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    batch_size: the batch size used for each forward pass. We must explicitly
      pass this instead of relying on `x.shape` because we may initialize the
      model with a batch of zeros with a different batch size.
    virtual_batch_size: the size of the virtual batches to construct on
      each device, which will be used to normalize sub-batches of each
      per-device batch. Must evenly divide the per-device batch size (as
      determined by `x`), and cannot be combined with `axis_index_groups`.
      Passing the default value of None will replicate the existing nn.BatchNorm
      behavior without virtual batches.
    total_batch_size: only necessary when using gradient accumulation, the total
      batch size used to calculate accumulated gradients. This is required here
      because we need to store `total_batch_size // virtual_batch_size` EMAs
      instead of just `x.shape[batch_axis] // virtual_batch_size`.
    data_format: only used when `virtual_batch_size` is set, to determine the
      batch axis.
  """
  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: Any = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None

  @nn.compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """Normalizes the input using batch statistics.

    NOTE:
    During initialization (when parameters are mutable) the running average
    of the batch statistics will not be updated. Therefore, the inputs
    fed during initialization don't need to match that of the actual input
    distribution and the reduction axis (set with `axis_name`) does not have
    to exist.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_average = nn.module.merge_param(
        'use_running_average', self.use_running_average, use_running_average)

    virtual_batch_size = self.virtual_batch_size
    batch_axis = _get_batch_axis(
        self.data_format,
        x,
        virtual_batch_size,
        use_running_average,
        self.axis_index_groups)
    if virtual_batch_size is None:
      virtual_batch_size = self.batch_size

    if use_running_average:
      # Virtual batch norm is not used during evaluation, and we cannot
      # guarantee the train and eval batch sizes are the same, so we use a
      # single virtual batch of size batch_size, and take the first element in
      # the running average array, assuming they have been properly synced
      # across their first dim.
      virtual_batch_size = x.shape[batch_axis]

    x = jnp.asarray(x, jnp.float32)
    # Note that this should only ever default to the first case if we are
    # passing in a batch `x` with less examples than `virtual_batch_size`, which
    # should only happen if we are initializing with dummy variables (typically
    # of batch size 2).
    num_sub_batches = max(1, x.shape[batch_axis] // virtual_batch_size)
    input_shape = x.shape
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    # Add an additional axis because we are going to reshape `x` to have a
    # leading dim of size `virtual_batch_size`.
    reduction_axis = tuple(i + 1 for i in range(x.ndim) if i not in axis)
    sub_batched_shape = (
        num_sub_batches,
        *x.shape[:batch_axis],
        # Necessary for when passing in a batch `x` with less examples than
        # `virtual_batch_size, which should only happen if we are initializing
        # with dummy variables (typically of batch size 2).
        min(x.shape[batch_axis], virtual_batch_size),
        *x.shape[batch_axis + 1:])
    x = jnp.reshape(x, sub_batched_shape)
    ra_mean = self.variable('batch_stats', 'batch_norm_running_mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            feature_shape)
    ra_var = self.variable('batch_stats', 'batch_norm_running_var',
                           lambda s: jnp.ones(s, jnp.float32),
                           feature_shape)
    # If using gradient accumulation, use these to accumulate the activations
    # for the current batch before folding them into the running average.
    mean_accumulator = self.variable(
        'batch_stats', 'batch_norm_mean_accumulator',
        lambda s: jnp.zeros(s, jnp.float32), feature_shape)
    mean2_accumulator = self.variable(
        'batch_stats', 'batch_norm_mean2_accumulator',
        lambda s: jnp.zeros(s, jnp.float32), feature_shape)

    # A counter that is used to determine which accumulation pass we are
    # currently in. This will increment from 0 until we have accumulated
    # gradients calculated on `self.total_batch_size` examples. This should only
    # ever be saved on disk as 0 because we only checkpoint after accumulating
    # enough examples to make an update.
    grad_accum_counter = self.variable('batch_stats', 'grad_accum_counter',
                                       lambda s: jnp.zeros(s, jnp.int32), [])

    # See NOTE above on initialization behavior.
    initializing = self.is_mutable_collection('params')

    if self.total_batch_size is None:
      passes_per_total_batch = 1
    else:
      passes_per_total_batch = self.total_batch_size // self.batch_size

    if use_running_average:
      # Note that we assume that the values across the first axis have been
      # properly synchronized.
      mean = jnp.expand_dims(ra_mean.value, 0)
      var = jnp.expand_dims(ra_var.value, 0)
    else:
      # Shape (num_sub_batches, x.shape[axis]).
      mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
      mean2 = jnp.mean(
          lax.square(x), axis=reduction_axis, keepdims=False)
      if self.axis_name is not None and not initializing:
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups), 2)
      var = mean2 - lax.square(mean)

      if not initializing:
        mean_accumulator.value += jnp.mean(mean, axis=0)
        mean2_accumulator.value += jnp.mean(mean2, axis=0)
        grad_accum_counter_inc = grad_accum_counter.value + 1
        # This will be 0 for all gradient accumulation passes except for the
        # last one when we have seen enough examples to make an update to the
        # running averages.
        should_update_ra = grad_accum_counter_inc // passes_per_total_batch
        ra_mean_update = (
            should_update_ra * mean_accumulator.value / grad_accum_counter_inc)
        ra_mean.value = (
            (1 - should_update_ra * (1 - self.momentum)) * ra_mean.value +
            (1 - self.momentum) * ra_mean_update)
        ra_var_update = should_update_ra * (
            mean2_accumulator.value / grad_accum_counter_inc -
            lax.square(mean_accumulator.value / grad_accum_counter_inc))
        ra_var.value = (
            (1 - should_update_ra * (1 - self.momentum)) * ra_var.value +
            (1 - self.momentum) * ra_var_update)

        grad_accum_counter.value = (
            grad_accum_counter_inc % passes_per_total_batch)
        # Reset the activation accumulators every `passes_per_total_batch` steps
        # (np.sign == 0 if grad_accum_counter == 0).
        mean_accumulator.value *= jnp.sign(grad_accum_counter.value)
        mean2_accumulator.value *= jnp.sign(grad_accum_counter.value)

    y = x - mean.reshape((num_sub_batches, *feature_shape))
    mul = lax.rsqrt(
        var.reshape((num_sub_batches, *feature_shape)) + self.epsilon)
    if self.use_scale:
      mul = mul * self.param(
          'scale', self.scale_init, reduced_feature_shape).reshape(
              (1, *feature_shape))
    y = y * mul
    if self.use_bias:
      y = y + self.param(
          'bias', self.bias_init, reduced_feature_shape).reshape(
              (1, *feature_shape))
    y = jnp.reshape(y, input_shape)
    return jnp.asarray(y, self.dtype)
