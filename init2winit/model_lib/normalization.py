# coding=utf-8
# Copyright 2021 The init2winit Authors.
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
from typing import Any, Optional

from flax import linen as nn
from init2winit.model_lib import model_utils

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
          '`x.shape[batch_axis]={}`.'.format(
              virtual_batch_size, x.shape[batch_axis]))
  return batch_axis


class VirtualBatchNorm(nn.Module):
  """VirtualBatchNorm Module. Normalizes the input using batch statistics.

  Forked from the original flax nn.BatchNorm layer, this allows users to have
  multiple EMAs per device, one for each virtual batch size. For example, if
  the per-device batch size is 128 and the user specifies
  `virtual_batch_size=32`, 4 EMAs will be created on each device, each updated
  with 1/4 of the per-device batch on each forward pass.

  WARNING: the multiple per-device EMAs this creates need to be manually
  synchronized within each device before being used for evaluation, or when
  synchronizing batch norm statistic across devices.

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
    virtual_batch_size: the size of the virtual batches to construct on
      each device, which will be used to normalize sub-batches of each
      per-device batch. Will create a running average
      with a leading dim of size `x.shape[batch_axis] // virtual_batch_size`,
      one for each sub-batch. Note that the first dim of each state must be
      synchronized whenever synchronizing batch norm running averages. Must
      evenly divide the per-device batch size (as determined by `x`), and
      cannot be combined with `axis_index_groups`. Passing the default value
      of None will replicate the existing nn.BatchNorm behavior without
      virtual batches.
    data_format: only used when `virtual_batch_size` is set, to determine the
      batch axis.
  """
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: model_utils.Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: model_utils.Initializer = initializers.zeros
  scale_init: model_utils.Initializer = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  virtual_batch_size: Optional[int] = None
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

    virtual_batch_size = self.virtual_batch_size
    batch_axis = _get_batch_axis(
        self.data_format,
        x,
        virtual_batch_size,
        use_running_average,
        self.axis_index_groups)
    if virtual_batch_size is None:
      virtual_batch_size = x.shape[batch_axis]

    if use_running_average:
      # Virtual batch norm is not used during evaluation, and we cannot
      # guarantee the train and eval batch sizes are the same, so we use a
      # single virtual batch of size batch_size, and take the first element in
      # the running average array, assuming they have been properly synced
      # across their first dim.
      virtual_batch_size = x.shape[batch_axis]

    x = jnp.asarray(x, jnp.float32)
    num_sub_batches = x.shape[batch_axis] // virtual_batch_size
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
        virtual_batch_size,
        *x.shape[batch_axis + 1:])
    x = jnp.reshape(x, sub_batched_shape)
    ra_means = self.variable('batch_stats', 'batch_norm_running_mean',
                             lambda s: jnp.zeros(s, jnp.float32),
                             (num_sub_batches, *reduced_feature_shape))
    ra_vars = self.variable('batch_stats', 'batch_norm_running_var',
                            lambda s: jnp.ones(s, jnp.float32),
                            (num_sub_batches, *reduced_feature_shape))

    # See NOTE above on initialization behavior.
    initializing = self.is_mutable_collection('params')

    if use_running_average:
      # Note that we assume that the values across the first axis have been
      # properly synchronized.
      mean = jnp.expand_dims(ra_means.value[0], 0)
      var = jnp.expand_dims(ra_vars.value[0], 0)
    else:
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
        ra_means.value = (
            self.momentum * ra_means.value + (1 - self.momentum) * mean)
        ra_vars.value = (
            self.momentum * ra_vars.value + (1 - self.momentum) * var)

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
