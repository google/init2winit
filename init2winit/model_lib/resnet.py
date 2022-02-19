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

"""Flax implementation of ResNet V1."""
import functools
from typing import Optional, Tuple

from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from init2winit.model_lib import normalization
import jax.numpy as jnp
from ml_collections.config_dict import config_dict


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=16,
    num_layers=18,  # Must be one of [18, 34, 50, 101, 152, 200]
    layer_rescale_factors={},
    lr_hparams={
        'schedule': 'constant',
        'base_lr': 0.2,
    },
    optimizer='momentum',
    opt_hparams={
        'momentum': 0.9,
    },
    batch_size=128,
    l2_decay_factor=0.0001,
    l2_decay_rank_threshold=2,
    label_smoothing=None,
    rng_seed=-1,
    use_shallue_label_smoothing=False,
    batch_norm_momentum=0.9,
    batch_norm_epsilon=1e-5,
    # Make this a string to avoid having to import jnp into the configs.
    model_dtype='float32',
    virtual_batch_size=64,
    total_accumulated_batch_size=None,
    data_format='NHWC',
    block_type='post_activation',  # either pre_activation or post_activation
    bn_relu_conv=True,  # only used for block_type='pre_activation'
    use_bn=True,
    dropout_rate=0.0,
    grad_clip=None,
    activation_function='relu',
))


class PreActResidualBlock(nn.Module):
  """PreActResidualBlock.

  As shown in Figure 4(a) in Appendix E of
  https://arxiv.org/abs/2101.08692. The post activation block applies the
  non-linearity after the residual connection, the preactivation block applies
  it before the residual connection.
  """
  filters: int
  strides: Tuple[int] = (1, 1)
  dtype: model_utils.Dtype = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None
  bn_relu_conv: bool = True
  use_bn: bool = True
  activation_function: str = 'relu'

  @nn.compact
  def __call__(self, x, train):
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    def maybe_normalize(name):
      if self.use_bn:
        return normalization.VirtualBatchNorm(
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon,
            dtype=self.dtype,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            total_batch_size=self.total_batch_size,
            data_format=self.data_format,
            name=name)
      else:
        return lambda x, **kwargs: x

    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(
          self.filters * 4, (1, 1), self.strides, name='proj_conv')(residual)
      residual = maybe_normalize(name='proj_bn')(
          residual, use_running_average=not train)

    def _bn_nonlin(y, name):
      if self.bn_relu_conv:
        y = maybe_normalize(name=name)(y, use_running_average=not train)
        y = model_utils.ACTIVATIONS[self.activation_function](y)
      else:
        y = model_utils.ACTIVATIONS[self.activation_function](y)
        y = maybe_normalize(name=name)(y, use_running_average=not train)
      return y

    y = _bn_nonlin(x, 'bn1')
    y = conv(self.filters, (1, 1), name='conv1')(y)
    y = _bn_nonlin(y, 'bn2')
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = _bn_nonlin(y, 'bn3')
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)

    return y + residual


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int] = (1, 1)
  dtype: model_utils.Dtype = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None
  use_bn: bool = True
  bn_relu_conv: bool = True  # Unused.
  activation_function: str = 'relu'

  @nn.compact
  def __call__(self, x, train):
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    def maybe_normalize(name, scale_init=nn.initializers.ones):
      if self.use_bn:
        return normalization.VirtualBatchNorm(
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon,
            dtype=self.dtype,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            total_batch_size=self.total_batch_size,
            data_format=self.data_format,
            scale_init=scale_init,
            name=name)
      else:
        return lambda x, **kwargs: x

    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(
          self.filters * 4, (1, 1), self.strides, name='proj_conv')(residual)
      residual = maybe_normalize(name='proj_bn')(
          residual, use_running_average=not train)

    y = conv(self.filters, (1, 1), name='conv1')(x)
    y = maybe_normalize(name='bn1')(y, use_running_average=not train)
    y = model_utils.ACTIVATIONS[self.activation_function](y)
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = maybe_normalize(name='bn2')(y, use_running_average=not train)
    y = model_utils.ACTIVATIONS[self.activation_function](y)
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)

    y = maybe_normalize(name='bn3', scale_init=nn.initializers.zeros)(
        y, use_running_average=not train)
    y = model_utils.ACTIVATIONS[self.activation_function](residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""
  num_outputs: int
  num_filters: int = 64
  num_layers: int = 50
  dtype: model_utils.Dtype = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None
  block_type: str = 'post_activation'
  bn_relu_conv: bool = True
  use_bn: bool = True
  dropout_rate: float = 0.0
  activation_function: str = 'relu'

  @nn.compact
  def __call__(self, x, train):
    if self.num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[self.num_layers]

    x = nn.Conv(self.num_filters, (7, 7), (2, 2),
                use_bias=False,
                dtype=self.dtype,
                name='init_conv')(x)
    if self.use_bn:
      x = normalization.VirtualBatchNorm(
          momentum=self.batch_norm_momentum,
          epsilon=self.batch_norm_epsilon,
          dtype=self.dtype,
          name='init_bn',
          batch_size=self.batch_size,
          virtual_batch_size=self.virtual_batch_size,
          total_batch_size=self.total_batch_size,
          data_format=self.data_format)(x, use_running_average=not train)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    if self.block_type == 'post_activation':
      residual_block = ResidualBlock
    elif self.block_type == 'pre_activation':
      residual_block = PreActResidualBlock
    else:
      raise ValueError('Invalid Block Type: {}'.format(self.block_type))
    index = 0
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        index += 1
        x = residual_block(
            self.num_filters * 2 ** i,
            strides=strides,
            dtype=self.dtype,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_epsilon=self.batch_norm_epsilon,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            total_batch_size=self.total_batch_size,
            data_format=self.data_format,
            bn_relu_conv=self.bn_relu_conv,
            use_bn=self.use_bn,
            activation_function=self.activation_function)(x, train=train)
    x = jnp.mean(x, axis=(1, 2))
    if self.dropout_rate > 0.0:
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = nn.Dense(self.num_outputs, dtype=self.dtype)(x)
    return x


# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    1: [1],  # used for testing only
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    1000: [3, 24 * 5, 36 * 5, 3]
}


class ResnetModel(base_model.BaseModel):

  def build_flax_module(self):
    """ResnetV1."""
    return ResNet(
        num_filters=self.hps.num_filters,
        num_layers=self.hps.num_layers,
        num_outputs=self.hps['output_shape'][-1],
        dtype=utils.dtype_from_str(self.hps.model_dtype),
        batch_norm_momentum=self.hps.batch_norm_momentum,
        batch_norm_epsilon=self.hps.batch_norm_epsilon,
        batch_size=self.hps.batch_size,
        virtual_batch_size=self.hps.virtual_batch_size,
        total_batch_size=self.hps.total_accumulated_batch_size,
        data_format=self.hps.data_format,
        block_type=self.hps.block_type,
        bn_relu_conv=self.hps.bn_relu_conv,
        use_bn=self.hps.use_bn,
        dropout_rate=self.hps.dropout_rate,
        activation_function=self.hps.activation_function)
