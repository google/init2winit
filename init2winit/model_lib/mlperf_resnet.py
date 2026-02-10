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

"""Flax implementation of the MLPerf ResNet V1.5 model."""
import functools

from typing import Any, Optional, Tuple

from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from init2winit.model_lib import normalization
import jax.numpy as jnp
from ml_collections.config_dict import config_dict


FAKE_MODEL_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=16,
    num_layers=18,  # Must be one of [18, 34, 50, 101, 152, 200]
    layer_rescale_factors={},
    lr_hparams={
        'batch_size': 128,
        'base_lr': 10.0,
        'decay_end': -1,
        'end_lr': 1e-4,
        'power': 2.0,
        'schedule': 'mlperf_polynomial',
        'start_lr': 0.0,
        'steps_per_epoch': 10009.250000000002,
        'warmup_steps': 18,
    },
    optimizer='mlperf_lars_resnet',
    opt_hparams={
        'weight_decay': 2e-4,
        'beta': 0.9
    },
    batch_size=128,
    l2_decay_factor=None,
    l2_decay_rank_threshold=2,
    label_smoothing=.1,
    use_shallue_label_smoothing=False,
    model_dtype='float32',
    virtual_batch_size=64,
    data_format='NHWC',
    activation_function='relu',
    grad_clip=None,
    dropout_rate=0.0,
))


# Used for the mlperf version of Resnet.
MLPERF_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=16,
    # We set default to 18 for faster unit tests.
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
    bn_output_scale=0.0,
    l2_decay_factor=None,
    l2_decay_rank_threshold=2,
    label_smoothing=None,
    rng_seed=-1,
    use_shallue_label_smoothing=False,
    batch_norm_momentum=0.9,
    batch_norm_epsilon=1e-5,
    model_dtype='float32',
    virtual_batch_size=64,
    total_accumulated_batch_size=None,
    data_format='NHWC',
    activation_function='relu',
    grad_clip=None,
    dropout_rate=0.0,
))


def _constant_init(factor):
  def init_fn(key, shape, dtype=jnp.float32):
    del key
    return jnp.ones(shape, dtype) * factor
  return init_fn


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  dtype: model_utils.Dtype = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  bn_output_scale: float = 0.0
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None
  activation_function: Optional[str] = 'relu'

  @nn.compact
  def __call__(self, x, train):
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    batch_norm = functools.partial(
        normalization.VirtualBatchNorm,
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        dtype=self.dtype,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size,
        data_format=self.data_format)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    residual = x
    if needs_projection:
      residual = conv(
          self.filters * 4, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(
          residual, use_running_average=not train)
    y = conv(self.filters, (1, 1), name='conv1')(x)
    y = batch_norm(name='bn1')(y, use_running_average=not train)
    activation_fn = model_utils.ACTIVATIONS[self.activation_function]
    y = activation_fn(y)
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = batch_norm(name='bn2')(y, use_running_average=not train)
    y = activation_fn(y)
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)
    y = batch_norm(
        name='bn3', scale_init=_constant_init(self.bn_output_scale))(
            y, use_running_average=not train)
    y = activation_fn(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""
  num_classes: int
  num_filters: int = 64
  num_layers: int = 50
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  dtype: model_utils.Dtype = jnp.float32
  batch_norm_momentum: float = 0.9
  batch_norm_epsilon: float = 1e-5
  bn_output_scale: float = 0.0
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None
  data_format: Optional[str] = None
  activation_function: Optional[str] = 'relu'
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, train):
    if self.num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[self.num_layers]
    conv = functools.partial(nn.Conv, padding=[(3, 3), (3, 3)])
    x = conv(self.num_filters, kernel_size=(7, 7), strides=(2, 2),
             use_bias=False, dtype=self.dtype, name='conv0')(x)
    x = normalization.VirtualBatchNorm(
        momentum=self.batch_norm_momentum,
        epsilon=self.batch_norm_epsilon,
        name='init_bn',
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        dtype=self.dtype,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size,
        data_format=self.data_format)(x, use_running_average=not train)
    x = model_utils.ACTIVATIONS[self.activation_function](x)  # MLperf-required
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(
            self.num_filters * 2 ** i,
            strides=strides,
            axis_name=self.axis_name,
            axis_index_groups=self.axis_index_groups,
            dtype=self.dtype,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_epsilon=self.batch_norm_epsilon,
            bn_output_scale=self.bn_output_scale,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            total_batch_size=self.total_batch_size,
            data_format=self.data_format,
            activation_function=self.activation_function,
            )(x, train=train)
    x = jnp.mean(x, axis=(1, 2))
    if self.dropout_rate > 0.0:
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(),
                 dtype=self.dtype)(x)
    return x

# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


class FakeResNet(nn.Module):
  """Minimal NN (for debugging) with the same signature as a ResNet."""
  num_classes: int
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Any] = None
  dtype: model_utils.Dtype = jnp.float32

  @nn.compact
  def __call__(self, x, train):
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        name='init_bn',
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        dtype=self.dtype)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(),
                 dtype=self.dtype)(x)
    return x


class ResnetModelMLPerf(base_model.BaseModel):
  """MLPerf ResNet."""

  def build_flax_module(self):
    return ResNet(
        num_classes=self.hps['output_shape'][-1],
        num_filters=self.hps.num_filters,
        num_layers=self.hps.num_layers,
        dtype=utils.dtype_from_str(self.hps.model_dtype),
        batch_norm_momentum=self.hps.batch_norm_momentum,
        batch_norm_epsilon=self.hps.batch_norm_epsilon,
        bn_output_scale=self.hps.bn_output_scale,
        batch_size=self.hps.batch_size,
        virtual_batch_size=self.hps.virtual_batch_size,
        total_batch_size=self.hps.total_accumulated_batch_size,
        data_format=self.hps.data_format,
        activation_function=self.hps.activation_function,
        dropout_rate=self.hps.dropout_rate)

  def get_fake_inputs(self, hps):
    """Helper method solely for purpose of initializing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs


class FakeModel(base_model.BaseModel):
  """Fake Model for easy debugging."""

  def build_flax_module(self):
    return FakeResNet(num_classes=self.hps['output_shape'][-1])

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs
