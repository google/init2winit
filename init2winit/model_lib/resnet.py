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

# Lint as: python3
"""Flax implementation of ResNet V1."""

from flax import nn
from init2winit import utils
from init2winit.model_lib import base_model
from init2winit.model_lib import normalization
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=16,
    num_layers=18,  # Must be one of [18, 34, 50, 101, 152, 200]
    layer_rescale_factors={},
    lr_hparams={
        'schedule': 'constant',
        'initial_value': 0.2,
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
    data_format='NHWC',
    grad_clip=None,
))


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(
      self,
      x,
      filters,
      strides=(1, 1),
      train=True,
      batch_stats=None,
      dtype=jnp.float32,
      batch_norm_momentum=0.9,
      batch_norm_epsilon=1e-5,
      virtual_batch_size=None,
      data_format=None):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = normalization.VirtualBatchNorm.partial(
        batch_stats=batch_stats,
        use_running_average=not train,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        dtype=dtype,
        virtual_batch_size=virtual_batch_size,
        data_format=data_format)
    conv = nn.Conv.partial(bias=False, dtype=dtype)

    residual = x
    if needs_projection:
      residual = conv(residual, filters * 4, (1, 1), strides, name='proj_conv')
      residual = batch_norm(residual, name='proj_bn')

    y = conv(x, filters, (1, 1), name='conv1')
    y = batch_norm(y, name='bn1')
    y = nn.relu(y)
    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = batch_norm(y, name='bn2')
    y = nn.relu(y)
    y = conv(y, filters * 4, (1, 1), name='conv3')

    y = batch_norm(y, name='bn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(
      self,
      x,
      num_outputs,
      num_filters=64,
      num_layers=50,
      train=True,
      batch_stats=None,
      dtype=jnp.float32,
      batch_norm_momentum=0.9,
      batch_norm_epsilon=1e-5,
      virtual_batch_size=None,
      data_format=None):
    if num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[num_layers]
    x = nn.Conv(x, num_filters, (7, 7), (2, 2),
                bias=False,
                dtype=dtype,
                name='init_conv')
    x = normalization.VirtualBatchNorm(
        x,
        batch_stats=batch_stats,
        use_running_average=not train,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon,
        dtype=dtype,
        name='init_bn',
        virtual_batch_size=virtual_batch_size,
        data_format=data_format)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(
            x,
            num_filters * 2 ** i,
            strides=strides,
            train=train,
            batch_stats=batch_stats,
            dtype=dtype,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
            virtual_batch_size=virtual_batch_size,
            data_format=data_format)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_outputs, dtype=dtype)
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
    200: [3, 24, 36, 3]
}


class ResnetModel(base_model.BaseModel):

  def build_flax_module(self):
    """ResnetV1."""
    return ResNet.partial(
        num_filters=self.hps.num_filters,
        num_layers=self.hps.num_layers,
        num_outputs=self.hps['output_shape'][-1],
        dtype=utils.dtype_from_str(self.hps.model_dtype),
        batch_norm_momentum=self.hps.batch_norm_momentum,
        batch_norm_epsilon=self.hps.batch_norm_epsilon,
        virtual_batch_size=self.hps.virtual_batch_size,
        data_format=self.hps.data_format)
