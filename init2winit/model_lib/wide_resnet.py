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

"""Wide Resnet Model."""
from typing import List, Optional, Tuple

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax.nn import initializers
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        blocks_per_group=3,
        channel_multiplier=2,
        lr_hparams={
            'base_lr': 0.001,
            'schedule': 'cosine'
        },
        normalizer='batch_norm',
        layer_rescale_factors={},
        conv_kernel_scale=1.0,
        dense_kernel_scale=1.0,
        dropout_rate=0.0,
        conv_kernel_init='lecun_normal',
        dense_kernel_init='lecun_normal',
        optimizer='momentum',
        opt_hparams={
            'momentum': 0.9,
        },
        batch_size=128,
        virtual_batch_size=None,
        total_accumulated_batch_size=None,
        l2_decay_factor=0.0001,
        l2_decay_rank_threshold=2,
        label_smoothing=None,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
        activation_function='relu',
        group_strides=[(1, 1), (2, 2), (2, 2)])
)


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock."""
  channels: int
  strides: List[Tuple[int]]
  conv_kernel_init: model_utils.Initializer = initializers.lecun_normal()
  normalizer: str = 'batch_norm'
  dropout_rate: float = 0.0
  activation_function: str = 'relu'
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None

  @nn.compact
  def __call__(self, x, train):
    maybe_normalize = model_utils.get_normalizer(
        self.normalizer,
        train,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size)
    y = maybe_normalize(name='bn1')(x)
    y = model_utils.ACTIVATIONS[self.activation_function](y)

    # Apply an up projection in case of channel mismatch
    if (x.shape[-1] != self.channels) or self.strides != (1, 1):
      x = nn.Conv(
          self.channels,
          (1, 1),  # Note: Some implementations use (3, 3) here.
          self.strides,
          padding='SAME',
          kernel_init=self.conv_kernel_init,
          use_bias=False)(y)

    y = nn.Conv(
        self.channels,
        (3, 3),
        self.strides,
        padding='SAME',
        name='conv1',
        kernel_init=self.conv_kernel_init,
        use_bias=False)(y)
    y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
    y = maybe_normalize(name='bn2')(y)
    y = model_utils.ACTIVATIONS[self.activation_function](y)
    y = nn.Conv(
        self.channels,
        (3, 3),
        padding='SAME',
        name='conv2',
        kernel_init=self.conv_kernel_init,
        use_bias=False)(y)

    if self.normalizer == 'none':
      y = model_utils.ScalarMultiply()(y)

    return x + y


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""
  blocks_per_group: int
  channels: int
  strides: Tuple[int, int] = (1, 1)
  conv_kernel_init: model_utils.Initializer = initializers.lecun_normal()
  normalizer: str = 'batch_norm'
  dropout_rate: float = 0.0
  activation_function: str = 'relu'
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None

  @nn.compact
  def __call__(self, x, train):
    for i in range(self.blocks_per_group):
      x = WideResnetBlock(
          self.channels,
          self.strides if i == 0 else (1, 1),
          conv_kernel_init=self.conv_kernel_init,
          normalizer=self.normalizer,
          dropout_rate=self.dropout_rate,
          activation_function=self.activation_function,
          batch_size=self.batch_size,
          virtual_batch_size=self.virtual_batch_size,
          total_batch_size=self.total_batch_size)(x, train=train)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""
  blocks_per_group: int
  channel_multiplier: int
  group_strides: List[Tuple[int]]
  num_outputs: int
  conv_kernel_init: model_utils.Initializer = initializers.lecun_normal()
  dense_kernel_init: model_utils.Initializer = initializers.lecun_normal()
  normalizer: str = 'batch_norm'
  dropout_rate: float = 0.0
  activation_function: str = 'relu'
  batch_size: Optional[int] = None
  virtual_batch_size: Optional[int] = None
  total_batch_size: Optional[int] = None

  @nn.compact
  def __call__(self, x, train):
    x = nn.Conv(
        16,
        (3, 3),
        padding='SAME',
        name='init_conv',
        kernel_init=self.conv_kernel_init,
        use_bias=False)(x)
    x = WideResnetGroup(
        self.blocks_per_group,
        16 * self.channel_multiplier,
        self.group_strides[0],
        conv_kernel_init=self.conv_kernel_init,
        normalizer=self.normalizer,
        dropout_rate=self.dropout_rate,
        activation_function=self.activation_function,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size)(x, train=train)
    x = WideResnetGroup(
        self.blocks_per_group,
        32 * self.channel_multiplier,
        self.group_strides[1],
        conv_kernel_init=self.conv_kernel_init,
        normalizer=self.normalizer,
        dropout_rate=self.dropout_rate,
        activation_function=self.activation_function,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size)(x, train=train)
    x = WideResnetGroup(
        self.blocks_per_group,
        64 * self.channel_multiplier,
        self.group_strides[2],
        conv_kernel_init=self.conv_kernel_init,
        dropout_rate=self.dropout_rate,
        normalizer=self.normalizer,
        activation_function=self.activation_function,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size)(x, train=train)
    maybe_normalize = model_utils.get_normalizer(
        self.normalizer,
        train,
        batch_size=self.batch_size,
        virtual_batch_size=self.virtual_batch_size,
        total_batch_size=self.total_batch_size)
    x = maybe_normalize()(x)
    x = model_utils.ACTIVATIONS[self.activation_function](x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(self.num_outputs, kernel_init=self.dense_kernel_init)(x)
    return x


class WideResnetModel(base_model.BaseModel):
  """Model class for wide Resnet Model."""

  def build_flax_module(self):
    """Wide Resnet."""
    return WideResnet(
        blocks_per_group=self.hps.blocks_per_group,
        channel_multiplier=self.hps.channel_multiplier,
        group_strides=self.hps.group_strides,
        num_outputs=self.hps['output_shape'][-1],
        conv_kernel_init=model_utils.INITIALIZERS[self.hps.conv_kernel_init](
            self.hps.conv_kernel_scale),
        dense_kernel_init=model_utils.INITIALIZERS[self.hps.dense_kernel_init](
            self.hps.dense_kernel_scale),
        dropout_rate=self.hps.dropout_rate,
        normalizer=self.hps.normalizer,
        activation_function=self.hps.activation_function,
        batch_size=self.hps.batch_size,
        virtual_batch_size=self.hps.virtual_batch_size,
        total_batch_size=self.hps.total_accumulated_batch_size)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs
