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

"""Wide Resnet Model."""

from flax.deprecated import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax.nn import initializers

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
        l2_decay_factor=0.0001,
        l2_decay_rank_threshold=2,
        label_smoothing=None,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
        activation_function='relu'))


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock."""

  def apply(self,
            x,
            channels,
            strides=(1, 1),
            conv_kernel_init=initializers.lecun_normal(),
            normalizer='batch_norm',
            dropout_rate=0.0,
            activation_function='relu',
            train=True):
    maybe_normalize = model_utils.get_normalizer(normalizer, train)
    y = maybe_normalize(x, name='bn1')
    y = model_utils.ACTIVATIONS[activation_function](y)

    # Apply an up projection in case of channel mismatch
    if (x.shape[-1] != channels) or strides != (1, 1):
      x = nn.Conv(
          y,
          channels,
          (1, 1),  # Note: Some implementations use (3, 3) here.
          strides,
          padding='SAME',
          kernel_init=conv_kernel_init,
          bias=False)

    y = nn.Conv(
        y,
        channels,
        (3, 3),
        strides,
        padding='SAME',
        name='conv1',
        kernel_init=conv_kernel_init,
        bias=False)
    y = nn.dropout(y, rate=dropout_rate, deterministic=not train)
    y = maybe_normalize(y, name='bn2')
    y = model_utils.ACTIVATIONS[activation_function](y)
    y = nn.Conv(
        y,
        channels,
        (3, 3),
        padding='SAME',
        name='conv2',
        kernel_init=conv_kernel_init,
        bias=False)

    if normalizer == 'none':
      y = model_utils.ScalarMultiply(y)

    return x + y


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            conv_kernel_init=initializers.lecun_normal(),
            normalizer='batch_norm',
            dropout_rate=0.0,
            activation_function='relu',
            train=True):
    for i in range(blocks_per_group):
      x = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          conv_kernel_init=conv_kernel_init,
          normalizer=normalizer,
          dropout_rate=dropout_rate,
          activation_function=activation_function,
          train=train)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""

  def apply(
      self,
      x,
      blocks_per_group,
      channel_multiplier,
      num_outputs,
      conv_kernel_init=initializers.lecun_normal(),
      dense_kernel_init=initializers.lecun_normal(),
      normalizer='batch_norm',
      dropout_rate=0.0,
      activation_function='relu',
      train=True,
  ):

    x = nn.Conv(
        x,
        16,
        (3, 3),
        padding='SAME',
        name='init_conv',
        kernel_init=conv_kernel_init,
        bias=False)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        conv_kernel_init=conv_kernel_init,
        normalizer=normalizer,
        dropout_rate=dropout_rate,
        activation_function=activation_function,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        conv_kernel_init=conv_kernel_init,
        normalizer=normalizer,
        dropout_rate=dropout_rate,
        activation_function=activation_function,
        train=train)
    x = WideResnetGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        conv_kernel_init=conv_kernel_init,
        dropout_rate=dropout_rate,
        normalizer=normalizer,
        activation_function=activation_function,
        train=train)
    maybe_normalize = model_utils.get_normalizer(normalizer, train)
    x = maybe_normalize(x)
    x = model_utils.ACTIVATIONS[activation_function](x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_outputs, kernel_init=dense_kernel_init)
    return x


class WideResnetModel(base_model.BaseModel):

  def build_flax_module(self):
    """Wide Resnet."""
    hps = self.hps
    return WideResnet.partial(
        blocks_per_group=hps.blocks_per_group,
        channel_multiplier=hps.channel_multiplier,
        num_outputs=self.hps['output_shape'][-1],
        conv_kernel_init=model_utils.INITIALIZERS[hps.conv_kernel_init](
            hps.conv_kernel_scale),
        dense_kernel_init=model_utils.INITIALIZERS[hps.dense_kernel_init](
            hps.dense_kernel_scale),
        dropout_rate=hps.dropout_rate,
        normalizer=self.hps.normalizer,
        activation_function=self.hps.activation_function)
