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
"""Flax implementation of Adabelief DenseNet.

This module ports the Adabelief implemetation of DenseNet to Flax.  The
Adabelief paper and github can be found here:

https://arxiv.org/abs/2010.07468

https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/PyTorch_Experiments/classification_cifar10/models/densenet.py

The original DenseNet paper can be found here:

https://arxiv.org/abs/1608.06993?source=post_page---------------------------

"""

import math

from flax.deprecated import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        num_layers=121,  # Must be one of [121, 169, 201, 161]
        growth_rate=32,
        reduction=0.5,
        layer_rescale_factors={},
        lr_hparams={
            'schedule': 'constant',
            'base_lr': 0.2,
        },
        normalizer='batch_norm',
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
    ))


class BottleneckBlock(nn.Module):
  """Composition function with a bottleneck layer.

  The composition function, which is the basic unit of a dense block, conists
  of a batch normalization, followed by a ReLU and 3x3 convolution.  In order
  to reduce the number of input feature maps, the above operations are
  preceded by 1x1 convoluational operation (and the correpsonding batch
  normalization and ReLU).
  """

  def apply(self,
            x,
            growth_rate,
            train=True,
            dtype=jnp.float32,
            normalizer='batch_norm'):
    conv = nn.Conv.partial(bias=False, dtype=dtype)
    maybe_normalize = model_utils.get_normalizer(normalizer, train)

    y = maybe_normalize(x)
    y = nn.relu(y)
    y = conv(y, features=4 * growth_rate, kernel_size=(1, 1), name='conv1')

    y = maybe_normalize(y)
    y = nn.relu(y)
    y = conv(
        y,
        features=growth_rate,
        kernel_size=(3, 3),
        padding=((1, 1), (1, 1)),
        name='conv2')

    # Concatenate the output and input along the features dimension.
    y = jnp.concatenate([y, x], axis=3)
    return y


class TransitionBlock(nn.Module):
  """Block that implements downsampling between dense layers.

  Downsampling is achieved by a 1x1 convoluationl layer (with the associated
  batch norm and ReLU) and a 2x2 average pooling layer.
  """

  def apply(self,
            x,
            num_features,
            train=True,
            dtype=jnp.float32,
            normalizer='batch_norm'):
    conv = nn.Conv.partial(bias=False, dtype=dtype)
    maybe_normalize = model_utils.get_normalizer(normalizer, train)

    y = maybe_normalize(x)
    y = nn.relu(y)
    y = conv(y, features=num_features, kernel_size=(1, 1))
    y = nn.avg_pool(y, window_shape=(2, 2))
    return y


class DenseNet(nn.Module):
  """Adabelief DenseNet.

  The network consists of an inital convolutaional layer, four dense blocks
  connected by transition blocks, a pooling layer and a classification layer.
  """

  def apply(self,
            x,
            num_layers,
            num_outputs,
            growth_rate,
            reduction,
            normalizer='batch_norm',
            dtype='float32',
            train=True):

    def dense_layers(y, block, num_blocks, growth_rate):
      for _ in range(num_blocks):
        y = block(y, growth_rate)
      return y

    def update_num_features(num_features, num_blocks, growth_rate, reduction):
      num_features += num_blocks * growth_rate
      if reduction is not None:
        num_features = int(math.floor(num_features * reduction))
      return num_features

    # Initial convolutional layer
    num_features = 2 * growth_rate
    conv = nn.Conv.partial(bias=False, dtype=dtype)
    y = conv(
        x,
        features=num_features,
        kernel_size=(3, 3),
        padding=((1, 1), (1, 1)),
        name='conv1')

    # Internal dense and transtion blocks
    num_blocks = _block_size_options[num_layers]
    block = BottleneckBlock.partial(
        train=train, dtype=dtype, normalizer=normalizer)
    for i in range(3):
      y = dense_layers(y, block, num_blocks[i], growth_rate)
      num_features = update_num_features(num_features, num_blocks[i],
                                         growth_rate, reduction)
      y = TransitionBlock(
          y, num_features, train=train, dtype=dtype, normalizer=normalizer)

    # Final dense block
    y = dense_layers(y, block, num_blocks[3], growth_rate)

    # Final pooling
    maybe_normalize = model_utils.get_normalizer(normalizer, train)
    y = maybe_normalize(y)
    y = nn.relu(y)
    y = nn.avg_pool(y, window_shape=(4, 4))

    # Classification layer
    y = jnp.reshape(y, (y.shape[0], -1))
    y = nn.Dense(y, num_outputs)
    return y


# A dictionary mapping the number of layers in a densenet to the number of
# basic units in each dense block of the model.
_block_size_options = {
    1: [1, 1, 1, 1],  # used for testing only
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    161: [6, 12, 32, 24],
}


class AdaBeliefDensenetModel(base_model.BaseModel):

  def build_flax_module(self):
    """Adabelief DenseNet."""
    return DenseNet.partial(
        num_layers=self.hps.num_layers,
        num_outputs=self.hps['output_shape'][-1],
        growth_rate=self.hps.growth_rate,
        reduction=self.hps.reduction,
        dtype=self.hps.model_dtype,
        normalizer=self.hps.normalizer)
