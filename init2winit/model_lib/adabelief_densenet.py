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

# Lint as: python3
"""Flax implementation of Adabelief DenseNet.

This module ports the Adabelief implemetation of DenseNet to Flax.  The
Adabelief paper and github can be found here:

https://arxiv.org/abs/2010.07468

https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/PyTorch_Experiments/classification_cifar10/models/densenet.py

The original DenseNet paper can be found here:

https://arxiv.org/abs/1608.06993?source=post_page---------------------------

"""

import functools
import math

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        num_layers=121,  # Must be one of [121, 169, 201, 161]
        growth_rate=32,
        reduction=0.5,
        # Set this to True to replicate the DenseNet behavior.  Setting it to
        # False results in stride 1 being used in the pooling layers. This
        # results in a large Dense matrix in the readout layer and unstable
        # training.
        use_kernel_size_as_stride_in_pooling=True,
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
        normalize_classifier_input='none',
        classification_scale_factor=1.0,
    ))


class BottleneckBlock(nn.Module):
  """Composition function with a bottleneck layer.

  The composition function, which is the basic unit of a dense block, conists
  of a batch normalization, followed by a ReLU and 3x3 convolution.  In order
  to reduce the number of input feature maps, the above operations are
  preceded by 1x1 convoluational operation (and the correpsonding batch
  normalization and ReLU).
  """
  growth_rate: int
  dtype: model_utils.Dtype = jnp.float32
  normalizer: str = 'batch_norm'

  @nn.compact
  def __call__(self, x, train):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    maybe_normalize = model_utils.get_normalizer(self.normalizer, train)

    y = maybe_normalize()(x)
    y = nn.relu(y)
    y = conv(features=4 * self.growth_rate, kernel_size=(1, 1), name='conv1')(y)

    y = maybe_normalize()(y)
    y = nn.relu(y)
    y = conv(
        features=self.growth_rate,
        kernel_size=(3, 3),
        padding=((1, 1), (1, 1)),
        name='conv2')(y)

    # Concatenate the output and input along the features dimension.
    y = jnp.concatenate([y, x], axis=3)
    return y


class TransitionBlock(nn.Module):
  """Block that implements downsampling between dense layers.

  Downsampling is achieved by a 1x1 convoluationl layer (with the associated
  batch norm and ReLU) and a 2x2 average pooling layer.
  """
  num_features: int
  use_kernel_size_as_stride_in_pooling: bool
  dtype: model_utils.Dtype = jnp.float32
  normalizer: str = 'batch_norm'

  @nn.compact
  def __call__(self, x, train):
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    maybe_normalize = model_utils.get_normalizer(self.normalizer, train)

    y = maybe_normalize()(x)
    y = nn.relu(y)
    y = conv(features=self.num_features, kernel_size=(1, 1))(y)
    y = nn.avg_pool(
        y,
        window_shape=(2, 2),
        strides=(2, 2) if self.use_kernel_size_as_stride_in_pooling else (1, 1))
    return y


class DenseNet(nn.Module):
  """Adabelief DenseNet.

  The network consists of an inital convolutaional layer, four dense blocks
  connected by transition blocks, a pooling layer and a classification layer.
  """
  num_layers: int
  num_outputs: int
  growth_rate: int
  reduction: int
  use_kernel_size_as_stride_in_pooling: bool
  normalizer: str = 'batch_norm'
  normalize_classifier_input: bool = False
  classification_scale_factor: float = 1.0
  dtype: model_utils.Dtype = jnp.float32

  @nn.compact
  def __call__(self, x, train):
    def dense_layers(y, block, num_blocks, growth_rate):
      for _ in range(num_blocks):
        y = block(growth_rate)(y, train=train)
      return y

    def update_num_features(num_features, num_blocks, growth_rate, reduction):
      num_features += num_blocks * growth_rate
      if reduction is not None:
        num_features = int(math.floor(num_features * reduction))
      return num_features

    # Initial convolutional layer
    num_features = 2 * self.growth_rate
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    y = conv(
        features=num_features,
        kernel_size=(3, 3),
        padding=((1, 1), (1, 1)),
        name='conv1')(x)

    # Internal dense and transtion blocks
    num_blocks = _block_size_options[self.num_layers]
    block = functools.partial(
        BottleneckBlock,
        dtype=self.dtype,
        normalizer=self.normalizer)
    for i in range(3):
      y = dense_layers(y, block, num_blocks[i], self.growth_rate)
      num_features = update_num_features(num_features, num_blocks[i],
                                         self.growth_rate, self.reduction)
      y = TransitionBlock(
          num_features,
          dtype=self.dtype,
          normalizer=self.normalizer,
          use_kernel_size_as_stride_in_pooling=self
          .use_kernel_size_as_stride_in_pooling)(
              y, train=train)

    # Final dense block
    y = dense_layers(y, block, num_blocks[3], self.growth_rate)

    # Final pooling
    maybe_normalize = model_utils.get_normalizer(self.normalizer, train)
    y = maybe_normalize()(y)
    y = nn.relu(y)
    y = nn.avg_pool(
        y,
        window_shape=(4, 4),
        strides=(4, 4) if self.use_kernel_size_as_stride_in_pooling else (1, 1))

    # Classification layer
    y = jnp.reshape(y, (y.shape[0], -1))
    if self.normalize_classifier_input:
      maybe_normalize = model_utils.get_normalizer(
          self.normalize_classifier_input, train)
      y = maybe_normalize()(y)
    y = y * self.classification_scale_factor

    y = nn.Dense(self.num_outputs)(y)
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
    return DenseNet(
        num_layers=self.hps.num_layers,
        num_outputs=self.hps['output_shape'][-1],
        growth_rate=self.hps.growth_rate,
        reduction=self.hps.reduction,
        use_kernel_size_as_stride_in_pooling=self.hps
        .use_kernel_size_as_stride_in_pooling,
        dtype=self.hps.model_dtype,
        normalizer=self.hps.normalizer,
        normalize_classifier_input=self.hps.normalize_classifier_input,
        classification_scale_factor=self.hps.classification_scale_factor,
    )
