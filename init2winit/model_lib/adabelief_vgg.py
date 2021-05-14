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
"""Flax implementation of Adabelief VGG.

This module ports the Adabelief implemetation of VGG to Flax.  The
Adabelief paper and github can be found here:

https://arxiv.org/abs/2010.07468

https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/PyTorch_Experiments/classification_cifar10/models/vgg.py

The original VGGNet paper can be found here:

https://arxiv.org/abs/1409.1556
"""

from flax import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        num_layers=11,  # Must be one of [11, 13, 16, 19]
        layer_rescale_factors={},
        lr_hparams={
            'schedule': 'constant',
            'initial_value': 0.2,
        },
        normalizer='none',
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
    ))


def classifier(x, num_outputs, dropout_rate, deterministic):
  """Implements the classification portion of the network."""

  x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
  x = nn.Dense(x, 512)
  x = nn.relu(x)
  x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
  x = nn.Dense(x, 512)
  x = nn.relu(x)
  x = nn.Dense(x, num_outputs)
  return x


def features(x, num_layers, normalizer, dtype, train):
  """Implements the feature extraction portion of the network."""

  layers = _layer_size_options[num_layers]
  conv = nn.Conv.partial(bias=False, dtype=dtype)
  maybe_normalize = model_utils.get_normalizer(normalizer, train)
  for l in layers:
    if l == 'M':
      x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    else:
      x = conv(x, features=l, kernel_size=(3, 3), padding=((1, 1), (1, 1)))
      x = maybe_normalize(x)
      x = nn.relu(x)
  return x


class VGG(nn.Module):
  """Adabelief VGG."""

  def apply(self,
            x,
            num_layers,
            num_outputs,
            normalizer='none',
            dtype='float32',
            train=True):
    x = features(x, num_layers, normalizer, dtype, train)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = classifier(x, num_outputs, dropout_rate=0.5, deterministic=not train)
    return x


# Specifies the sequence of layers in the feature extraction section of the
# network for a given size.
# The numbers indicate the feature size of a convolutional layer, the
# letter M indicates a max pooling layer.
_layer_size_options = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [
        64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ],
    16: [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    19: [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


class AdaBeliefVGGModel(base_model.BaseModel):

  def build_flax_module(self):
    """Adabelief VGG."""
    return VGG.partial(
        num_layers=self.hps.num_layers,
        num_outputs=self.hps['output_shape'][-1],
        dtype=self.hps.model_dtype,
        normalizer=self.hps.normalizer)
