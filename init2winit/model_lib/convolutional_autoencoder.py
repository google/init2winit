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

r"""Convolutional autoencoder.

This model uses a convolutional encoder-decoder network to reconstruct input
images as outputs.

"""

from typing import Any, Dict, Sequence

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils

from ml_collections.config_dict import config_dict

# small test hparams from
# https://blog.keras.io/building-autoencoders-in-keras.html
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        encoder={
            'filter_sizes': [16, 8, 8],
            'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
            'kernel_paddings': ['SAME', 'SAME', 'SAME'],
            'window_sizes': [(2, 2), (2, 2), (2, 2)],
            'window_paddings': ['SAME', 'SAME', 'SAME'],
            'strides': [(2, 2), (2, 2), (2, 2)],
            'activations': ['relu', 'relu', 'relu'],
        },
        decoder={
            'filter_sizes': [8, 8, 16, 1],
            'kernel_sizes': [(3, 3), (3, 3), (3, 3), (3, 3)],
            'window_sizes': [(2, 2), (2, 2), (2, 2), None],
            'paddings': ['SAME', ((1, 0), (1, 0)), 'SAME', 'SAME'],
            'activations': ['relu', 'relu', 'relu', 'id'],
        },

        activation_function='relu',
        lr_hparams={
            'base_lr': 0.02,
            'schedule': 'constant'
        },
        layer_rescale_factors={},
        optimizer='momentum',
        opt_hparams={
            'momentum': 0,
        },
        batch_size=128,
        l2_decay_factor=None,
        l2_decay_rank_threshold=0,
        label_smoothing=None,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
    ))


class ConvAutoEncoder(nn.Module):
  """Defines a fully connected neural network.

  The model assumes the input data has shape
  [batch_size_per_device, *input_shape] where input_shape may be of arbitrary
  rank. The model flatten the input before applying a dense layer.
  """
  output_shape: Sequence[int]
  encoder: Dict[str, Any]
  decoder: Dict[str, Any]

  @nn.compact
  def __call__(self, x, train):
    del train
    encoder_keys = [
        'filter_sizes',
        'kernel_sizes',
        'kernel_paddings',
        'window_sizes',
        'window_paddings',
        'strides',
        'activations',
    ]
    if len(set(len(self.encoder[k]) for k in encoder_keys)) > 1:
      raise ValueError(
          'The elements in encoder dict do not have the same length.')

    decoder_keys = [
        'filter_sizes',
        'kernel_sizes',
        'window_sizes',
        'paddings',
        'activations',
    ]
    if len(set(len(self.decoder[k]) for k in decoder_keys)) > 1:
      raise ValueError(
          'The elements in decoder dict do not have the same length.')

    # encoder
    for i in range(len(self.encoder['filter_sizes'])):
      x = nn.Conv(
          self.encoder['filter_sizes'][i],
          self.encoder['kernel_sizes'][i],
          padding=self.encoder['kernel_paddings'][i])(x)
      x = model_utils.ACTIVATIONS[self.encoder['activations'][i]](x)
      x = nn.max_pool(
          x, self.encoder['window_sizes'][i],
          strides=self.encoder['strides'][i],
          padding=self.encoder['window_paddings'][i])

    # decoder
    for i in range(len(self.decoder['filter_sizes'])):
      x = nn.ConvTranspose(
          self.decoder['filter_sizes'][i],
          self.decoder['kernel_sizes'][i],
          self.decoder['window_sizes'][i],
          padding=self.decoder['paddings'][i])(x)
      x = model_utils.ACTIVATIONS[self.decoder['activations'][i]](x)
    return x


class ConvAutoEncoderModel(base_model.BaseModel):

  def build_flax_module(self):
    return ConvAutoEncoder(
        output_shape=self.hps.output_shape,
        encoder=self.hps.encoder,
        decoder=self.hps.decoder)
