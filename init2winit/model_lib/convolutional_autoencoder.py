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

from flax import nn
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
            'initial_value': 0.02,
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

  def apply(self,
            x,
            output_shape,
            encoder,
            decoder,
            train=True):
    if not len(encoder['filter_sizes']) == len(encoder['kernel_sizes']) == len(
        encoder['kernel_paddings']) == len(encoder['window_sizes']) == len(
            encoder['window_paddings']) == len(encoder['strides']) == len(
                encoder['activations']):
      raise ValueError(
          'The elements in encoder dict do not have the same length.')

    if not len(decoder['filter_sizes']) == len(decoder['kernel_sizes']) == len(
        decoder['window_sizes']) == len(decoder['paddings']) == len(
            decoder['activations']):
      raise ValueError(
          'The elements in decoder dict do not have the same length.')

    # encoder
    for i in range(len(encoder['filter_sizes'])):
      x = nn.Conv(
          x,
          encoder['filter_sizes'][i],
          encoder['kernel_sizes'][i],
          padding=encoder['kernel_paddings'][i])
      x = model_utils.ACTIVATIONS[encoder['activations'][i]](x)
      x = nn.max_pool(
          x, encoder['window_sizes'][i],
          strides=encoder['strides'][i],
          padding=encoder['window_paddings'][i])

    # decoder
    for i in range(len(decoder['filter_sizes'])):
      x = nn.ConvTranspose(
          x,
          decoder['filter_sizes'][i],
          decoder['kernel_sizes'][i],
          decoder['window_sizes'][i],
          padding=decoder['paddings'][i])
      x = model_utils.ACTIVATIONS[decoder['activations'][i]](x)
    return x


class ConvAutoEncoderModel(base_model.BaseModel):

  def build_flax_module(self):
    return ConvAutoEncoder.partial(
        output_shape=self.hps.output_shape,
        encoder=self.hps.encoder,
        decoder=self.hps.decoder)
