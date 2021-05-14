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

"""Max pooling convnet classifier.

This model can be used to implement the 3c3d architecture from:
https://github.com/fsschneider/DeepOBS/blob/master/deepobs/tensorflow/testproblems/_3c3d.py
"""

from flax import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax.nn import initializers
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=[64, 96, 128],
    kernel_sizes=[5, 3, 3],
    kernel_paddings=['VALID', 'VALID', 'SAME'],
    window_sizes=[3, 3, 3],
    window_paddings=['SAME', 'SAME', 'SAME'],
    strides=[2, 2, 2],
    num_dense_units=[512, 256],
    lr_hparams={
        'initial_value': 0.001,
        'schedule': 'constant'
    },
    layer_rescale_factors={},
    optimizer='momentum',
    opt_hparams={
        'momentum': 0.9,
    },
    batch_size=128,
    activation_fn='relu',
    normalizer='none',
    l2_decay_factor=.0005,
    l2_decay_rank_threshold=2,
    label_smoothing=None,
    rng_seed=-1,
    use_shallue_label_smoothing=False,
    model_dtype='float32',
))


class MaxPoolingCNN(nn.Module):
  """Defines a CNN model with max pooling.

  The model assumes the input shape is [batch, H, W, C].
  """

  def apply(self,
            x,
            num_outputs,
            num_filters,
            kernel_sizes,
            kernel_paddings,
            window_sizes,
            window_paddings,
            strides,
            num_dense_units,
            activation_fn,
            normalizer='none',
            kernel_init=initializers.lecun_normal(),
            bias_init=initializers.zeros,
            train=True):

    maybe_normalize = model_utils.get_normalizer(normalizer, train)
    for num_filters, kernel_size, kernel_padding, window_size, window_padding, stride in zip(
        num_filters, kernel_sizes, kernel_paddings, window_sizes,
        window_paddings, strides):
      x = nn.Conv(
          x,
          num_filters, (kernel_size, kernel_size), (1, 1),
          padding=kernel_padding,
          kernel_init=kernel_init,
          bias_init=bias_init)
      x = model_utils.ACTIVATIONS[activation_fn](x)
      x = maybe_normalize(x)
      x = nn.max_pool(
          x,
          window_shape=(window_size, window_size),
          strides=(stride, stride),
          padding=window_padding)
    x = jnp.reshape(x, (x.shape[0], -1))
    for num_units in num_dense_units:
      x = nn.Dense(
          x, num_units, kernel_init=kernel_init, bias_init=bias_init)
      x = model_utils.ACTIVATIONS[activation_fn](x)
      x = maybe_normalize(x)
    x = nn.Dense(x, num_outputs, kernel_init=kernel_init, bias_init=bias_init)
    return x


class MaxPoolingCNNModel(base_model.BaseModel):

  def build_flax_module(self):
    """CNN with a set of conv layers with max pooling followed by fully connected layers."""
    return MaxPoolingCNN.partial(
        num_outputs=self.hps['output_shape'][-1],
        num_filters=self.hps.num_filters,
        kernel_sizes=self.hps.kernel_sizes,
        kernel_paddings=self.hps.kernel_paddings,
        window_sizes=self.hps.window_sizes,
        window_paddings=self.hps.window_paddings,
        strides=self.hps.strides,
        num_dense_units=self.hps.num_dense_units,
        activation_fn=self.hps.activation_fn,
        normalizer=self.hps.normalizer,)
