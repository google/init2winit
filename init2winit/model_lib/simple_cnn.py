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

"""Simple convnet classifier."""

from flax import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax.nn import initializers
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    num_filters=[20, 10],
    kernel_sizes=[3, 3],
    lr_hparams={
        'base_lr': 0.001,
        'schedule': 'constant'
    },
    layer_rescale_factors={},
    optimizer='momentum',
    opt_hparams={
        'momentum': 0.9,
    },
    batch_size=128,
    activation_function='relu',
    l2_decay_factor=.0005,
    l2_decay_rank_threshold=2,
    label_smoothing=None,
    rng_seed=-1,
    use_shallue_label_smoothing=False,
    model_dtype='float32',
))


class SimpleCNN(nn.Module):
  """Defines a simple CNN model.

  The model assumes the input shape is [batch, H, W, C].
  """

  def apply(self,
            x,
            num_outputs,
            num_filters,
            kernel_sizes,
            activation_function,
            kernel_init=initializers.lecun_normal(),
            bias_init=initializers.zeros,
            train=True):

    for num_filters, kernel_size in zip(num_filters, kernel_sizes):
      x = nn.Conv(
          x,
          num_filters, (kernel_size, kernel_size), (1, 1),
          kernel_init=kernel_init,
          bias_init=bias_init)
      x = model_utils.ACTIVATIONS[activation_function](x)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(x, num_outputs, kernel_init=kernel_init, bias_init=bias_init)
    return x


class SimpleCNNModel(base_model.BaseModel):

  def build_flax_module(self):
    """Simple CNN with a set of conv layers followed by fully connected layers."""
    return SimpleCNN.partial(
        num_outputs=self.hps['output_shape'][-1],
        num_filters=self.hps.num_filters,
        kernel_sizes=self.hps.kernel_sizes,
        activation_function=self.hps.activation_function)
