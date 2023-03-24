# coding=utf-8
# Copyright 2023 The init2winit Authors.
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

"""Simple fully connected feedforward neural network classifier."""
import copy
from typing import Any, Tuple

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax.nn import initializers
import jax.numpy as jnp
from ml_collections.config_dict import config_dict


# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        hid_sizes=[20, 10],
        kernel_scales=[1.0, 1.0, 1.0],
        lr_hparams={
            'base_lr': 0.1,
            'schedule': 'constant'
        },
        layer_rescale_factors={},
        optimizer='momentum',
        opt_hparams={
            'momentum': 0.9,
        },
        batch_size=128,
        total_accumulated_batch_size=None,
        activation_function='relu',
        l2_decay_factor=.0005,
        l2_decay_rank_threshold=2,
        label_smoothing=None,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
    ))


class FullyConnected(nn.Module):
  """Defines a fully connected neural network.

  The model assumes the input data has shape
  [batch_size_per_device, *input_shape] where input_shape may be of arbitrary
  rank. The model flatten the input before applying a dense layer.
  """
  num_outputs: int
  hid_sizes: Tuple[int]
  activation_function: Any
  kernel_inits: Tuple[model_utils.Initializer]
  bias_init: model_utils.Initializer = initializers.zeros

  @nn.compact
  def __call__(self, x, train):
    del train
    if not isinstance(self.activation_function, str):
      if len(self.activation_function) != len(self.hid_sizes):
        raise ValueError(
            'The number of activation functions must be equal to the number '
            'of hidden layers')
      activation_function = copy.deepcopy(self.activation_function)
    else:
      activation_function = [self.activation_function] * len(self.hid_sizes)

    x = jnp.reshape(x, (x.shape[0], -1))
    for i, (num_hid, init) in enumerate(
        zip(self.hid_sizes, self.kernel_inits[:-1])):
      x = nn.Dense(num_hid, kernel_init=init, bias_init=self.bias_init)(x)
      x = model_utils.ACTIVATIONS[activation_function[i]](x)
    x = nn.Dense(
        self.num_outputs,
        kernel_init=self.kernel_inits[-1],
        bias_init=self.bias_init)(x)
    return x


# pylint: disable=missing-class-docstring
class FullyConnectedModel(base_model.BaseModel):
  """Model class for fully connected model."""

  def build_flax_module(self):
    kernel_inits = [
        initializers.variance_scaling(scale, 'fan_in', 'truncated_normal')
        for scale in self.hps.kernel_scales
    ]
    return FullyConnected(
        num_outputs=self.hps['output_shape'][-1],
        hid_sizes=tuple(self.hps.hid_sizes),
        activation_function=self.hps.activation_function,
        kernel_inits=tuple(kernel_inits))

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs
