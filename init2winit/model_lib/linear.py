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

"""Simple fully connected feedforward neural network classifier."""

from flax import nn
from init2winit.model_lib import base_model
from jax.nn import initializers
import jax.numpy as jnp

from ml_collections.config_dict import config_dict


# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    optimizer='sgd',
    batch_size=128,
    lr_hparams={
        'base_lr': 0.01,
        'schedule': 'constant'},
    label_smoothing=None,
    l2_decay_factor=None,
    l2_decay_rank_threshold=0,
    rng_seed=-1,
    use_shallue_label_smoothing=False,
    model_dtype='float32',
    grad_clip=None,
))


class Linear(nn.Module):
  """Defines a Linear model.

  The model assumes the input data has shape
  [batch_size_per_device, *input_shape] where input_shape may be of arbitrary
  rank. The model flatten the input before applying a dense layer.
  """

  def apply(self,
            x,
            kernel_init=initializers.zeros,
            bias_init=initializers.zeros,
            train=True):
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(
        x, 1, kernel_init=kernel_init, bias_init=bias_init)
    return x


class LinearModel(base_model.BaseModel):

  def build_flax_module(self):
    return Linear.partial()
