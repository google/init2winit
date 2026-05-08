# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

r"""Fully connected autoencoder.

This model builds an autoencoder using FullyConnected module.
More information on the fully connected autoencoder model can be found here:

https://www.cs.toronto.edu/~hinton/science.pdf

"""

from init2winit.model_lib import base_model
from init2winit.model_lib.fully_connected import FullyConnected
from jax.nn import initializers
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

# small test hparams
# https://blog.keras.io/building-autoencoders-in-keras.html
# for the configuration of a standard fully connected autoencoder model,
# see https://www.cs.toronto.edu/~hinton/science.pdf.
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        hid_sizes=[128, 64, 32, 64, 128],
        activation_function=['relu', 'relu', 'relu', 'relu', 'relu'],
        kernel_scales=[1.0] * 6,
        model_dtype='float32',
    ))


class AutoEncoderModel(base_model.BaseModel):
  """Model class for AutoEncoder model."""

  def build_flax_module(self):
    kernel_inits = [
        initializers.normal(scale)
        for scale in self.hps.kernel_scales
    ]

    return FullyConnected(
        num_outputs=self.hps['output_shape'][-1],
        hid_sizes=self.hps.hid_sizes,
        activation_function=self.hps.activation_function,
        kernel_inits=kernel_inits)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
        ]
    return dummy_inputs
