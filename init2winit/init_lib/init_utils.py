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

"""Utility functions related to initialization."""
import copy
import functools

from init2winit import utils
from init2winit.model_lib import model_utils
import jax
import numpy as np


def initialize(model, initializer, hps, rng, metrics_logger):
  """Run the given initializer.

  We initialize in 3 phases. First we run the default initializer that is
  specified by the model constructor. Next we apply any rescaling as specified
  by hps.layer_rescale_factors. Finally we run the black box initializer
  provided by the initializer arg (the default is noop).

  Args:
    model: Model subclass of i2w parent BaseModel class type.
    initializer: An initializer defined in init_lib.
    hps: A dictionary specifying the model and initializer hparams.
    rng: An rng key to seed the initialization.
    metrics_logger: Used for black box initializers that have learning curves.

  Returns:
    A tuple (model, batch_stats), where model is the initialized
    flax.nn.Model and batch_stats is the collection used for batch norm.
  """
  model_dtype = utils.dtype_from_str(hps.model_dtype)
  # Note that this fake input batch will be optimized away because the init
  # function is jitted. However, this can still cause memory issues if it is
  # large because it is passed in as an XLA argument. Therefore we use a fake
  # batch size of 2 (we do not want to use 1 in case there is any np.squeeze
  # calls that would remove it), because we assume that there is no dependence
  # on the batch size with the model (batch norm reduces across a batch dim of
  # any size). This is similar to how the Flax examples initialize models:
  # https://github.com/google/flax/blob/44ee6f2f4130856d47159dc58981fb26ea2240f4/examples/imagenet/train.py#L70.
  fake_batch_hps = copy.copy(hps)
  fake_batch_hps.batch_size = 2
  fake_inputs = model.get_fake_inputs(fake_batch_hps)
  if fake_inputs:
    fake_input_batch = fake_inputs
  elif isinstance(hps.input_shape, list):  # Typical case for seq2seq models
    fake_input_batch = [np.zeros((2, *x), model_dtype) for x in hps.input_shape]
  else:  # Typical case for classification models
    fake_input_batch = [np.zeros((2, *hps.input_shape), model_dtype)]
  params_rng, init_rng, dropout_rng = jax.random.split(rng, num=3)

  # By jitting the model init function, we initialize the model parameters
  # lazily without computing a full forward pass. For further documentation, see
  # https://flax.readthedocs.io/en/latest/flax.linen.html?highlight=jax.jit#flax.linen.Module.init.
  # We need to close over train=False here because otherwise the jitted init
  # function will convert the train Python bool to a jax boolean, which will
  # mess up Pythonic boolean statements like `not train` inside the model
  # construction.
  model_init_fn = jax.jit(
      functools.partial(model.flax_module.init, train=False))
  init_dict = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                            *fake_input_batch)
  # Trainable model parameters.
  params = init_dict['params']
  batch_stats = init_dict.get('batch_stats', {})

  if hps.get('layer_rescale_factors'):
    params = model_utils.rescale_layers(params, hps.layer_rescale_factors)
  # We don't pass batch_stats to the initializer, the initializer will just
  # run batch_norm in train mode and does not need to maintain the batch_stats.
  # TODO(gilmer): We hardcode here weighted_cross_entropy, but this will need
  # to change for other models. Maybe have meta_loss_inner as an initializer
  # hyper_param?
  # TODO(gilmer): instead of passing in weighted_xent, pass in the model and get
  # the loss from that.
  params = initializer(model.loss_fn, model.flax_module, params, hps,
                       hps.input_shape, hps.output_shape, init_rng,
                       metrics_logger)

  return params, batch_stats
