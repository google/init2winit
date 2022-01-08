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

"""Registry for the available initializers we can test.

API of an initializer:
new_params = init(loss, init_params, hps, num_outputs, input_shape)

TODO(gilmer, gdahl, schsam): The API of an initializer should in general be
aware of the moments of the data. Currently we assume that all input coordinates
are iid standard normal distributions.
"""

from init2winit.init_lib import meta_init
from init2winit.init_lib import sparse_init
from ml_collections.config_dict import config_dict


# This function is meant to match the general API of an initializer
# pylint: disable=unused-argument
def noop(
    loss_fn=None,
    flax_module=None,
    params=None,
    hps=None,
    input_shape=None,
    output_shape=None,
    rng_key=None,
    metrics_logger=None,
):
  """No-op init."""
  return params
# pylint: enable=unused-argument

DEFAULT_HPARAMS = config_dict.ConfigDict()

_ALL_INITIALIZERS = {
    'noop': (noop, DEFAULT_HPARAMS),
    'meta_init': (meta_init.meta_init, meta_init.DEFAULT_HPARAMS),
    'sparse_init': (sparse_init.sparse_init, sparse_init.DEFAULT_HPARAMS),
}


def get_initializer(initializer_name):
  """Get the corresponding initializer function based on the initializer string.

  API of an initializer:
  init_fn, hparams = get_initializer(init)
  new_params, final_l = init_fn(loss, init_params, hps,
                                num_outputs, input_shape)

  Args:
    initializer_name: (str) e.g. default.

  Returns:
    initializer
  Raises:
    ValueError if model is unrecognized.
  """
  try:
    return _ALL_INITIALIZERS[initializer_name][0]
  except KeyError:
    raise ValueError('Unrecognized initializer: {}'.format(initializer_name))


def get_initializer_hparams(initializer_name):
  """Get the corresponding hyperparameters based on the initializer string.

  Args:
    initializer_name: (str) e.g. default.

  Returns:
    hps
  Raises:
    ValueError if model is unrecognized.
  """
  try:
    return _ALL_INITIALIZERS[initializer_name][1]
  except KeyError:
    raise ValueError('Unrecognized initializer: {}'.format(initializer_name))
