# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Data generators for init2winit."""

from init2winit.dataset_lib import data_utils
import jax.random
from ml_collections.config_dict import config_dict
import numpy as np


NQM_HPARAMS = config_dict.ConfigDict(
    dict(
        train_size=1e10,
        valid_size=0,
        test_size=0,
        input_shape=(100,),  # This determines the dimension.
        output_shape=(1,),
    ))
NQM_METADATA = {
    'apply_one_hot_in_loss': False,
}


def get_nqm_noise(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Returns the noise seed for the nqm model.

  NOTE: This dataset is only meant to be used with the nqm model.
  This just generates isotropic Gaussian noise of the desired dimension.
  The nqm model will then multiple this noise by a matrix D, with the properly
  that D^T D = C. This yields noise with gradient covariance C.

  Args:
    shuffle_rng: Not used.
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: Not used.
    hps: Hparams object. We only refer to hps.input_shape to determine the
      dimension of the noise.
  Returns:
    train_epoch, eval_train_epoch, valid_epoch, test_epoch: three generators.
      Only train_epoch is used.
  """
  del eval_batch_size

  per_host_batch_size = batch_size // jax.process_count()
  # Should train_rng / eval_rng possibly have different seeds?
  seed = data_utils.convert_jax_to_tf_random_seed(shuffle_rng)
  train_rng = np.random.RandomState(seed=seed)
  eval_rng = np.random.RandomState(seed=seed)

  def train_iterator_fn():
    while True:
      yield {
          'inputs': train_rng.normal(
              size=(per_host_batch_size, *hps.input_shape)
          )
      }

  def eval_train_epoch(num_batches):
    for _ in range(num_batches):
      yield {
          'inputs': eval_rng.normal(
              size=(per_host_batch_size, *hps.input_shape)
          )
      }

  # pylint: disable=unreachable
  def valid_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable

  return data_utils.Dataset(
      train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch
  )
