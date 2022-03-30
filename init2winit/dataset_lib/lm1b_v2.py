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

"""LM1B input pipeline."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import lm1b_input_pipeline_v2
from init2winit.dataset_lib.data_utils import Dataset
import jax
import jax.numpy as jnp

from ml_collections.config_dict import config_dict

VOCAB_SIZE = 30000

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        max_target_length=128,
        max_eval_target_length=512,
        input_shape=(16,),
        output_shape=(VOCAB_SIZE,),
        vocab_size=VOCAB_SIZE,
        eval_split='test',
        vocab_path=None,
        train_size=30301028,
        pack_examples=False,
        max_corpus_chars=10**7))

METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_inputs': True,
    'causal': True,
}


def _batch_to_dict(batch):
  batch_np = data_utils.tf_to_numpy(batch)
  batch_np['weights'] = jnp.where(batch_np['inputs'] > 0, 1.0, 0.0)
  return batch_np


def get_lm1b(shuffle_rng, batch_size, eval_batch_size=None, hps=None):
  """Wrapper to conform to the general dataset API."""
  process_count = jax.process_count()
  if batch_size % process_count != 0:
    raise ValueError('process_count={} must divide batch_size={}.'.format(
        process_count, batch_size))

  per_host_batch_size = batch_size // process_count
  if eval_batch_size is None:
    eval_batch_size = batch_size

  if eval_batch_size % process_count != 0:
    raise ValueError('process_count={} must divide eval_batch_size={}.'.format(
        process_count, eval_batch_size))
  per_host_eval_batch_size = eval_batch_size // process_count

  return _get_lm1b(hps, per_host_batch_size, per_host_eval_batch_size,
                   shuffle_rng)


def _get_lm1b(hps, per_host_batch_size, per_host_eval_batch_size, shuffle_rng):
  """Data generators for lm1b."""
  n_devices = jax.local_device_count()
  if per_host_batch_size % n_devices != 0:
    raise ValueError('n_devices={} must divide per_host_batch_size={}.'.format(
        n_devices, per_host_batch_size))

  if per_host_eval_batch_size % n_devices != 0:
    raise ValueError(
        'n_devices={} must divide per_host_eval_batch_size={}.'.format(
            n_devices, per_host_eval_batch_size))

  train_ds, eval_ds = lm1b_input_pipeline_v2.get_lm1b_datasets(
      hps, per_host_batch_size, per_host_eval_batch_size, shuffle_rng)

  def train_iterator_fn():
    for batch in iter(train_ds):
      yield _batch_to_dict(batch)

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      yield _batch_to_dict(batch)

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch in itertools.islice(valid_iter, num_batches):
      yield _batch_to_dict(batch)

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable
  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
