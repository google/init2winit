# coding=utf-8
# Copyright 2025 The init2winit Authors.
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
from init2winit.dataset_lib import librispeech_input_pipeline
from init2winit.dataset_lib.data_utils import Dataset
import jax
from ml_collections.config_dict import config_dict
import numpy as np

MAX_INPUT_LENGTH = 320000
MAX_TARGET_LENGTH = 256
VOCAB_SIZE = 1024

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        train_split='train_clean100+train_clean360+train_other500',
        eval_split='dev_clean+dev_other',
        test_split='test_clean',
        input_shape=[(MAX_INPUT_LENGTH,), (MAX_INPUT_LENGTH,)],
        output_shape=(-1, VOCAB_SIZE),
        train_size=281241,
        tokenizer_vocab_path='',
        tokenizer_type='SPM'))

METADATA = {'apply_one_hot_in_loss': False}


def _batch_to_dict(batch):
  batch_np = data_utils.tf_to_numpy(batch)
  return batch_np


def get_librispeech(shuffle_rng, batch_size, eval_batch_size=None, hps=None):
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

  return _get_librispeech(hps, per_host_batch_size, per_host_eval_batch_size,
                          shuffle_rng)


def _get_librispeech(hps, per_host_batch_size, per_host_eval_batch_size,
                     shuffle_rng):
  """Data generators for lm1b."""
  n_devices = jax.local_device_count()
  if per_host_batch_size % n_devices != 0:
    raise ValueError('n_devices={} must divide per_host_batch_size={}.'.format(
        n_devices, per_host_batch_size))

  if per_host_eval_batch_size % n_devices != 0:
    raise ValueError(
        'n_devices={} must divide per_host_eval_batch_size={}.'.format(
            n_devices, per_host_eval_batch_size))

  train_ds, eval_ds, test_ds = librispeech_input_pipeline.get_librispeech_datasets(
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
      batch = _batch_to_dict(batch)
      yield data_utils.maybe_pad_batch(
          batch, desired_batch_size=per_host_eval_batch_size, padding_value=1.0)

  def test_epoch(num_batches=None):
    test_iter = iter(test_ds)
    for batch in itertools.islice(test_iter, num_batches):
      batch = _batch_to_dict(batch)
      yield data_utils.maybe_pad_batch(
          batch, desired_batch_size=per_host_eval_batch_size, padding_value=1.0)

  # pylint: enable=unreachable
  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)


def get_fake_batch(hps):
  return {
      'inputs':
          np.ones((hps.batch_size, hps.max_input_length),
                  dtype=hps.model_dtype),
      'input_paddings':
          np.ones((hps.batch_size, hps.max_input_length),
                  dtype=hps.model_dtype),
      'targets':
          np.ones((hps.batch_size, hps.max_target_length),
                  dtype=hps.model_dtype),
      'target_paddings':
          np.ones((hps.batch_size, hps.max_target_length),
                  dtype=hps.model_dtype),
  }
