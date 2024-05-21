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

"""LM1B input pipeline."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import lm1b_input_pipeline_v2
from init2winit.dataset_lib.data_utils import Dataset
import jax
from ml_collections.config_dict import config_dict
import numpy as np

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
  batch_np['weights'] = np.where(batch_np['inputs'] > 0, 1.0, 0.0)
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


def maybe_pad_batch(batch,
                    desired_batch_size,
                    mask_key='targets'):
  """Zero pad the batch on the right to desired_batch_size.

  All keys in the batch dictionary will have their corresponding arrays padded.
  Will return a dictionary with the same keys, additionally with the key
  'weights' added, with 1.0 indicating indices which are true data and 0.0
  indicating a padded index.

  Args:
    batch: A dictionary mapping keys to arrays. We assume that inputs is one of
      the keys.
    desired_batch_size: All arrays in the dict will be padded to have first
      dimension equal to desired_batch_size.
    mask_key: Typically used for text datasets, it's either 'inputs' (for
      encoder only models like language models) or 'targets'
      (for encoder-decoder models like seq2seq tasks) to decide weights for
      padded sequence. For Image datasets, this will be (most likely) unused.

  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally we
    add a key representing weights, to indicate how the batch was padded.
  """
  batch_size = batch['inputs'].shape[0]
  batch_pad = desired_batch_size - batch_size

  if mask_key not in ['targets', 'inputs']:
    raise ValueError(f'Incorrect mask key {mask_key}.')

  if 'weights' in batch:
    batch['weights'] = np.multiply(batch['weights'],
                                   np.where(batch[mask_key] > 0, 1, 0))
  else:
    batch['weights'] = np.where(batch[mask_key] > 0, 1, 0)

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if batch_pad == 0:
    new_batch = jax.tree.map(lambda x: x, batch)
    return new_batch

  def zero_pad(ar, pad_axis):
    pw = [(0, 0)] * ar.ndim
    pw[pad_axis] = (0, batch_pad)
    return np.pad(ar, pw, mode='constant')

  padded_batch = {'inputs': zero_pad(batch['inputs'], 0)}
  batch_keys = list(batch.keys())
  batch_keys.remove('inputs')
  for key in batch_keys:
    padded_batch[key] = zero_pad(batch[key], 0)
  return padded_batch


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

  train_ds, eval_train_ds, eval_ds = lm1b_input_pipeline_v2.get_lm1b_datasets(
      hps, per_host_batch_size, per_host_eval_batch_size, shuffle_rng)

  def train_iterator_fn():
    for batch in iter(train_ds):
      yield _batch_to_dict(batch)

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(eval_train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      batch = _batch_to_dict(batch)
      batch = maybe_pad_batch(batch, per_host_eval_batch_size)

      yield batch

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch in itertools.islice(valid_iter, num_batches):
      batch = _batch_to_dict(batch)
      batch = maybe_pad_batch(batch, per_host_eval_batch_size)

      yield batch

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable
  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
