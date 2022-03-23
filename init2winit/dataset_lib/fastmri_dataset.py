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

"""FastMRI knee singlecoil input pipeline."""

import itertools

from absl import logging
from init2winit.dataset_lib import data_utils
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds


# TODO(dsuo): update these.
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(320, 320),
    output_shape=(320, 320),
    train_size=34742,
    valid_size=7135,
    ))


METADATA = {
    'apply_one_hot_in_loss': False,
}


def load_split(per_host_batch_size, split, hps, shuffle_rng=None):
  """Creates a split from the FastMRI dataset using tfds.

  NOTE: only creates knee singlecoil datasets.

  Args:
    per_host_batch_size: the batch size returned by the data pipeline.
    split: One of ['train', 'eval_train', 'val'].
    hps: The hparams the experiment is run with. Required fields are train_size
      and valid_size.
    shuffle_rng: The RNG used to shuffle the split. Only used if
      `split == 'train'`.
  Returns:
    A `tf.data.Dataset`.
  """
  if split not in ['train', 'eval_train', 'val']:
    raise ValueError('Unrecognized split {}'.format(split))
  if split in ['train']:
    split_size = hps.train_size // jax.process_count()
  else:
    split_size = hps.valid_size // jax.process_count()
  start = jax.process_index() * split_size
  end = start + split_size
  # In order to properly load the full dataset, it is important that we load
  # entirely to the end of it on the last host, because otherwise we will drop
  # the last `{train,valid}_size % split_size` elements.
  if jax.process_index() == jax.process_count() - 1:
    end = -1

  logging.info('Loaded data [%d: %d] from %s', start, end, split)
  if split in ['train', 'eval_train']:
    tfds_split = 'train[{}:{}]'.format(start, end)
  else:  # split == 'val':
    tfds_split = 'val[{}:{}]'.format(start, end)

  ds = tfds.load(
      'fast_mri',
      split=tfds_split,
      shuffle_files=True,
  )

  ds = ds.cache()

  if split == 'train':
    ds = ds.shuffle(
        16 * per_host_batch_size,
        seed=shuffle_rng[0],
        reshuffle_each_iteration=True)
    ds = ds.repeat()

  ds = ds.batch(per_host_batch_size, drop_remainder=False)

  if split != 'train':
    ds = ds.cache()
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds


def get_fastmri(shuffle_rng, batch_size, eval_batch_size, hps):
  """FastMRI dataset.

  Args:
    shuffle_rng: rng for shuffling.
    batch_size: batch size.
    eval_batch_size: batch size for eval.
    hps: hyperparameters.

  Returns:
    An init2winit Dataset.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  train_ds = load_split(per_host_batch_size, 'train', hps, shuffle_rng)
  train_ds = tfds.as_numpy(train_ds)
  eval_train_ds = load_split(per_host_eval_batch_size, 'eval_train', hps)
  eval_train_ds = tfds.as_numpy(eval_train_ds)
  eval_ds = load_split(per_host_eval_batch_size, 'val', hps)
  eval_ds = tfds.as_numpy(eval_ds)

  def train_iterator_fn():
    for batch in iter(train_ds):
      yield {
          'inputs': batch['image'],
          'targets': batch['target']
      }

  def eval_train_epoch(num_batches=None):
    for batch in itertools.islice(iter(eval_train_ds), num_batches):
      batch_dict = {
          'inputs': batch['image'],
          'targets': batch['target']
      }
      yield data_utils.maybe_pad_batch(batch_dict, per_host_eval_batch_size)

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(iter(eval_ds), num_batches):
      batch_dict = {
          'inputs': batch['image'],
          'targets': batch['target']
      }
      yield data_utils.maybe_pad_batch(batch_dict, per_host_eval_batch_size)

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  return data_utils.Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                            test_epoch)
