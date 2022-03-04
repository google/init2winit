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

"""Data loader for pre-processed Criteo data."""

import functools
import itertools
import os

from absl import logging
from init2winit.dataset_lib.data_utils import Dataset
from ml_collections.config_dict import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds

# Change below to the path to dataset files.
CRITEO1TB_FILE_PATH = ''
CRITEO1TB_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(13 + 26,),
    train_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'train/train*'),
    eval_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'eval/eval*'),
    num_train_datasets=8192,
    # TODO(eamid): find the exact train_size
    train_size=4e9,
    num_eval_datasets=1072,
    parallelism=16,
    dataset_index=0,
    dataset_num_shards=4
))
CRITEO1TB_METADATA = {
    'apply_one_hot_in_loss': False,
}


class CriteoTsvReader(object):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.
  """

  def __init__(self,
               file_path=None,
               num_dense_features=None,
               vocab_sizes=None,
               batch_size=None,
               dataset_index=0,
               dataset_num_shards=None,
               is_training=True,
               parallelism=1,
               num_datasets=None):
    self._file_path = file_path
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._batch_size = batch_size
    self._dataset_index = dataset_index
    self._dataset_num_shards = dataset_num_shards
    self._is_training = is_training
    self._parallelism = parallelism

    num_files = len(tf.io.gfile.glob(file_path))
    if num_datasets > num_files:
      raise ValueError('There must be at least one file per dataset.')
    if num_datasets * parallelism > num_files:
      logging.warning(
          'Reducing dataset parallelism due to insufficient number of files.')
      self._parallelism = num_files // num_datasets

  def __call__(self):
    batch_size = self._batch_size

    @tf.function
    def _parse_example_fn(example):
      """Parser function for pre-processed Criteo TSV records."""
      label_defaults = [[0.0]]
      int_defaults = [[0.0] for _ in range(self._num_dense_features)]
      categorical_defaults = [[0] for _ in range(len(self._vocab_sizes))]
      record_defaults = label_defaults + int_defaults + categorical_defaults
      fields = tf.io.decode_csv(
          example, record_defaults, field_delim='\t', na_value='-1')

      num_labels = 1
      num_dense = len(int_defaults)
      features = {}
      features['targets'] = tf.reshape(fields[0], [batch_size, 1])

      int_features = []
      for idx in range(num_dense):
        int_features.append(fields[idx + num_labels])
      int_features = tf.stack(int_features, axis=1)

      cat_features = []
      for idx in range(len(self._vocab_sizes)):
        cat_features.append(
            tf.cast(fields[idx + num_dense + num_labels], dtype=tf.int32))
      cat_features = tf.cast(
          tf.stack(cat_features, axis=1), dtype=int_features.dtype)
      features['inputs'] = tf.concat([int_features, cat_features], axis=1)
      features['weights'] = tf.ones(
          shape=(features['inputs'].shape[0],), dtype=features['inputs'].dtype)
      return features

    filenames = tf.data.Dataset.list_files(self._file_path, shuffle=False)
    filenames = filenames.shard(self._dataset_num_shards, self._dataset_index)

    def make_dataset(ds_index, repeat=True):
      ds = filenames.shard(self._parallelism, ds_index)
      if repeat:
        ds = ds.repeat()
      ds = ds.interleave(
          tf.data.TextLineDataset,
          cycle_length=16,
          block_length=batch_size // 8,
          num_parallel_calls=8,
          deterministic=False)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.map(_parse_example_fn, num_parallel_calls=16)
      return ds

    ds_indices = tf.data.Dataset.range(self._parallelism)
    ds = ds_indices.interleave(
        functools.partial(make_dataset, repeat=self._is_training),
        cycle_length=self._parallelism,
        block_length=1,
        num_parallel_calls=self._parallelism,
        deterministic=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def convert_to_numpy_iterator_fn(num_batches, tf_dataset_fn):
  return itertools.islice(tfds.as_numpy(tf_dataset_fn()), num_batches)


def get_criteo1tb(unused_shuffle_rng,
                  batch_size,
                  eval_batch_size,
                  hps):
  """Get the Criteo 1TB train and eval iterators."""
  train_dataset = CriteoTsvReader(
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=batch_size,
      dataset_index=hps.dataset_index,
      dataset_num_shards=hps.dataset_num_shards,
      is_training=True,
      parallelism=hps.parallelism,
      num_datasets=hps.num_train_datasets)
  train_iterator_fn = lambda: tfds.as_numpy(train_dataset())
  eval_train_dataset = CriteoTsvReader(
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=eval_batch_size,
      dataset_index=hps.dataset_index,
      dataset_num_shards=hps.dataset_num_shards,
      is_training=False,
      parallelism=hps.parallelism,
      num_datasets=hps.num_train_datasets)
  eval_train_epoch = functools.partial(
      convert_to_numpy_iterator_fn, tf_dataset_fn=eval_train_dataset)
  eval_dataset = CriteoTsvReader(
      file_path=hps.eval_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=eval_batch_size,
      dataset_index=hps.dataset_index,
      dataset_num_shards=hps.dataset_num_shards,
      is_training=False,
      parallelism=hps.parallelism,
      num_datasets=hps.num_eval_datasets)
  eval_iterator_fn = functools.partial(
      convert_to_numpy_iterator_fn, tf_dataset_fn=eval_dataset)
  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable
  return Dataset(train_iterator_fn, eval_train_epoch, eval_iterator_fn,
                 test_epoch)

