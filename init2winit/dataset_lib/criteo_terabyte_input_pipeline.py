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

from init2winit.dataset_lib.data_utils import Dataset
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds

# Change below to the path to dataset files.
CRITEO1TB_FILE_PATH = ''
CRITEO1TB_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(13 + 26,),
    train_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'train/train*'),
    eval_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'eval/eval*'),
    train_size=4_195_197_692,
    valid_size=89_137_318,
))
CRITEO1TB_METADATA = {
    'apply_one_hot_in_loss': True,
}


def _criteo_tsv_reader(
    file_path=None,
    num_dense_features=None,
    vocab_sizes=None,
    batch_size=None,
    is_training=True):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.

  Args:
    file_path: filepath to the criteo dataset.
    num_dense_features: number of dense features.
    vocab_sizes: vocabulary size.
    batch_size: batch size.
    is_training: whether or not this split is the one trained on.
  Returns:
    A tf.data.Dataset object.
  """

  @tf.function
  def _parse_example_fn(example):
    """Parser function for pre-processed Criteo TSV records."""
    label_defaults = [[0.0]]
    int_defaults = [[0.0] for _ in range(num_dense_features)]
    categorical_defaults = [[0] for _ in range(len(vocab_sizes))]
    record_defaults = label_defaults + int_defaults + categorical_defaults
    fields = tf.io.decode_csv(
        example, record_defaults, field_delim='\t', na_value='-1')

    num_labels = 1
    num_dense = len(int_defaults)
    features = {}
    features['targets'] = tf.reshape(fields[0], [batch_size])

    int_features = []
    for idx in range(num_dense):
      int_features.append(fields[idx + num_labels])
    int_features = tf.stack(int_features, axis=1)

    cat_features = []
    for idx in range(len(vocab_sizes)):
      cat_features.append(
          tf.cast(fields[idx + num_dense + num_labels], dtype=tf.int32))
    cat_features = tf.cast(
        tf.stack(cat_features, axis=1), dtype=int_features.dtype)
    features['inputs'] = tf.concat([int_features, cat_features], axis=1)
    features['weights'] = tf.ones(
        shape=(features['inputs'].shape[0],), dtype=features['inputs'].dtype)
    return features

  filenames = tf.data.Dataset.list_files(file_path, shuffle=False)
  index = jax.process_index()
  num_hosts = jax.process_count()
  ds = filenames.shard(num_hosts, index)

  if is_training:
    ds = ds.repeat()
  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=128,
      block_length=batch_size // 8,
      num_parallel_calls=128,
      deterministic=False)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.map(_parse_example_fn, num_parallel_calls=16)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def convert_to_numpy_iterator_fn(num_batches, tf_dataset_fn):
  return itertools.islice(tfds.as_numpy(tf_dataset_fn()), num_batches)


def get_criteo1tb(unused_shuffle_rng,
                  batch_size,
                  eval_batch_size,
                  hps):
  """Get the Criteo 1TB train and eval iterators."""
  process_count = jax.process_count()
  if batch_size % process_count != 0:
    raise ValueError('process_count={} must divide batch_size={}.'.format(
        process_count, batch_size))
  if eval_batch_size is None:
    eval_batch_size = batch_size
  if eval_batch_size % process_count != 0:
    raise ValueError('process_count={} must divide eval_batch_size={}.'.format(
        process_count, eval_batch_size))
  per_host_eval_batch_size = eval_batch_size // process_count
  per_host_batch_size = batch_size // process_count
  train_dataset = _criteo_tsv_reader(
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_batch_size,
      is_training=True)
  train_iterator_fn = lambda: tfds.as_numpy(train_dataset())
  eval_train_dataset = _criteo_tsv_reader(
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_eval_batch_size,
      is_training=False)
  eval_train_epoch = functools.partial(
      convert_to_numpy_iterator_fn, tf_dataset_fn=eval_train_dataset)
  eval_dataset = _criteo_tsv_reader(
      file_path=hps.eval_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_eval_batch_size,
      is_training=False)
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
