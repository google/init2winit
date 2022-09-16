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

"""Data loader for pre-processed Criteo data.

Similar to how the NVIDIA example works, we split data from the last day into a
validation and test split (taking the first half for test and second half for
validation). See here for the NVIDIA example:
https://github.com/NVIDIA/DeepLearningExamples/blob/4e764dcd78732ebfe105fc05ea3dc359a54f6d5e/PyTorch/Recommendation/DLRM/preproc/run_spark_cpu.sh#L119.
"""
import functools
import math
import os

from absl import logging
from init2winit.dataset_lib import data_utils
import jax
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Change below to the path to dataset files.
CRITEO1TB_FILE_PATH = ''
CRITEO1TB_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(13 + 26,),
    train_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'train/train*'),
    eval_file_path=os.path.join(CRITEO1TB_FILE_PATH, 'eval/eval*'),
    train_size=4_195_197_692,
    # This will be exactly double if use_half_last_criteo_day_as_test=False.
    valid_size=89_137_318 // 2,
    test_size=89_137_318 // 2,
    # Set to False if the entire final day is used as a validation set (and no
    # test set is used).
    use_half_last_criteo_day_as_test=True,
))
CRITEO1TB_METADATA = {
    'apply_one_hot_in_loss': True,
}


@tf.function
def _parse_example_fn(num_dense_features, vocab_sizes, example):
  """Parser function for pre-processed Criteo TSV records."""
  label_defaults = [[0.0]]
  int_defaults = [[0.0] for _ in range(num_dense_features)]
  categorical_defaults = [[0] for _ in range(len(vocab_sizes))]
  record_defaults = label_defaults + int_defaults + categorical_defaults
  fields = tf.io.decode_csv(
      example, record_defaults, field_delim='\t', na_value='-1')

  num_labels = 1
  features = {}
  features['targets'] = tf.reshape(fields[0], (-1,))

  int_features = []
  for idx in range(num_dense_features):
    int_features.append(fields[idx + num_labels])
  int_features = tf.stack(int_features, axis=1)

  cat_features = []
  for idx in range(len(vocab_sizes)):
    cat_features.append(
        tf.cast(fields[idx + num_dense_features + num_labels], dtype=tf.int32))
  cat_features = tf.cast(
      tf.stack(cat_features, axis=1), dtype=int_features.dtype)
  features['inputs'] = tf.concat([int_features, cat_features], axis=1)
  return features


def _criteo_tsv_reader(
    split,
    file_path,
    num_dense_features,
    vocab_sizes,
    batch_size,
    use_half_last_criteo_day_as_test=True):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.

  Args:
    split: a text string indicating which split, one of
      {'train', 'eval_train', 'validation', 'test'}.
    file_path: filepath to the criteo dataset.
    num_dense_features: number of dense features.
    vocab_sizes: vocabulary size.
    batch_size: batch size.
    use_half_last_criteo_day_as_test: whether or not to split the first half of
      the last day of data into a test set, using the rest for validation.
  Returns:
    A tf.data.Dataset object.
  """
  if split not in ['train', 'eval_train', 'validation', 'test']:
    raise ValueError(f'Invalid split name {split}.')
  ds = tf.data.Dataset.list_files(file_path, shuffle=False)
  if split == 'test':
    if use_half_last_criteo_day_as_test:
      ds = ds.take(536)
    else:
      raise ValueError(
          'Need use_half_last_criteo_day_as_test=True if using a test split.')
  elif split == 'validation' and use_half_last_criteo_day_as_test:
    ds = ds.skip(536)
  index = jax.process_index()
  num_hosts = jax.process_count()

  ds = ds.shard(num_hosts, index)
  is_training = split == 'train'
  if is_training:
    ds = ds.repeat()
  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=128,
      block_length=batch_size // 8,
      num_parallel_calls=128,
      deterministic=False)
  ds = ds.batch(batch_size, drop_remainder=is_training)
  parse_fn = functools.partial(
      _parse_example_fn, num_dense_features, vocab_sizes)
  ds = ds.map(parse_fn, num_parallel_calls=16)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def _convert_to_numpy_iterator_fn(
    num_batches, per_host_eval_batch_size, tf_dataset, split_size):
  """Make eval iterator. This function is called at the start of each eval."""
  # Some hosts could see different numbers of examples, which in some cases
  # could lead to some hosts not having enough examples to make the same number
  # of batches. This makes pmap hang because it is waiting for a batch that will
  # never come from the host with less data.
  #
  # We assume that all files have the same number of examples in them, and while
  # this may not always be true, it dramatically simplifies/speeds up the logic
  # because the alternative is to run the same iterator (without data file
  # sharding) on each host, skipping (num_hosts - 1) / num_hosts batches.
  #
  # Any final partial batches are padded to be the full batch size, so we can
  # treat them all as being the same batch size.
  num_hosts = jax.process_count()
  num_batches_in_split = math.ceil(
      split_size / (per_host_eval_batch_size * num_hosts))
  if (num_batches is None or num_batches < 0 or
      num_batches > num_batches_in_split):
    logging.info('Setting num_batches to %d.', num_batches_in_split)
    num_batches = num_batches_in_split

  iterator = iter(tfds.as_numpy(tf_dataset))
  zeros_batch = None
  for _ in range(num_batches):
    try:
      batch = next(iterator)
    except StopIteration:
      if zeros_batch is None:
        zeros_batch = jax.tree_map(
            lambda x: np.zeros_like(x, dtype=x.dtype), batch)
      yield zeros_batch
      continue
    batch = data_utils.maybe_pad_batch(
        batch, desired_batch_size=per_host_eval_batch_size)
    yield batch


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
      split='train',
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_batch_size)
  train_iterator_fn = lambda: tfds.as_numpy(train_dataset)
  eval_train_dataset = _criteo_tsv_reader(
      split='eval_train',
      file_path=hps.train_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_eval_batch_size)
  eval_train_iterator_fn = functools.partial(
      _convert_to_numpy_iterator_fn,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=eval_train_dataset,
      split_size=hps.train_size)
  validation_dataset = _criteo_tsv_reader(
      split='validation',
      file_path=hps.eval_file_path,
      num_dense_features=hps.num_dense_features,
      vocab_sizes=hps.vocab_sizes,
      batch_size=per_host_eval_batch_size,
      use_half_last_criteo_day_as_test=hps.use_half_last_criteo_day_as_test)
  validation_iterator_fn = functools.partial(
      _convert_to_numpy_iterator_fn,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=validation_dataset,
      split_size=hps.valid_size)
  if hps.use_half_last_criteo_day_as_test:
    test_dataset = _criteo_tsv_reader(
        split='test',
        file_path=hps.eval_file_path,
        num_dense_features=hps.num_dense_features,
        vocab_sizes=hps.vocab_sizes,
        batch_size=per_host_eval_batch_size,
        use_half_last_criteo_day_as_test=hps.use_half_last_criteo_day_as_test)
    test_iterator_fn = functools.partial(
        _convert_to_numpy_iterator_fn,
        per_host_eval_batch_size=per_host_eval_batch_size,
        tf_dataset=test_dataset,
        split_size=hps.valid_size)
  else:
    # pylint: disable=unreachable
    def test_iterator_fn(*args, **kwargs):
      del args
      del kwargs
      return
      yield  # This yield is needed to make this a valid (null) iterator.
    # pylint: enable=unreachable
  return data_utils.Dataset(
      train_iterator_fn,
      eval_train_iterator_fn,
      validation_iterator_fn,
      test_iterator_fn)
