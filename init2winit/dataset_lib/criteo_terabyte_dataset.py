# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Criteo 1TB dataset.

Similar to how the NVIDIA example works, we split data from the last day into a
validation and test split (taking the first half for test and second half for
validation). See here for the NVIDIA example:
https://github.com/NVIDIA/DeepLearningExamples/blob/4e764dcd78732ebfe105fc05ea3dc359a54f6d5e/PyTorch/Recommendation/DLRM/preproc/run_spark_cpu.sh#L119.
"""

import functools
import math
import os.path

from absl import logging
from init2winit.dataset_lib import data_utils
import jax
from jax.experimental import multihost_utils
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Change to the path to raw dataset files.
RAW_CRITEO1TB_FILE_PATH = ''
PREPROCESSED_CRITEO1TB_FILE_PATH = ''  # pylint: disable=invalid-name
CRITEO1TB_DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        input_shape=(13 + 26,),
        train_size=4_195_197_692,
        # We assume the tie breaking example went to the validation set, because
        # the test set in the mlperf version has 89_137_318 examples.
        valid_size=89_137_319,
        test_size=89_137_318,
    )
)
CRITEO1TB_METADATA = {
    'apply_one_hot_in_loss': True,
}

# Raw vocab sizes from
# https://cloud.google.com/tpu/docs/tutorials/dlrm-dcn-2.x#run-model.
_VOCAB_SIZES = [
    39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546,
    403346, 10, 2208, 11938, 155, 4, 976, 14, 39979771, 25641295, 39664984,
    585935, 12972, 108, 36
]


@tf.function
def _parse_example_fn(num_dense_features, example):
  """Parser function for pre-processed Criteo TSV records."""
  label_defaults = [[0.0]]
  int_defaults = [[0.0] for _ in range(num_dense_features)]
  categorical_defaults = [['00000000'] for _ in range(len(_VOCAB_SIZES))]
  record_defaults = label_defaults + int_defaults + categorical_defaults
  fields = tf.io.decode_csv(
      example, record_defaults, field_delim='\t', na_value='-1')

  num_labels = 1
  features = {}
  features['targets'] = tf.reshape(fields[0], (-1,))

  int_features = []
  for idx in range(num_dense_features):
    positive_val = tf.nn.relu(fields[idx + num_labels])
    int_features.append(tf.math.log(positive_val + 1))
  int_features = tf.stack(int_features, axis=1)

  cat_features = []
  for idx in range(len(_VOCAB_SIZES)):
    field = fields[idx + num_dense_features + num_labels]
    # We append the column index to the string to make the same id in different
    # columns unique.
    cat_features.append(
        tf.strings.to_hash_bucket_fast(field + str(idx), _VOCAB_SIZES[idx]))
  cat_features = tf.cast(
      tf.stack(cat_features, axis=1), dtype=int_features.dtype)
  features['inputs'] = tf.concat([int_features, cat_features], axis=1)
  return features


def criteo_tsv_reader(
    split,
    shuffle_rng,
    file_path,
    num_dense_features,
    batch_size,
    num_batches_to_prefetch):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.

  Args:
    split: a text string indicating which split, one of
      {'train', 'eval_train', 'validation', 'test'}.
    shuffle_rng: jax.random.PRNGKey used for shuffling, only used in training.
    file_path: filepath to the criteo dataset.
    num_dense_features: number of dense features.
    batch_size: per-host batch size.
    num_batches_to_prefetch: number of batches to prefetch.
  Returns:
    A tf.data.Dataset object.
  """
  if split not in ['train', 'eval_train', 'validation', 'test']:
    raise ValueError(f'Invalid split name {split}.')
  file_shuffle_seed = None
  data_shuffle_seed = None

  is_training = split == 'train'
  if is_training:
    file_shuffle_seed, data_shuffle_seed = jax.random.split(shuffle_rng, 2)
    file_shuffle_seed = data_utils.convert_jax_to_tf_random_seed(file_shuffle_seed)
    data_shuffle_seed = data_utils.convert_jax_to_tf_random_seed(data_shuffle_seed)

    file_shuffle_seed = multihost_utils.broadcast_one_to_all(
        file_shuffle_seed,
        is_source=jax.process_index() == 0
    )

  ds = tf.data.Dataset.list_files(
      file_path, shuffle=is_training, seed=file_shuffle_seed)
  index = jax.process_index()
  num_hosts = jax.process_count()
  ds = ds.shard(num_hosts, index)
  if is_training:
    ds = ds.repeat()
  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=64,
      block_length=batch_size // 8,
      num_parallel_calls=64,
      deterministic=False)
  if is_training:
    ds = ds.shuffle(buffer_size=524_288 * 100, seed=data_shuffle_seed)
  ds = ds.batch(batch_size, drop_remainder=is_training)
  parse_fn = functools.partial(_parse_example_fn, num_dense_features)
  ds = ds.map(parse_fn, num_parallel_calls=256)
  ds = ds.prefetch(num_batches_to_prefetch)
  return ds


_ARRAYRECORD_FEATURE_SPEC = {
    'inputs': tf.io.FixedLenFeature([13 + 26], tf.float32),
    'targets': tf.io.FixedLenFeature([1], tf.float32),
}


@tf.function
def _parse_arrayrecord_example_fn(serialized_examples):
  """Parse a batch of serialized tf.train.Examples from ArrayRecord."""
  parsed = tf.io.parse_example(serialized_examples, _ARRAYRECORD_FEATURE_SPEC)
  return {
      'inputs': parsed['inputs'],
      'targets': tf.squeeze(parsed['targets'], axis=-1),
  }


def criteo_arrayrecord_reader(
    split, shuffle_rng, file_path, batch_size, num_batches_to_prefetch
):
  """Input reader for preprocessed Criteo ArrayRecord data.

  Args:
    split: one of {'train', 'eval_train', 'validation', 'test'}.
    shuffle_rng: jax.random.PRNGKey for shuffling (train).
    file_path: glob pattern for .array_record files.
    batch_size: per-host batch size.
    num_batches_to_prefetch: number of batches to prefetch.

  Returns:
    A tf.data.Dataset object.
  """
  # Import here to avoid hard dependency for TSV-only users.
  if split not in ['train', 'eval_train', 'validation', 'test']:
    raise ValueError(f'Invalid split name {split}.')
  data_shuffle_seed = None

  is_training = split == 'train'
  if is_training:
    _, data_shuffle_seed = jax.random.split(shuffle_rng, 2)
    data_shuffle_seed = data_utils.convert_jax_to_tf_random_seed(
        data_shuffle_seed
    )

  # Discover all matching files.
  all_files = sorted(tf.io.gfile.glob(file_path))
  if not all_files:
    raise ValueError(f'No ArrayRecord files found matching: {file_path}')

  # Shard files across hosts.
  index = jax.process_index()
  num_hosts = jax.process_count()
  host_files = all_files[index::num_hosts]

  # Interleave per-file datasets, with batch+parse inside each file's
  # sub-pipeline. This is critical for performance: interleaving dense float
  # tensors (post-parse) is much faster than interleaving raw byte strings
  # and batching them later.
  file_ds = tf.data.Dataset.from_tensor_slices(host_files)
  if is_training:
    file_ds = file_ds.repeat()
    file_ds = file_ds.shuffle(
        buffer_size=2 * len(host_files), seed=data_shuffle_seed
    )

  ds = file_ds.interleave(
      lambda f: (
          ar_dataset.ArrayRecordDataset([f])
          .batch(
              batch_size,
              drop_remainder=is_training,
              num_parallel_calls=tf.data.AUTOTUNE,
              deterministic=False,
          )
          .map(
              _parse_arrayrecord_example_fn,
              num_parallel_calls=tf.data.AUTOTUNE,
              deterministic=False,
          )
      ),
      cycle_length=64,
      block_length=batch_size // 8,
      num_parallel_calls=64,
      deterministic=False,
  )
  ds = ds.prefetch(num_batches_to_prefetch)
  return ds


def _eval_numpy_iterator(
    num_batches, per_host_eval_batch_size, tf_dataset, split_size
):
  """Yields padded numpy batches from a tf.data eval split.

  Caps the number of batches to the split size divided across hosts. If the
  source runs out before `num_batches`, yields zero-filled batches so every
  host sees the same count.
  
  Args:
    num_batches (int): number of batches to process.
    per_host_eval_batch_size (int): batch size per host.
    tf_dataset (tfds.Dataset): source tensorflow dataset.
    split_size (int): total number of examples in the eval split.
    
  Yields:
    Padded numpy batches.
  """
  # We assume all files have the same number of examples. This simplifies the
  # logic because the alternative is to run the same iterator (without data file
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
        zeros_batch = jax.tree.map(
            lambda x: np.zeros_like(x, dtype=x.dtype), batch)
      yield zeros_batch
      continue
    batch = data_utils.maybe_pad_batch(
        batch, desired_batch_size=per_host_eval_batch_size)
    yield batch


def get_criteo1tb(shuffle_rng,
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

  use_raw_tsv = hps.get('use_raw_tsv', False)
  num_batches_to_prefetch = (
      hps.num_tf_data_prefetches
      if hps.num_tf_data_prefetches > 0
      else tf.data.AUTOTUNE
  )
  num_device_prefetches = hps.get('num_device_prefetches', 0)

  if use_raw_tsv:
    return _get_criteo1tb_tsv(
        shuffle_rng,
        per_host_batch_size,
        per_host_eval_batch_size,
        hps,
        num_batches_to_prefetch,
        num_device_prefetches,
    )
  else:
    return _get_criteo1tb_arrayrecord(
        shuffle_rng,
        per_host_batch_size,
        per_host_eval_batch_size,
        hps,
        num_batches_to_prefetch,
        num_device_prefetches,
    )


def _get_criteo1tb_arrayrecord(
    shuffle_rng,
    per_host_batch_size,
    per_host_eval_batch_size,
    hps,
    num_batches_to_prefetch,
    num_device_prefetches,
):
  """Load Criteo 1TB from preprocessed ArrayRecord files."""
  base = hps.get('preprocessed_data_path', PREPROCESSED_CRITEO1TB_FILE_PATH)
  train_file_path = os.path.join(base, 'train', '*')
  validation_file_path = os.path.join(
      base, 'val_set_second_half_of_day23_not_used', '*'
  )
  test_file_path = os.path.join(base, 'eval', '*')

  train_dataset = criteo_arrayrecord_reader(
      split='train',
      shuffle_rng=shuffle_rng,
      file_path=train_file_path,
      batch_size=per_host_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch,
  )
  data_utils.log_rss('train arrayrecord dataset created')

  if num_device_prefetches > 0:
    train_iterator_fn = lambda: data_utils.prefetch_iterator(
        tfds.as_numpy(train_dataset), num_device_prefetches
    )
    data_utils.log_rss(
        f'using prefetching with {num_device_prefetches} in the train dataset'
    )
  else:
    train_iterator_fn = lambda: tfds.as_numpy(train_dataset)

  eval_train_dataset = criteo_arrayrecord_reader(
      split='eval_train',
      shuffle_rng=None,
      file_path=train_file_path,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch,
  )
  eval_train_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=eval_train_dataset,
      split_size=hps.train_size,
  )
  data_utils.log_rss('eval_train arrayrecord dataset created')

  validation_dataset = criteo_arrayrecord_reader(
      split='validation',
      shuffle_rng=None,
      file_path=validation_file_path,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch,
  )
  validation_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=validation_dataset,
      split_size=hps.valid_size,
  )
  data_utils.log_rss('validation arrayrecord dataset created')

  test_dataset = criteo_arrayrecord_reader(
      split='test',
      shuffle_rng=None,
      file_path=test_file_path,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch,
  )
  test_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=test_dataset,
      split_size=hps.test_size,
  )
  data_utils.log_rss('test arrayrecord dataset created')

  eval_train_iterator_fn = data_utils.CachedIteratorFactory(
      eval_train_iterator_fn(None), 'eval_train'
  )
  validation_iterator_fn = data_utils.CachedIteratorFactory(
      validation_iterator_fn(None), 'validation'
  )
  test_iterator_fn = data_utils.CachedIteratorFactory(
      test_iterator_fn(None), 'test'
  )

  return data_utils.Dataset(
      train_iterator_fn,
      eval_train_iterator_fn,
      validation_iterator_fn,
      test_iterator_fn,
  )


def _get_criteo1tb_tsv(
    shuffle_rng,
    per_host_batch_size,
    per_host_eval_batch_size,
    hps,
    num_batches_to_prefetch,
    num_device_prefetches,
):
  """Load Criteo 1TB from raw TSV files (legacy path)."""
  train_file_path = os.path.join(RAW_CRITEO1TB_FILE_PATH, 'train/*/*')
  validation_file_path = os.path.join(
      RAW_CRITEO1TB_FILE_PATH, 'val_set_second_half_of_day23_not_used/*')
  test_file_path = os.path.join(RAW_CRITEO1TB_FILE_PATH, 'eval/day_23/*')

  train_dataset = criteo_tsv_reader(
      split='train',
      shuffle_rng=shuffle_rng,
      file_path=train_file_path,
      num_dense_features=hps.num_dense_features,
      batch_size=per_host_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch)
  data_utils.log_rss('train dataset created')
  if num_device_prefetches > 0:
    train_iterator_fn = lambda: data_utils.prefetch_iterator(
        tfds.as_numpy(train_dataset), num_device_prefetches
    )
    data_utils.log_rss(
        f'using prefetching with {num_device_prefetches} in the train dataset'
    )
  else:
    train_iterator_fn = lambda: tfds.as_numpy(train_dataset)
  eval_train_dataset = criteo_tsv_reader(
      split='eval_train',
      shuffle_rng=None,
      file_path=train_file_path,
      num_dense_features=hps.num_dense_features,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch)
  eval_train_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=eval_train_dataset,
      split_size=hps.train_size,
  )
  data_utils.log_rss('eval_train dataset created')
  validation_dataset = criteo_tsv_reader(
      split='validation',
      shuffle_rng=None,
      file_path=validation_file_path,
      num_dense_features=hps.num_dense_features,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch)
  validation_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=validation_dataset,
      split_size=hps.valid_size,
  )
  data_utils.log_rss('validation dataset created')
  test_dataset = criteo_tsv_reader(
      split='test',
      shuffle_rng=None,
      file_path=test_file_path,
      num_dense_features=hps.num_dense_features,
      batch_size=per_host_eval_batch_size,
      num_batches_to_prefetch=num_batches_to_prefetch)
  test_iterator_fn = functools.partial(
      _eval_numpy_iterator,
      per_host_eval_batch_size=per_host_eval_batch_size,
      tf_dataset=test_dataset,
      split_size=hps.test_size,
  )
  data_utils.log_rss('test dataset created')

  # Cache all the eval_train/validation/test iterators to avoid re-processing the same data files.
  eval_train_iterator_fn = data_utils.CachedIteratorFactory(
      eval_train_iterator_fn(None), 'eval_train'
  )
  validation_iterator_fn = data_utils.CachedIteratorFactory(
      validation_iterator_fn(None), 'validation'
  )
  test_iterator_fn = data_utils.CachedIteratorFactory(
      test_iterator_fn(None), 'test'
  )

  return data_utils.Dataset(
      train_iterator_fn,
      eval_train_iterator_fn,
      validation_iterator_fn,
      test_iterator_fn)


def get_fake_batch(hps):
  return {
      'inputs':
          np.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype),
      'targets':
          np.zeros((hps.batch_size,), dtype=hps.model_dtype),
  }
