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

"""Implements a PG-19 dataset.

PG-19 reference:
@article{raecompressive2019,
author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and
          Hillier, Chloe and Lillicrap, Timothy P},
title = {Compressive Transformers for Long-Range Sequence Modelling},
journal = {arXiv preprint},
url = {https://arxiv.org/abs/1911.05507},
year = {2019},
}

This module implements a preprocessed PG-19 dataset from TFRecords. The PG-19
textfiles were tokenized and encoded with SubwordTextEncoder and aggregated into
tensors of maximum lenght of 8192.
"""
import functools
import itertools
import logging
import os
from typing import Dict, Optional, Tuple
from init2winit.dataset_lib import spm_tokenizer
from init2winit.dataset_lib.data_utils import Dataset
from init2winit.dataset_lib.data_utils import maybe_pad_batch
from init2winit.dataset_lib.data_utils import tf_to_numpy
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds

exists = tf.io.gfile.exists
makedirs = tf.io.gfile.makedirs

AUTOTUNE = tf.data.experimental.AUTOTUNE

VOCAB_SIZE = 35561

DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    max_target_length=8192,
    shuffle_size=512,
    train_size=302013,
    input_shape=(16,),
    output_shape=(VOCAB_SIZE,),
    vocab_path=None,
    data_dir=None,
    vocab_size=VOCAB_SIZE,
    max_corpus_chars=10**7,
    eod_id=1,
    eval_split='test'))

METADATA = {
    'apply_one_hot_in_loss': True,
}

TFRECORDS_SHARDS = {'train': 1000, 'validation': 1, 'test': 1}

Feature = Dict[str, tf.Tensor]


def split_strings(tensor: tf.Tensor) -> tf.data.Dataset:
  """Splits PG-19 book text by new lines and creates a TF Dataset."""
  return tf.data.Dataset.from_tensor_slices(tf.strings.split(tensor, sep=b'\n'))


def add_inputs_and_targets(tensor: tf.Tensor) -> Feature:
  """Adds 'inputs' to a feature dictonary as required by base_model.py."""
  return {'inputs': tensor, 'targets': tensor}


def generate_dataset(dataset_builder, split: str) -> tf.data.Dataset:
  """Loads TFDS PG-19 dataset and extracts relevant features.

  Args:
    dataset_builder: a TFDS dataset builder.
    split: a split of the dataset to load.

  Returns:
    output: a tensor with encoded data only.
  """
  split = tfds.split_for_jax_process(split, drop_remainder=False)
  dataset = dataset_builder.as_dataset(split=split, shuffle_files=False)
  dataset = dataset.map(lambda x: x['book_text'], num_parallel_calls=AUTOTUNE)
  dataset = dataset.flat_map(split_strings)
  dataset = dataset.map(tf.strings.strip, num_parallel_calls=AUTOTUNE)
  dataset = dataset.map(add_inputs_and_targets, num_parallel_calls=AUTOTUNE)
  return dataset


def map_line_length(tensor: tf.Tensor) -> Feature:
  """Maps length of a tokenized length for a tensor."""
  return (tf.cast(len(tensor), tf.int64), tensor)


def scan_func(state: tf.Tensor,
              element: tf.Tensor,
              max_target_length: int = 8192):
  """Maps (old_state, input_element) to (new_state, output_element).

  Args:
    state: an index of tensors that should be concatenated together.
    element: a tuple of tensor length and tokenized words.
    max_target_length: a maximum tensor length.

  Returns:
    output: new_state, output_element.
  """
  current_key, current_len = state
  line_length, _ = element
  if current_len + line_length > max_target_length - 1:
    current_key += 1
    current_len = line_length
  else:
    current_len += line_length
  return (current_key, current_len), (current_key, element)


def key_func(key: tf.Tensor, element: Feature) -> tf.Tensor:
  """Maps a nested structure of tensors to a scalar tf.int64 tensor."""
  del element  # Not used by group_by_reducer
  return key


def init_func(key: tf.Tensor) -> tf.TensorArray:
  """Maps a nested structure of tensors to a scalar tf.int64 tensor."""
  del key  # Not used by Reducer
  return tf.TensorArray(
      tf.int64, size=0, dynamic_size=True, clear_after_read=False)


def reduce_func(state: tf.Tensor, element: Feature) -> tf.TensorArray:
  """Reduce operation for tensors with the same key returning a new state."""
  return state.write(state.size(), element[1][1])


def finalize_func(state: tf.Tensor, eod_id: int = 1) -> tf.TensorArray:
  """Operation to be performed to return the final state."""
  state = state.write(state.size(), tf.constant([eod_id], dtype=tf.int64))
  return state.concat()


def preprocess_example(example: Feature) -> Dict[tf.Tensor, tf.Tensor]:
  """Performs initial preprocessing on tokenized examples.

  Args:
    example: a dictioary with tokenized 'inputs' and 'targets'.

  Returns:
    output: a dictionary with tensor length and tensor values.
  """
  example = example['targets']
  example = tf.cast(example, tf.int64)
  return map_line_length(example)


def generate_features(dataset: tf.data.Dataset,
                      hps: config_dict.ConfigDict) -> Feature:
  """Preprocesses a dataset before serliazing features and saving TFRecords.

  Args:
    dataset: TF Dataset to preprocess.
    hps: a configuariton dictionary with a set of hyperparameters.

  Returns:
    output: a preprocessed TF dataset.
  """
  dataset = dataset.map(preprocess_example, num_parallel_calls=AUTOTUNE)
  dataset = dataset.scan(
      (tf.convert_to_tensor(
          0, dtype=tf.int64), tf.convert_to_tensor(0, dtype=tf.int64)),
      functools.partial(scan_func, max_target_length=hps.max_target_length))
  reducer = tf.data.experimental.Reducer(
      init_func, reduce_func,
      functools.partial(finalize_func, eod_id=hps.eod_id))
  dataset = dataset.apply(
      tf.data.experimental.group_by_reducer(key_func, reducer))
  return dataset


def tensor_generator(dataset: tf.data.Dataset) -> tf.Tensor:
  """Yields preprocessed tensors to be saved as TFRecords.

  Args:
    dataset: TF Dataset to iterate on.

  Yields:
    output: a single preprocessed example to be written in TFRecords.
  """
  for tensor in dataset.as_numpy_iterator():
    yield tensor


def int64_feature(tensor: tf.Tensor) -> tf.train.Feature:
  """Creates a single TF Feature to be included in TF Example.

  Args:
    tensor: a tensor with a int64 sequence to be included in a TF Feature.

  Returns:
    output: a TF Feature.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=tensor))


def create_example(tensor: tf.Tensor) -> tf.train.Example:
  """Creates a single TF Example to be written in TF Records.

  Args:
    tensor: a tensor with a int64 sequence to be included in a TF Example.

  Returns:
    output: a TF Example.
  """
  feature = {'targets': int64_feature(tensor)}
  return tf.train.Example(features=tf.train.Features(feature=feature))


def write_pg19_tfrecords(data_dir: str, split: str, vocab_path: str,
                         hps: config_dict.ConfigDict, dataset_builder):
  """Writes preprocessed PG-19 data as TF Records split into shards.

  Args:
    data_dir: a path to a directory to save TFRecords.
    split: a dataset split name: 'train', 'validation' or 'test'.
    vocab_path: a vocab_path for the SentencePiece tokenizer.
    hps: a configuariton dictionary with a set of hyperparameters.
    dataset_builder: a TFDS Dataset Builder.

  Returns:
    output: a TF Example.
  """
  if not exists(data_dir):
    makedirs(data_dir)

  train_data = generate_dataset(dataset_builder, 'train')

  sp_tokenizer = spm_tokenizer.load_or_train_tokenizer(
      train_data,
      vocab_path=vocab_path,
      vocab_size=hps.vocab_size,
      max_corpus_chars=hps.max_corpus_chars)

  if split == 'dev':
    split = 'validation'

  split_dataset = generate_dataset(dataset_builder, split)
  split_dataset = split_dataset.map(
      spm_tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  split_dataset = generate_features(split_dataset, hps)

  num_shards = TFRECORDS_SHARDS[split]

  if split == 'validation':
    split = 'dev'

  for shard in range(num_shards):
    output_filename = '%s-%.5d-of-%.5d' % (split, shard, num_shards)
    output_file = os.path.join(data_dir, output_filename)
    with tf.io.TFRecordWriter(output_file) as writer:
      for feature in tensor_generator(split_dataset):
        feature = create_example(feature)
        writer.write(feature.SerializeToString())


def decode_and_preprocess_example(encoded_example: Feature,
                                  hps: config_dict.ConfigDict) -> Feature:
  """Decodes a serialized data in an encoded_example.

  Args:
    encoded_example: a feature dictionary with encoded data.
    hps: a configuariton dictionary with a set of hyperparameters.

  Returns:
    output: a feature dictionary with decoded data.
  """
  example = tf.io.parse_example(
      encoded_example, {
          'targets':
              tf.io.FixedLenSequenceFeature(
                  shape=[], dtype=tf.int64, allow_missing=True)
      })
  return example['targets'][:hps.max_target_length]


def output_preprocess(tensor: tf.Tensor) -> Tuple[Feature, Feature]:
  """Performs final preprocesing of each tensor.

  Args:
    tensor: a tensor to process.

  Returns:
    output: tuple of dictionaries with 'inputs' and 'targets' with identical
    tensors assinged.
  """
  tensor = tf.cast(tensor, tf.int32)
  return add_inputs_and_targets(tensor)


def get_dataset(data_dir: str,
                split: str,
                vocab_path: str,
                per_host_batch_size: int,
                hps: config_dict.ConfigDict,
                shuffle: bool,
                shuffle_rng: jax.random.PRNGKey,
                process_count: int,
                repeat: bool = False,
                drop_remainder: bool = False) -> tf.data.Dataset:
  """Loads and decodes PG-19 TFRecords.

  Args:
    data_dir: a path to a directory with 'train', 'validation' and 'test'
      TFRecords.
    split: a dataset split name: 'train', 'validation' or 'test'.
    vocab_path: a vocab_path for the SentencePiece tokenizer.
    per_host_batch_size: batch size in the batching process.
    hps: a configuariton dictionary with a set of hyperparameters.
    shuffle: buffer size in the TF Dataset shuffling process.
    shuffle_rng: a JAX PRNGKey used in shuffling the dataset.
    process_count: JAX process count for sharding TFRecords across hosts.
    repeat: whether an example should be repeated.
    drop_remainder: whether to drop remainder in the batching process.

  Returns:
    output: a TF Dataset with decoded and preprocessed PG-19 TFRecords.
  """
  if split == 'validation':
    split = 'dev'
  data_files = tf.io.matching_files(os.path.join(data_dir, (split + '*')))
  if not exists(data_dir) and tf.size(data_files) == 0:
    logging.info('There is no directory like %s or'
                 ' there is no TFRecords for %s split', data_dir, split)
    logging.info('Generating TFRecords for the %s split.'
                 ' If it is a train split then it might'
                 ' take 5-6 hours to complete.', split)

    pg19_builder = tfds.builder('pg19')
    write_pg19_tfrecords(
        data_dir=data_dir,
        split=split,
        vocab_path=vocab_path,
        hps=hps,
        dataset_builder=pg19_builder)
    data_files = tf.io.matching_files(os.path.join(data_dir, (split + '*')))

  if split == 'train':
    tf.random.shuffle(data_files, shuffle_rng)
  data_files = tf.data.Dataset.from_tensor_slices(data_files)
  data_files = data_files.shard(process_count, jax.process_index())
  dataset = tf.data.TFRecordDataset(data_files, buffer_size=8 * 1024 * 1024)
  dataset = dataset.map(
      lambda x: decode_and_preprocess_example(x, hps=hps),
      num_parallel_calls=AUTOTUNE)
  # In T2T they shuffle twice with a buffer sizes of 1024 and 512 respectively
  # We shuffle only once with a buffer size 512 since it does not seem to affect
  # results
  if shuffle:
    dataset = dataset.shuffle(hps.shuffle_size, shuffle_rng)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.padded_batch(
      batch_size=per_host_batch_size,
      padded_shapes=hps.max_target_length,
      drop_remainder=drop_remainder)
  dataset = dataset.map(output_preprocess, num_parallel_calls=AUTOTUNE)
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset


def get_pg19_datasets(
    hps: config_dict.ConfigDict,
    per_host_batch_size: int,
    per_host_eval_batch_size: int,
    shuffle_rng: jax.random.PRNGKey,
    process_count: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Preprocesses a dataset.

  Args:
    hps: a configuariton dictionary with a set of hyperparameters.
    per_host_batch_size: batch size in the batching process.
    per_host_eval_batch_size: batch size in the evaluation dataset batching
      process.
    shuffle_rng: a JAX PRNGKey used in shuffling the dataset.
    process_count: JAX process count for sharding TFRecords across hosts.

  Returns:
    output: preprocessed train, eval and test TF datasets.
  """
  if hps.vocab_path is None:
    vocab_path = os.path.expanduser('~/pg19_sentencepiece_model')
  else:
    vocab_path = hps.vocab_path

  logging.info('vocab_path = %s', vocab_path)

  if hps.data_dir is None:
    data_dir = os.path.expanduser('~/pg19_data')
  else:
    data_dir = hps.data_dir

  logging.info('data_dir = %s', data_dir)

  train_ds = get_dataset(
      data_dir=data_dir,
      split='train',
      vocab_path=vocab_path,
      per_host_batch_size=per_host_batch_size,
      hps=hps,
      repeat=True,
      shuffle=True,
      # TODO(b/280322542): this should be jax.random.bits(shuffle_rng)
      shuffle_rng=jax.random.key_data(shuffle_rng)[0],
      drop_remainder=True,
      process_count=process_count)
  eval_ds = get_dataset(
      data_dir=data_dir,
      split='validation',
      vocab_path=vocab_path,
      per_host_batch_size=per_host_eval_batch_size,
      hps=hps,
      shuffle=False,
      shuffle_rng=None,
      process_count=process_count)
  test_ds = get_dataset(
      data_dir=data_dir,
      split='test',
      vocab_path=vocab_path,
      per_host_batch_size=per_host_eval_batch_size,
      hps=hps,
      shuffle=False,
      shuffle_rng=None,
      process_count=process_count)

  return train_ds, eval_ds, test_ds


def get_pg19(shuffle_rng: jax.random.PRNGKey = None,
             batch_size: int = 8,
             eval_batch_size: Optional[int] = None,
             hps: config_dict.ConfigDict = None):
  """PG-19 data generator.

  Args:
    shuffle_rng: a JAX PRNGKey used in shuffling the dataset.
    batch_size: batch size in the train batching process.
    eval_batch_size: batch size in the evaluation batching process.
    hps: a configuration dictionary with hyperparameters.

  Returns:
    output: PG-19 data generators.
  """
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

  n_devices = jax.local_device_count()
  if per_host_batch_size % 1 != 0:
    raise ValueError('n_devices={} must divide per_host_batch_size={}.'.format(
        n_devices, per_host_batch_size))

  if per_host_eval_batch_size % 1 != 0:
    raise ValueError(
        'n_devices={} must divide per_host_eval_batch_size={}.'.format(
            n_devices, per_host_eval_batch_size))

  train_ds, eval_ds, test_ds = get_pg19_datasets(hps, per_host_batch_size,
                                                 per_host_eval_batch_size,
                                                 shuffle_rng,
                                                 process_count)

  def train_iterator_fn():
    """Iterates over the train dataset and yields Numpy batches."""
    for batch in iter(train_ds):
      yield tf_to_numpy(batch)

  def eval_train_epoch(num_batches: int = None):
    """Iterates over the train dataset and yields Numpy batches."""
    for batch in itertools.islice(iter(train_ds), num_batches):
      yield tf_to_numpy(batch)

  if hps.eval_split == 'test':

    def valid_epoch(num_batches: int = None):
      """Iterates over the evaluation dataset and yields Numpy batches."""
      for batch in itertools.islice(iter(test_ds), num_batches):
        batch = tf_to_numpy(batch)
        yield maybe_pad_batch(
            batch, desired_batch_size=per_host_eval_batch_size, padding_value=0)

    # pylint: disable=unreachable
    def test_epoch(*args, **kwargs):
      del args
      del kwargs
      return
      yield  # This yield is needed to make this a valid (null) iterator.

  else:

    def valid_epoch(num_batches: int = None):
      """Iterates over the evaluation dataset and yields Numpy batches."""
      for batch in itertools.islice(iter(eval_ds), num_batches):
        batch = tf_to_numpy(batch)
        yield maybe_pad_batch(
            batch, desired_batch_size=per_host_eval_batch_size, padding_value=0)

    def test_epoch(num_batches: int = None):
      """Iterates over the test dataset and yields Numpy batches."""
      for batch in itertools.islice(iter(test_ds), num_batches):
        batch = tf_to_numpy(batch)
        yield maybe_pad_batch(
            batch, desired_batch_size=per_host_eval_batch_size, padding_value=0)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
