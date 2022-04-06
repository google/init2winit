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

"""Imports a PG-19 dataset.

PG-19 reference:
@article{raecompressive2019,
author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and
          Hillier, Chloe and Lillicrap, Timothy P},
title = {Compressive Transformers for Long-Range Sequence Modelling},
journal = {arXiv preprint},
url = {https://arxiv.org/abs/1911.05507},
year = {2019},
}

This module imports a preprocessed PG-19 dataset from TFRecords. The PG-19 text
files were tokenized and encoded with SubwordTextEncoder and aggregated into
tensors of maximum lenght of 8192.
"""
import itertools
from init2winit.dataset_lib.data_utils import Dataset
from init2winit.dataset_lib.data_utils import tf_to_numpy
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE

VOCAB_SIZE = 98305

DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
        max_target_length=8192,
        shuffle_size=512,
        train_size=302013,
        # Used for initializing the model, the acutal length does not matter
        # See //third_party/py/init2winit/init_lib/init_utils.py
        input_shape=(16,),
        output_shape=(VOCAB_SIZE,)))

METADATA = {
    'train_ds_length': 302013,
    'validation_ds_length': 456,
    'test_ds_length': 1063,
    'vocab_length': 98305,
    'apply_one_hot_in_loss': True,
}


def get_pg19(shuffle_rng: jax.random.PRNGKey,
             batch_size: int,
             eval_batch_size: int = None,
             hps: config_dict.ConfigDict = None):
  """Generates PG-19 data.

  Args:
    shuffle_rng: a JAX PRNGKey used in shuffling the dataset.
    batch_size: batch size in the train batching process.
    eval_batch_size: batch size in the evaluation batching process.
    hps: a configuration dictionary with hyperparameters.

  Returns:
    output: a dataset with data generators.
  """
  process_count = jax.process_count()
  if batch_size % process_count != 0:
    raise ValueError('process_count={} must divide batch_size={}.'.format(
        process_count, batch_size))

  per_host_batch_size = batch_size // process_count
  if eval_batch_size is None:
    eval_batch_size = per_host_batch_size

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

  def decode_fn(tensor: tf.Tensor):
    """Decodes a serialized data in a tensor.

    Args:
      tensor: a tensor with encoded data.

    Returns:
      output: a tensor with decoded data.
    """
    return tf.io.parse_example(
        tensor, {
            'targets':
                tf.io.FixedLenSequenceFeature(
                    shape=[], dtype=tf.int64, allow_missing=True)
        })

  def add_inputs(tensor):
    """Adds 'inputs' to a feature dictonary as required by base_model.py.

    Args:
      tensor: a tensor with decoded data.

    Returns:
      output: a tensor with decoded data.
    """
    data = tensor['targets']
    return {'inputs': data, 'targets': data}

  # NOTE(krasowiak): build new preprocessing logic with an alternative tokenizer
  def load_and_decode_tf_records(split: str):
    """Loads a dataset from TFRecords contained in a SSTable.

    Args:
      split: a dataset split name: 'train', 'validation' or 'test'.

    Returns:
      output: a decoded TF dataset.
    """


    dataset = tf.data.TFRecordDataset(
        gfile.GenerateShardedFilenames(
            file_pattern=tftrecord_dataset_paths[split]))
    dataset = dataset.map(map_func=decode_fn, num_parallel_calls=AUTOTUNE)
    return dataset

  def prepare_ds(split: str,
                 per_host_batch_size: int,
                 hps: config_dict.ConfigDict,
                 shuffle: bool = False,
                 shuffle_rng: bool = None,
                 num_epochs: int = 1,
                 drop_remainder: bool = True):
    """Preprocesses a dataset.

    Args:
      split: a dataset split name: 'train', 'validation' or 'test'.
      per_host_batch_size: batch size in the batching process.
      hps: a configuariton dictionary with a set of hyperparameters.
      shuffle: whether to shuffle the TF dataset.
      shuffle_rng: a JAX PRNGKey used in shuffling the dataset.
      num_epochs: number of time each orginal value will be repeated.
      drop_remainder: whether to drop remainder in the batching process.

    Returns:
      output: a preprocessed TF dataset.
    """
    dataset = load_and_decode_tf_records(split=split)
    if shuffle:
      if shuffle_rng:
        dataset = dataset.shuffle(
            buffer_size=hps.shuffle_size, seed=shuffle_rng)
      else:
        dataset = dataset.shuffle(buffer_size=hps.shuffle_size)
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.padded_batch(
        batch_size=per_host_batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(map_func=add_inputs, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

  train_ds = prepare_ds(
      split='train',
      per_host_batch_size=per_host_batch_size,
      hps=hps,
      shuffle=True,
      shuffle_rng=shuffle_rng[0],
      num_epochs=None,
  )
  eval_ds = prepare_ds(
      split='validation',
      per_host_batch_size=per_host_eval_batch_size,
      hps=hps,
      drop_remainder=False)
  test_ds = prepare_ds(
      split='test',
      per_host_batch_size=per_host_eval_batch_size,
      hps=hps,
      drop_remainder=False)

  def train_iterator_fn():
    """Iterates over the train dataset and yields Numpy batches.

    Yields:
      output: train Numpy batches.
    """
    for batch in iter(train_ds):
      yield tf_to_numpy(tfds_data=batch)

  def eval_train_epoch(num_batches: int = None):
    """Iterates over the train dataset and yields Numpy batches.

    Args:
      num_batches: number of elements from the dataset to iterate

    Yields:
      output: in-training evaluation Numpy batches.
    """
    for batch in itertools.islice(iter(train_ds), num_batches):
      yield tf_to_numpy(tfds_data=batch)

  def valid_epoch(num_batches: int = None):
    """Iterates over the evaluation dataset and yields Numpy batches.

    Args:
      num_batches: number of elements from the dataset to iterate

    Yields:
      output: post-training evaluation Numpy batches.
    """
    for batch in itertools.islice(iter(eval_ds), num_batches):
      yield tf_to_numpy(tfds_data=batch)

  def test_epoch(num_batches=None):
    """Iterates over the test dataset and yields Numpy batches.

    Args:
      num_batches: number of elements from the dataset to iterate

    Yields:
      output: post-training test Numpy batches.
    """
    for batch in itertools.islice(iter(test_ds), num_batches):
      yield tf_to_numpy(tfds_data=batch)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
