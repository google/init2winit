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

"""Module for processing wikitext-2 train, val and test datasets from raw text files to tokenized and batched tensorflow.data.Datasets."""

import os

from init2winit.dataset_lib import wikitext_tokenizer
from ml_collections.config_dict import config_dict
import tensorflow as tf

TRAIN_FILENAME = 'train.txt'
VALID_FILENAME = 'valid.txt'
TEST_FILENAME = 'test.txt'
SHUFFLE_BUFFER_SIZE = 1000_000
PAD_ID = -1


def get_trained_tokenizer(train_dataset: tf.data.Dataset,) -> tf.data.Dataset:
  tokenizer = wikitext_tokenizer.Tokenizer()
  tokenizer.train(train_dataset)
  return tokenizer


def split_input_target(sequence):
  input_sequence = sequence[:-1]
  target_sequence = sequence[1:]
  return {'inputs': input_sequence, 'targets': target_sequence}


def batch_with_padding(dataset: tf.data.Dataset,
                       batch_size,
                       padded_shapes=None,
                       padding_id=PAD_ID,
                       ):
  """Batches a tf.data.Dataset and adds padding if len(dataset) not divisible by the batch size.

  Args:
    dataset: tf.data.Dataset
    batch_size: batch size of resulting batched dataset
    padded_shapes: shapes of the padded batches
    padding_id: value for padding, for elements in new batch

  Returns:

  """
  batched_dataset = dataset.batch(batch_size, drop_remainder=False)

  # tf.data.Dataset.padded.batch pads elements in the batch so we call it
  # again with batch_size=1 to pad each element in original batch.
  padded_batched_dataset = batched_dataset.padded_batch(
      1, padded_shapes=padded_shapes, padding_values=padding_id)

  # Remove extra dimension resulting from the batch_size=1.
  padded_batched_dataset = padded_batched_dataset.unbatch()

  return padded_batched_dataset


def get_wikitext103_dataset(
    hps: config_dict.ConfigDict,
    train_batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
    shuffle_seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Returns wikitext-103 dataset.

  Args:
    hps: Dataset hyper parameters.
    train_batch_size: Batch size for train iterations
    valid_batch_size: Batch size for validation iterations
    test_batch_size: Batch size for test iterations
    shuffle_seed: seed for shuffling dataset sequences

  Returns:
    train_dataset, eval_train_dataset, valid_dataset, test_dataset
  """
  train_path = os.path.join(DATA_DIR, TRAIN_FILENAME)
  valid_path = os.path.join(DATA_DIR, VALID_FILENAME)
  test_path = os.path.join(DATA_DIR, TEST_FILENAME)

  # Get TextLineDataset from raw files
  train_text_dataset = tf.data.TextLineDataset(train_path)
  valid_text_dataset = tf.data.TextLineDataset(valid_path)
  test_text_dataset = tf.data.TextLineDataset(test_path)

  # Tokenize data
  tokenizer = get_trained_tokenizer(train_text_dataset)

  train_dataset_tokenized = tokenizer.tokenize(
      train_text_dataset)
  valid_dataset_tokenized = tokenizer.tokenize(
      valid_text_dataset)
  test_dataset_tokenized = tokenizer.tokenize(
      test_text_dataset)

  # Divide data in sequences of length sequence_length + 1, to contain inputs
  # and corresponding targets
  train_dataset_sequences = batch_with_padding(
      train_dataset_tokenized,
      hps.sequence_length + 1,
      padded_shapes=hps.sequence_length + 1,
  )
  valid_dataset_sequences = batch_with_padding(
      valid_dataset_tokenized,
      hps.sequence_length + 1,
      padded_shapes=hps.sequence_length + 1,
  )
  test_dataset_sequences = batch_with_padding(
      test_dataset_tokenized,
      hps.sequence_length + 1,
      padded_shapes=hps.sequence_length + 1,
  )

  # Split the sequences into inputs and targets.
  train_dataset_sequences = train_dataset_sequences.map(split_input_target)
  valid_dataset_sequences = valid_dataset_sequences.map(split_input_target)
  test_dataset_sequences = test_dataset_sequences.map(split_input_target)

  # Copy the train_dataset_sequences to a non repeating dataset
  eval_train_dataset_sequences = train_dataset_sequences

  # Shuffle the train sequences.
  train_dataset_sequences = train_dataset_sequences.shuffle(
      SHUFFLE_BUFFER_SIZE, seed=shuffle_seed)

  # Perform batching for training, validation and testing.
  # Make training data repeat indefinitely.
  train_dataset_sequences = train_dataset_sequences.repeat()
  train_dataset = train_dataset_sequences.batch(
      train_batch_size,
      drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  # Use padded batches for eval_train, validation and test_datasets since the
  # sequences do not repeat indefintely.
  eval_train_dataset = batch_with_padding(
      eval_train_dataset_sequences,
      train_batch_size,
      padded_shapes={
          'inputs': (train_batch_size, None),
          'targets': (train_batch_size, None)
      }).prefetch(tf.data.experimental.AUTOTUNE)

  valid_dataset = batch_with_padding(
      valid_dataset_sequences,
      valid_batch_size,
      padded_shapes={
          'inputs': (valid_batch_size, None),
          'targets': (valid_batch_size, None)
      }).prefetch(tf.data.experimental.AUTOTUNE)

  test_dataset = batch_with_padding(
      test_dataset_sequences,
      test_batch_size,
      padded_shapes={
          'inputs': (test_batch_size, None),
          'targets': (test_batch_size, None)
      }).prefetch(tf.data.experimental.AUTOTUNE)

  return train_dataset, eval_train_dataset, valid_dataset, test_dataset
