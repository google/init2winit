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

"""Module for processing wikitext-2 train, val and test datasets from raw text files to tokenized and batched tensorflow.data.Datasets."""

import os
from typing import Any, Union

from init2winit.dataset_lib import spm_tokenizer
import numpy as np
import tensorflow as tf

DATA_DIR = 'wikitext-103'
SPM_TOKENIZER_VOCAB_PATH = os.path.join(
    DATA_DIR, 'wikitext_103_spm_vocab_8192.model'
)
TRAIN_FILENAME = 'train.txt'
VALID_FILENAME = 'valid.txt'
TEST_FILENAME = 'test.txt'

MAX_CORPUS_CHARS = 1_000_000_000  # for training tokenizer
SPM_TOKENIZER_VOCAB_SIZE = 8_192  # 2**13
SEQUENCE_LENGTH = 256
EVAL_SEQUENCE_LENGTH = 256

PAD_ID = -1

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_trained_tokenizer(
    train_dataset: Union[tf.data.Dataset, str],
    vocab_path: str = SPM_TOKENIZER_VOCAB_PATH,
    vocab_size: int = SPM_TOKENIZER_VOCAB_SIZE,
    max_corpus_chars: int = MAX_CORPUS_CHARS,
):
  """Returns an SPM tokenizer trained on the train dataset.

  Args:
    train_dataset: The training dataset to train the tokenizer on.
    vocab_path: The path to the vocabulary file.
    vocab_size: The size of the vocabulary.
    max_corpus_chars: The maximum number of characters to use for training the
      tokenizer.

  Returns:
    A tokenizer trained on the train dataset.
  """
  tokenizer = spm_tokenizer.load_or_train_tokenizer(
      dataset=train_dataset,
      vocab_path=vocab_path,
      vocab_size=vocab_size,
      max_corpus_chars=max_corpus_chars,
      data_keys=None,
  )
  return tokenizer


def split_and_pad(
    dataset: tf.data.Dataset,
    sequence_length: int,
    padded_shapes=None,
    padding_id=PAD_ID,
):
  """Splits a tf.data.Dataset into sequences of length sequence_length and padding if len(dataset) is not divisible by the sequence_length.

  Args:
    dataset: tf.data.Dataset
    sequence_length: length of sequences in resulting batched dataset
    padded_shapes: shapes of the padded batches
    padding_id: value for padding, for elements in new batch

  Returns:
  """
  batched_dataset = dataset.batch(sequence_length, drop_remainder=False)

  # tf.data.Dataset.padded.batch pads elements in the batch so we call it
  # again with batch_size=1 to pad each element in original batch.
  padded_batched_dataset = batched_dataset.padded_batch(
      1, padded_shapes=padded_shapes, padding_values=padding_id
  )

  # Remove extra dimension resulting from the batch_size=1.
  padded_batched_dataset = padded_batched_dataset.unbatch()

  return padded_batched_dataset


def fetch_dataset_into_arrays(
    dataset: tf.data.Dataset,
) -> dict[str, np.ndarray]:
  """Fetches a tf.data.Dataset into numpy arrays.

  Args:
    dataset: tf.data.Dataset where each element is a dictionary with keys
      'inputs' and 'targets'.

  Returns:
    A dictionary with keys 'inputs' and 'targets' and values numpy arrays.
  """
  input_array = []
  target_array = []
  for element in dataset:
    input_array.append(element['inputs'].numpy())
    target_array.append(element['targets'].numpy())
  return {'inputs': np.array(input_array), 'targets': np.array(target_array)}


def get_wikitext103_dataset(
    tokenizer_vocab_path: str = SPM_TOKENIZER_VOCAB_PATH,
    vocab_size: int = SPM_TOKENIZER_VOCAB_SIZE,
    sequence_length: int = SEQUENCE_LENGTH,
    eval_sequence_length: int = EVAL_SEQUENCE_LENGTH,
    data_dir: str = DATA_DIR,
) -> tuple[
    dict[str, np.ndarray[Any, np.dtype]],
    dict[str, np.ndarray[Any, np.dtype]],
    dict[str, np.ndarray[Any, np.dtype]],
]:
  """Returns wikitext-103 dataset.

  Args:
    tokenizer_vocab_path: The path to the vocabulary file.
    vocab_size: The size of the vocabulary.
    sequence_length: The length of the sequences.
    eval_sequence_length: The length of the sequences for evaluation.
    data_dir: The directory containing the wikitext-103 data.

  Returns:
    train_dataset, valid_dataset, test_dataset
  """
  train_path = os.path.join(data_dir, TRAIN_FILENAME)
  valid_path = os.path.join(data_dir, VALID_FILENAME)
  test_path = os.path.join(data_dir, TEST_FILENAME)

  # Get TextLineDataset from raw files
  train_text_dataset = tf.data.TextLineDataset(train_path)
  valid_text_dataset = tf.data.TextLineDataset(valid_path)
  test_text_dataset = tf.data.TextLineDataset(test_path)

  # Tokenize data
  tokenizer = get_trained_tokenizer(
      train_text_dataset, tokenizer_vocab_path, vocab_size
  )
  train_dataset_tokenized = train_text_dataset.map(tokenizer.tokenize)
  valid_dataset_tokenized = valid_text_dataset.map(tokenizer.tokenize)
  test_dataset_tokenized = test_text_dataset.map(tokenizer.tokenize)

  # Flatten datasets into tokens so that they can be batched in sequences of
  # length sequence_length.
  flattened_train_dataset = train_dataset_tokenized.flat_map(
      tf.data.Dataset.from_tensor_slices
  )
  flattened_valid_dataset = valid_dataset_tokenized.flat_map(
      tf.data.Dataset.from_tensor_slices
  )
  flattened_test_dataset = test_dataset_tokenized.flat_map(
      tf.data.Dataset.from_tensor_slices
  )

  # Divide data in sequences of length sequence_length.
  train_dataset_sequences = split_and_pad(
      flattened_train_dataset,
      sequence_length,
      padded_shapes=sequence_length,
  )

  valid_dataset_sequences = split_and_pad(
      flattened_valid_dataset,
      eval_sequence_length,
      padded_shapes=eval_sequence_length,
  )
  test_dataset_sequences = split_and_pad(
      flattened_test_dataset,
      eval_sequence_length,
      padded_shapes=eval_sequence_length,
  )

  # Split the sequences into inputs and targets.
  train_dataset_sequences = train_dataset_sequences.map(
      lambda x: {'inputs': x, 'targets': x}, num_parallel_calls=AUTOTUNE
  ).prefetch(AUTOTUNE)
  valid_dataset_sequences = valid_dataset_sequences.map(
      lambda x: {'inputs': x, 'targets': x}, num_parallel_calls=AUTOTUNE
  ).prefetch(AUTOTUNE)
  test_dataset_sequences = test_dataset_sequences.map(
      lambda x: {'inputs': x, 'targets': x}, num_parallel_calls=AUTOTUNE
  ).prefetch(AUTOTUNE)

  # Fetch the datasets into arrays.
  train_dataset = fetch_dataset_into_arrays(train_dataset_sequences)
  valid_dataset = fetch_dataset_into_arrays(valid_dataset_sequences)
  test_dataset = fetch_dataset_into_arrays(test_dataset_sequences)

  return train_dataset, valid_dataset, test_dataset
