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

"""Module for loading tokenized FineWeb-Edu 10BT sample dataset.

Source:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT.

Preprocessing and tokenization script to generate data saved to CNS:
https://github.com/mlcommons/algorithmic-efficiency/blob/main/datasets/dataset_setup.py.
"""

# import tensorflow.compat.v2 as tf
import os
from absl import logging
from ml_collections.config_dict import config_dict
import tensorflow as tf

TRAIN_DIR = 'train'
VAL_DIR = 'val'

MAX_CORPUS_CHARS = 1_000_000_000
SHUFFLE_BUFFER_SIZE = 100_000
VOCAB_SIZE = 50_257

PAD_ID = tf.constant(-1, dtype=tf.int64)
# PAD_ID = -1

AUTOTUNE = tf.data.experimental.AUTOTUNE


def batch_with_padding(
    dataset: tf.data.Dataset,
    batch_size,
    padded_shapes=None,
    padding_id=PAD_ID,
):
  """Batches a tf.data.Dataset and adds padding if len(dataset) is not divisible by the batch size.

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
      1, padded_shapes=padded_shapes, padding_values=padding_id
  )

  # Remove extra dimension resulting from the batch_size=1.
  padded_batched_dataset = padded_batched_dataset.unbatch()

  return padded_batched_dataset


def get_fineweb_edu_dataset(
    hps: config_dict.ConfigDict,
    train_batch_size: int,
    valid_batch_size: int,
    shuffle_seed: int,
    shift: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Returns wikitext-103 dataset.

  Args:
    hps: Dataset hyper parameters.
    train_batch_size: Batch size for train iterations
    valid_batch_size: Batch size for validation iterations
    shuffle_seed: seed for shuffling dataset sequences
    shift: If True (default), inputs = x[:-1], targets = x[1:] (AR mode). If
      False, inputs = targets = x (MDLM mode).

  Returns:
    train_dataset, eval_train_dataset, valid_dataset, test_dataset
  """
  logging.info('batch_size in input_pipeline: %s', train_batch_size)
  train_path = os.path.join(DATA_DIR, TRAIN_DIR)
  val_path = os.path.join(DATA_DIR, VAL_DIR)

  # Load datasets and cast to int32.
  train_dataset = tf.data.Dataset.load(train_path)
  val_dataset = tf.data.Dataset.load(val_path)

  # pack sequences
  train_tokens = train_dataset.flat_map(tf.data.Dataset.from_tensor_slices)
  val_tokens = val_dataset.flat_map(tf.data.Dataset.from_tensor_slices)

  # split into sequences
  seq_batch_len = hps.sequence_length + 1 if shift else hps.sequence_length
  eval_seq_batch_len = (
      hps.eval_sequence_length + 1 if shift else hps.eval_sequence_length
  )
  train_sequences_dataset = train_tokens.batch(
      seq_batch_len, drop_remainder=True
  )
  eval_train_sequences_dataset = train_tokens.batch(
      eval_seq_batch_len, drop_remainder=True
  )
  val_sequences_dataset = val_tokens.batch(
      eval_seq_batch_len, drop_remainder=True
  )

  # Split the sequences into inputs and targets.
  if shift:
    map_fn = lambda x: {
        'inputs': x['input_ids'][: hps.sequence_length],
        'targets': x['input_ids'][1:],
    }
    eval_map_fn = lambda x: {
        'inputs': x['input_ids'][: hps.eval_sequence_length],
        'targets': x['input_ids'][1:],
    }
  else:
    map_fn = lambda x: {
        'inputs': x['input_ids'][: hps.sequence_length],
        'targets': x['input_ids'][: hps.sequence_length],
    }
    eval_map_fn = lambda x: {
        'inputs': x['input_ids'][: hps.eval_sequence_length],
        'targets': x['input_ids'][: hps.eval_sequence_length],
    }
  train_sequences_dataset = train_sequences_dataset.map(
      map_fn, num_parallel_calls=AUTOTUNE
  )
  eval_train_sequences_dataset = eval_train_sequences_dataset.map(
      eval_map_fn, num_parallel_calls=AUTOTUNE
  )
  val_sequences_dataset = val_sequences_dataset.map(
      eval_map_fn, num_parallel_calls=AUTOTUNE
  )

  # Shuffle the train sequences.
  train_sequences_dataset = train_sequences_dataset.shuffle(
      SHUFFLE_BUFFER_SIZE, seed=shuffle_seed
  )

  # Perform batching for training, validation and testing.
  # Make training data repeat indefinitely.
  train_sequences_dataset = train_sequences_dataset.repeat()
  train_dataset = train_sequences_dataset.batch(
      train_batch_size, drop_remainder=False
  ).prefetch(tf.data.experimental.AUTOTUNE)

  # Use padded batches for eval_train, validation and test_datasets since the
  # sequences do not repeat indefintely.
  eval_train_dataset = batch_with_padding(
      eval_train_sequences_dataset,
      train_batch_size,
      padded_shapes={
          'inputs': (train_batch_size, None),
          'targets': (train_batch_size, None),
      },
  ).prefetch(tf.data.experimental.AUTOTUNE)

  valid_dataset = batch_with_padding(
      val_sequences_dataset,
      valid_batch_size,
      padded_shapes={
          'inputs': (valid_batch_size, None),
          'targets': (valid_batch_size, None),
      },
  ).prefetch(tf.data.experimental.AUTOTUNE)

  return train_dataset, eval_train_dataset, valid_dataset
