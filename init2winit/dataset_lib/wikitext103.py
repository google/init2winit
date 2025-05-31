# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Module containing hyperparameters, metadata and dataset getter for Wikitext-103 dataset."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import wikitext103_input_pipeline as input_pipeline
from init2winit.dataset_lib import wikitext2_input_pipeline
import jax
from ml_collections.config_dict import config_dict
import numpy as np

PAD_ID = wikitext2_input_pipeline.PAD_ID
Dataset = data_utils.Dataset

VOCAB_SIZE = 267735

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        sequence_length=128,
        max_target_length=128,
        max_eval_target_length=128,
        eval_sequence_length=128,
        input_shape=(128,),
        output_shape=(input_pipeline.WORD_VOCAB_SIZE,),
        train_size=800210,  # Number of sequences.
        tokenizer='word',
        tokenizer_vocab_path=None,
        vocab_size=input_pipeline.WORD_VOCAB_SIZE,
    ))


METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_inputs': True,
    'causal': True,
    'pad_token': -1,
}


def add_weights_to_batch(batch, pad_id: int = PAD_ID):
  """Add weights for the input values so that paddings have 0 weight.

  Args:
    batch: Batch represented by dict containing 'inputs' and 'targets'.
    pad_id: Value for 'inputs' that will have weight 0.

  Returns:
    batch with weights
  """
  batch['weights'] = np.where(batch['inputs'] == pad_id, 0.0, 1.0)
  return batch


def get_wikitext103(
    shuffle_rng,
    batch_size: int,
    eval_batch_size: int = None,
    hps: config_dict.ConfigDict = None,
    pad_id: int = PAD_ID) -> Dataset:
  """Returns Wikitext-103 Dataset.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: training batch size
    eval_batch_size: validation batch size
    hps: Hyper parameters
    pad_id: Value for 'inputs' that will have weight 0.

  Returns:
    Dataset

  Raises:
    ValueError: If batch_size is not divisible by jax process count.
    ValueError: If eval_batch_size is not divisible by jax process count.
  """
  process_count = jax.process_count()

  if batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size))

  if eval_batch_size is None:
    eval_batch_size = batch_size

  if eval_batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size))

  train_dataset, eval_train_dataset, valid_dataset, test_dataset = (
      input_pipeline.get_wikitext103_dataset(
          hps,
          train_batch_size=batch_size,
          valid_batch_size=eval_batch_size,
          test_batch_size=eval_batch_size,
          shuffle_seed=data_utils.convert_jax_to_tf_random_seed(shuffle_rng),
      )
  )

  def train_iterator_fn():
    for batch in train_dataset:
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch), pad_id)

  def eval_train_epoch(num_batches=None):
    for batch in itertools.islice(iter(eval_train_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch), pad_id)

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(iter(valid_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch), pad_id)

  def test_epoch(num_batches=None):
    for batch in itertools.islice(iter(test_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch), pad_id)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                 test_epoch)
