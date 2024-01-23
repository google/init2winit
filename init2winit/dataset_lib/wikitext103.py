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

"""Module containing hyperparameters, metadata and dataset getter for Wikitext-2 dataset."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import wikitext103_input_pipeline as input_pipeline
from init2winit.dataset_lib.data_utils import Dataset
from init2winit.dataset_lib.wikitext2_input_pipeline import PAD_ID
import jax
from ml_collections.config_dict import config_dict
import numpy as np

VOCAB_SIZE = 267735

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        sequence_length=128,
        max_target_length=128,
        max_eval_target_length=128,
        eval_sequence_length=128,
        input_shape=(128,),
        output_shape=(VOCAB_SIZE,),
        vocab_size=VOCAB_SIZE,
        train_size=800210,  # Number of sequences.
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
    hps: config_dict.ConfigDict = None) -> Dataset:
  """Returns Wikitext-103 Dataset.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: training batch size
    eval_batch_size: validation batch size
    hps: Hyper parameters

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
          # TODO(b/280322542): this should be jax.random.bits(shuffle_rng)
          shuffle_seed=jax.random.key_data(shuffle_rng)[0],
      )
  )

  def train_iterator_fn():
    for batch in train_dataset:
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch))

  def eval_train_epoch(num_batches=None):
    for batch in itertools.islice(iter(eval_train_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch))

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(iter(valid_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch))

  def test_epoch(num_batches=None):
    for batch in itertools.islice(iter(test_dataset), num_batches):
      yield add_weights_to_batch(data_utils.tf_to_numpy(batch))

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                 test_epoch)
