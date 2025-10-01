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

"""Module for loading tokenized FineWeb-Edu 10BT sample dataset.

This dataset is subset of 10B gpt2 tokens randomly sampled from the whole
Fineweb-Edu dataset. We split this 10B dataset into train and validation sets.

Source:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/sample-10BT.
"""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import fineweb_edu_10b_input_pipeline as input_pipeline
import jax
from ml_collections.config_dict import config_dict
import numpy as np

PAD_ID = input_pipeline.PAD_ID
VOCAB_SIZE = input_pipeline.VOCAB_SIZE
TRAIN_SIZE = 8_959_866_880  # Number of tokens.
Dataset = data_utils.Dataset

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        sequence_length=512,
        max_target_length=512,
        max_eval_target_length=512,
        eval_sequence_length=512,
        input_shape=(512,),
        output_shape=(VOCAB_SIZE,),
        train_size=TRAIN_SIZE,  # Number of tokens.
        vocab_size=VOCAB_SIZE,
    )
)

METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_inputs': True,
    'causal': True,
    'pad_token': PAD_ID,
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


def get_fineweb_edu(
    shuffle_rng,
    batch_size: int,
    eval_batch_size: int = None,
    hps: config_dict.ConfigDict = None,
    pad_id: int = PAD_ID,
) -> Dataset:
  """Returns Fineweb-EDU 10B Dataset.

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
  n_devices = jax.local_device_count()

  if batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size
        )
    )

  if eval_batch_size is None:
    eval_batch_size = batch_size

  if batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size
        )
    )
  if eval_batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size
        )
    )

  per_host_batch_size = int(batch_size / process_count)
  per_host_eval_batch_size = int(eval_batch_size / process_count)

  if per_host_batch_size % n_devices != 0:
    raise ValueError(
        'per_host_batch_size={} must be divisible by n_devices={}.'.format(
            per_host_batch_size, n_devices
        )
    )
  if per_host_eval_batch_size % n_devices != 0:
    raise ValueError(
        'per_host_eval_batch_size={} must be divisible by n_devices={}.'.format(
            per_host_eval_batch_size, n_devices
        )
    )

  train_dataset, eval_train_dataset, valid_dataset = (
      input_pipeline.get_fineweb_edu_dataset(
          hps,
          train_batch_size=per_host_batch_size,
          valid_batch_size=per_host_eval_batch_size,
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

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
