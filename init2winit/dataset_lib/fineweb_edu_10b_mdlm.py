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

"""MDLM variant of FineWeb-Edu 10B dataset.

Wraps the standard fineweb_edu_10b dataset with metadata appropriate for
masked diffusion language modeling (no shifting, bidirectional).
"""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import fineweb_edu_10b
from init2winit.dataset_lib import fineweb_edu_10b_input_pipeline as input_pipeline
import jax

PAD_ID = input_pipeline.PAD_ID
VOCAB_SIZE = input_pipeline.VOCAB_SIZE

DEFAULT_HPARAMS = fineweb_edu_10b.DEFAULT_HPARAMS
Dataset = data_utils.Dataset


METADATA = {
    'apply_one_hot_in_loss': False,
    'shift_inputs': False,
    'causal': False,
    'pad_token': PAD_ID,
}


def get_fineweb_edu_mdlm(
    shuffle_rng, batch_size, eval_batch_size=None, hps=None, pad_id=PAD_ID
):
  """Returns FineWeb-Edu 10B Dataset without input shifting for MDLM."""
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
          shift=False,
      )
  )

  def train_iterator_fn():
    for batch in train_dataset:
      yield fineweb_edu_10b.add_weights_to_batch(
          data_utils.tf_to_numpy(batch), pad_id
      )

  def eval_train_epoch(num_batches=None):
    for batch in itertools.islice(iter(eval_train_dataset), num_batches):
      yield fineweb_edu_10b.add_weights_to_batch(
          data_utils.tf_to_numpy(batch), pad_id
      )

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(iter(valid_dataset), num_batches):
      yield fineweb_edu_10b.add_weights_to_batch(
          data_utils.tf_to_numpy(batch), pad_id
      )

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
