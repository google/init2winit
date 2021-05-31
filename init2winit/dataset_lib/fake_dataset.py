# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

"""Fake ImageNet input pipeline."""

from init2winit.dataset_lib import data_utils
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


TRAIN_IMAGES = 1281167
EVAL_IMAGES = 50000


NUM_CLASSES = 1000
IMAGE_SIZE = 224


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(224, 224, 3),
    output_shape=(NUM_CLASSES,),
    train_size=TRAIN_IMAGES,
    valid_size=EVAL_IMAGES))

METADATA = {
    'apply_one_hot_in_loss': False,
}


def get_fake_batch(batch_size, input_shape, num_classes):
  """Generate Gaussian random images."""
  train_input_shape = (batch_size, *input_shape)
  images = np.random.normal(size=train_input_shape).astype(np.float32)
  labels = jax.nn.one_hot(
      np.zeros((batch_size,)), num_classes, dtype=jnp.int32)
  batch = {
      'inputs': images,
      'targets': labels,
      'weights': jnp.ones(batch_size, dtype=images.dtype),
  }
  return batch


def get_fake(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Data generators for imagenet."""
  del shuffle_rng
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  fake_train_batch = get_fake_batch(
      per_host_batch_size, hps.input_shape, hps.output_shape[0])
  fake_test_batch = get_fake_batch(
      per_host_eval_batch_size, hps.input_shape, hps.output_shape[0])

  def train_iterator_fn():
    while True:
      yield fake_train_batch

  def valid_epoch(epoch, num_batches=None):
    del num_batches
    del epoch
    # Note that we do // beacuse we do not support partial batching for the fake
    # dataset.
    for _ in range(hps.valid_size // eval_batch_size):
      yield fake_test_batch

  # pylint: disable=unreachable
  def eval_train_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable
  # pylint: disable=unreachable

  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  return data_utils.Dataset(
      train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)

