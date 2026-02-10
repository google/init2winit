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

"""ImageNet input pipeline with MLPerf preprocessing."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import imagenet_dataset
from init2winit.dataset_lib import mlperf_input_pipeline
import jax
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(224, 224, 3),
    output_shape=(1000,),
    train_size=1281167,
    valid_size=50000,
    test_size=10000,  # ImageNet-v2.
    use_imagenetv2_test=True))

METADATA = {
    'apply_one_hot_in_loss': False,
}


def get_mlperf_imagenet(rng,
                        batch_size,
                        eval_batch_size,
                        hps=None):
  """Data generators for imagenet.

  Args:
    rng: RNG seed that is split into a shuffle seed and a seed that is folded
      into a per-example seed.
    batch_size: the *global* batch size used for training.
    eval_batch_size: the *global* batch size used for evaluation.
    hps: the hparams for the experiment, only required field is valid_size.

  Returns:
    A data_utils.Dataset for the MLPerf version of ImageNet.
  """
  if batch_size % jax.device_count() != 0:
    raise ValueError(
        'Require batch_size % jax.device_count(), received '
        'batch_size={}, device_count={}.'.format(
            batch_size, jax.device_count()))
  if eval_batch_size % jax.device_count() != 0:
    raise ValueError(
        'Require eval_batch_size % jax.device_count(), received '
        'eval_batch_size={}, device_count={}.'.format(
            eval_batch_size, jax.device_count()))
  host_batch_size = batch_size // jax.process_count()
  eval_host_batch_size = eval_batch_size // jax.process_count()

  max_eval_steps = hps.valid_size // eval_batch_size + 1

  input_dtype = tf.bfloat16
  shuffle_buffer_size = 16384

  train_ds = mlperf_input_pipeline.load_split(
      host_batch_size,
      dtype=input_dtype,
      split='train',
      rng=rng,
      shuffle_size=shuffle_buffer_size)

  eval_train_ds = mlperf_input_pipeline.load_split(
      host_batch_size,
      dtype=input_dtype,
      split='eval_train',
      rng=rng,
      shuffle_size=shuffle_buffer_size)

  eval_ds = mlperf_input_pipeline.load_split(
      eval_host_batch_size,
      dtype=input_dtype,
      split='validation',
      rng=rng,
      shuffle_size=shuffle_buffer_size)

  # We do not have TFRecords of ImageNet-v2 in the same format as the
  # train/validation splits above, so we reuse the same test split from the
  # non-MLPerf pipeline.
  test_ds = None
  if hps.use_imagenetv2_test:
    test_ds = imagenet_dataset.load_split(
        eval_host_batch_size,
        'test',
        hps=hps,
        image_size=224,
        tfds_dataset_name='imagenet_v2/matched-frequency')

  # We cannot use tfds.as_numpy because this calls tensor.numpy() which does an
  # additional copy of the tensor, instead we call tensor._numpy() below.
  def train_iterator_fn():
    return data_utils.iterator_as_numpy(iter(train_ds))

  def eval_train_epoch(num_batches=None):
    if num_batches is None:
      num_batches = 0
    eval_train_iter = iter(eval_train_ds)
    np_iter = data_utils.iterator_as_numpy(
        itertools.islice(eval_train_iter, num_batches))
    for batch in np_iter:
      yield data_utils.maybe_pad_batch(batch, eval_host_batch_size)

  def valid_epoch(num_batches=None):
    if num_batches is None:
      num_batches = max_eval_steps
    valid_iter = iter(eval_ds)
    np_iter = data_utils.iterator_as_numpy(
        itertools.islice(valid_iter, num_batches))
    for batch in np_iter:
      yield data_utils.maybe_pad_batch(batch, eval_host_batch_size)

  def test_epoch(num_batches=None):
    if test_ds:
      test_iter = iter(test_ds)
      np_iter = data_utils.iterator_as_numpy(
          itertools.islice(test_iter, num_batches))
      for batch in np_iter:
        yield data_utils.maybe_pad_batch(batch, eval_host_batch_size)
    else:
      # pylint: disable=unreachable
      return
      yield  # This yield is needed to make this a valid (null) iterator.
      # pylint: enable=unreachable

  return data_utils.Dataset(
      train_iterator_fn,
      eval_train_epoch,
      valid_epoch,
      test_epoch)


def get_fake_batch(hps):
  return {
      'inputs':
          np.ones((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype),
      'targets':
          np.ones((hps.batch_size, *hps.output_shape), dtype=hps.model_dtype),
      'weights':
          np.ones((hps.batch_size,), dtype=hps.model_dtype),
  }
