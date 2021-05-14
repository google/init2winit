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

"""Unit tests for mt_pipeline.py."""

import itertools
import os
import pathlib
import tempfile

from absl.testing import absltest
from init2winit.dataset_lib import mt_pipeline
from init2winit.dataset_lib import translate_wmt
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

# We just use different values here to verify that the input pipeline uses the
# the correct value for the 3 different datasets.
_TARGET_LENGTH = 32
_EVAL_TARGET_LENGTH = 48
_PREDICT_TARGET_LENGTH = 64


class MTPipelineTest(absltest.TestCase):

  def _get_datasets(self, shuffle_seed=None, pack_examples=False):
    config = translate_wmt.DEFAULT_HPARAMS
    config.vocab_size = 32
    config.max_corpus_chars = 1000
    config.max_target_length = _TARGET_LENGTH
    config.max_eval_target_length = _EVAL_TARGET_LENGTH
    config.max_predict_length = _PREDICT_TARGET_LENGTH
    config.pack_examples = pack_examples

    vocab_path = os.path.join(tempfile.mkdtemp(), 'sentencepiece_model')

    # Go one directory up to the root of the init2winit directory.
    root_dir = pathlib.Path(__file__).parents[1]
    data_dir = str(root_dir) + '/.tfds/metadata'  # pylint: disable=unused-variable

    np.random.seed(0)  # set the seed so the mock data is deterministic.
    with tfds.testing.mock_data(num_examples=128):
      train_ds, eval_ds, predict_ds = mt_pipeline.get_wmt_datasets(
          config=config,
          shuffle_seed=shuffle_seed,
          n_devices=2,
          per_host_batch_size=4,
          per_host_eval_batch_size=4,
          vocab_path=vocab_path)
    return train_ds, eval_ds, predict_ds

  def test_train_ds_golden(self):
    # train dataset without packing
    train_ds, _, _ = self._get_datasets(pack_examples=False)
    expected_shape = [4, _TARGET_LENGTH]  # 4 batch_size.
    for batch in train_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })

  def test_train_ds_packed(self):
    # packed train dataset
    train_ds, _, _ = self._get_datasets(pack_examples=True)
    expected_shape = [4, _TARGET_LENGTH]  # 4 batch_size.
    # For training we pack multiple short examples in one example.
    # *_position and *_segmentation indicate the boundaries.
    for batch in train_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'inputs_position': expected_shape,
          'inputs_segmentation': expected_shape,
          'targets': expected_shape,
          'targets_position': expected_shape,
          'targets_segmentation': expected_shape,
      })

  def test_train_ds_determinism(self):
    # unpacked train dataset
    train_ds, _, _ = self._get_datasets(shuffle_seed=12345)
    train_ds_copy, _, _ = self._get_datasets(shuffle_seed=12345)
    batch_idx_to_test = 1
    train_ds_batch = next(
        itertools.islice(train_ds, batch_idx_to_test, batch_idx_to_test + 1))
    train_ds_copy_batch = next(
        itertools.islice(train_ds_copy, batch_idx_to_test,
                         batch_idx_to_test + 1))
    self.assertTrue(
        jnp.array_equal(train_ds_batch['inputs'],
                        train_ds_copy_batch['inputs']))
    self.assertTrue(
        jnp.array_equal(train_ds_batch['targets'],
                        train_ds_copy_batch['targets']))

  def test_eval_ds(self):
    _, eval_ds, _ = self._get_datasets()
    expected_shape = [4, _EVAL_TARGET_LENGTH]  # 4 batch_size.
    for batch in eval_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })

  def test_predict_ds(self):
    _, _, predict_ds = self._get_datasets()
    expected_shape = [4, _PREDICT_TARGET_LENGTH]  # 4 batch_size.
    for batch in predict_ds.take(3):
      self.assertEqual({k: v.shape.as_list() for k, v in batch.items()}, {
          'inputs': expected_shape,
          'targets': expected_shape,
      })


if __name__ == '__main__':
  absltest.main()
