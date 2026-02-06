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

"""Tests for ogbg_molpcba_preprocessed."""

import logging
import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from init2winit.dataset_lib import datasets
from init2winit.dataset_lib import ogbg_molpcba
from init2winit.dataset_lib import ogbg_molpcba_preprocessed
import jax
import jraph
from ml_collections import config_dict
import numpy as np

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


class OgbgMolpcbaPreprocessedTest(absltest.TestCase):
  """Tests for ogbg_molpcba_preprocessed.

  This test mocks ogbg_molpcba.get_ogbg_molpcba to return a dummy dataset
  iterator, and checks that generate_and_save_dataset and the
  ogbg_molpcba_preprocessed dataset loader work as expected by verifying that
  the data can be loaded and iterated over, and that the data matches the
  original dummy dataset.
  """

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.dataset_path = os.path.join(self.test_dir, 'test_dataset')

    # Create dummy batch for mocking
    self.dummy_batch = {
        'inputs': jraph.GraphsTuple(
            nodes=np.random.randn(2, 2).astype(np.float32),
            edges=np.random.randn(2, 2).astype(np.float32),
            receivers=np.array([0, 1], dtype=np.int32),
            senders=np.array([0, 1], dtype=np.int32),
            n_node=np.array([2], dtype=np.int32),
            n_edge=np.array([2], dtype=np.int32),
            globals=np.zeros((1, 2), dtype=np.float32),
        ),
        'targets': np.random.randn(1, 128).astype(np.float32),
        'weights': np.random.randn(1, 128).astype(np.float32),
    }

    # Mock ogbg_molpcba.get_ogbg_molpcba to return a dummy builder
    self.patcher = mock.patch.object(ogbg_molpcba, 'get_ogbg_molpcba')
    self.mock_get_ogbg = self.patcher.start()

    mock_ds = mock.Mock()
    # Return iterator that yields dummy_batch infinitely (or enough times)
    mock_ds.train_iterator_fn.return_value = iter([self.dummy_batch] * 100)
    self.mock_get_ogbg.return_value = mock_ds

    # Generate a small dataset
    hps = config_dict.ConfigDict(ogbg_molpcba.DEFAULT_HPARAMS)
    logging.info('Generating dataset...')
    ogbg_molpcba_preprocessed.generate_and_save_dataset(
        hps=hps,
        output_path=self.dataset_path,
        num_steps=5,
        batch_size=2,
        seed=0,
    )
    logging.info('Dataset generated at %s', self.dataset_path)

  def tearDown(self):
    self.patcher.stop()
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_load_dataset_train_iterator(self):
    """Tests that the preprocessed dataset can be loaded and iterated over."""

    hps = config_dict.ConfigDict(ogbg_molpcba.DEFAULT_HPARAMS)
    hps.dataset_path = self.dataset_path

    # Call the loader directly or via datasets registry
    logging.info('Loading dataset...')
    loader = datasets.get_dataset('ogbg_molpcba_preprocessed')
    ds = loader(
        shuffle_rng=jax.random.PRNGKey(0),
        batch_size=2,
        eval_batch_size=2,
        hps=hps,
    )

    # Verify train iterator
    train_iter = ds.train_iterator_fn()
    batch = next(train_iter)
    self.assertTrue(hasattr(batch['inputs'], 'nodes'))
    self.assertTrue(hasattr(batch['inputs'], 'edges'))
    self.assertTrue(hasattr(batch['inputs'], 'senders'))
    self.assertTrue(hasattr(batch['inputs'], 'receivers'))
    self.assertNotEmpty(batch['inputs'].nodes)

    def assert_arrays_equal(arr1, arr2):
      np.testing.assert_allclose(arr1, arr2)

    assert_arrays_equal(batch['inputs'].nodes, self.dummy_batch['inputs'].nodes)
    assert_arrays_equal(batch['inputs'].edges, self.dummy_batch['inputs'].edges)
    assert_arrays_equal(
        batch['inputs'].senders, self.dummy_batch['inputs'].senders
    )
    assert_arrays_equal(
        batch['inputs'].receivers, self.dummy_batch['inputs'].receivers
    )
    assert_arrays_equal(
        batch['inputs'].globals, self.dummy_batch['inputs'].globals
    )
    assert_arrays_equal(
        batch['inputs'].n_node, self.dummy_batch['inputs'].n_node
    )
    assert_arrays_equal(
        batch['inputs'].n_edge, self.dummy_batch['inputs'].n_edge
    )

    assert_arrays_equal(batch['targets'], self.dummy_batch['targets'])
    assert_arrays_equal(batch['weights'], self.dummy_batch['weights'])

    # Verify that we can iterate multiple times (it loads from disk)
    batch2 = next(train_iter)
    self.assertTrue(hasattr(batch2['inputs'], 'nodes'))


if __name__ == '__main__':
  absltest.main()
