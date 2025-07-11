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

"""Tests for the ogbg-molpcba dataset."""

import itertools

from init2winit.dataset_lib.datasets import get_dataset
from init2winit.dataset_lib.datasets import get_dataset_hparams
import jax.random
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()

NUM_LABELS = 2
NORMAL_LABELS = np.array([1, 1]).astype('float32')
NAN_LABELS = np.array([np.nan, 1]).astype('float32')

NUMS_NODES = [4, 7, 15, 6]
NUMS_EDGES = [4, 6, 20, 9]
LABELS = [NORMAL_LABELS, NAN_LABELS, NORMAL_LABELS, NAN_LABELS]

BATCH_SIZE = 2
NODES_SIZE_MULTIPLIER = 8
EDGES_SIZE_MULTIPLIER = 40


def _make_graph(num_nodes, num_edges, labels):
  return {
      'num_edges':
          np.array([num_edges]),
      'num_nodes':
          np.array([num_nodes]),
      'edge_index':
          np.array(
              list(
                  itertools.islice(
                      itertools.combinations(range(num_nodes), 2), num_edges))),
      'edge_feat':
          np.ones((num_edges, 3)).astype('float32'),
      'node_feat':
          np.ones((num_nodes, 9)).astype('float32'),
      'labels':
          labels
  }


def _as_dataset(*args, **kwargs):
  """Creates a mock TFDS graphs dataset."""
  del args, kwargs

  def get_iter():
    return (
        _make_graph(num_nodes, num_edges, labels)
        for num_nodes, num_edges, labels in zip(NUMS_NODES, NUMS_EDGES, LABELS))

  return tf.data.Dataset.from_generator(
      get_iter,
      output_signature={
          'edge_feat': tf.TensorSpec(shape=(None, 3), dtype=np.float32),
          'edge_index': tf.TensorSpec(shape=(None, 2), dtype=np.int64),
          'labels': tf.TensorSpec(shape=(NUM_LABELS,), dtype=np.float32),
          'node_feat': tf.TensorSpec(shape=(None, 9), dtype=np.float32),
          'num_edges': tf.TensorSpec(shape=(1,), dtype=np.int64),
          'num_nodes': tf.TensorSpec(shape=(1,), dtype=np.int64),
      })


def _get_dataset(shuffle_seed, additional_hps=None):
  """Loads the ogbg-molpcba dataset using mock data."""
  with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
    ds = 'ogbg_molpcba'
    dataset_builder = get_dataset(ds)
    hps_dict = get_dataset_hparams(ds).to_dict()
    if additional_hps is not None:
      hps_dict.update(additional_hps)
    hps = config_dict.ConfigDict(hps_dict)
    hps.train_size = 4
    hps.valid_size = 4
    hps.test_size = 4
    hps.avg_nodes_per_graph = NODES_SIZE_MULTIPLIER
    hps.avg_edges_per_graph = EDGES_SIZE_MULTIPLIER
    hps.batch_nodes_multiplier = 1.0
    hps.batch_edges_multiplier = 1.0
    batch_size = BATCH_SIZE
    eval_batch_size = BATCH_SIZE
    dataset = dataset_builder(
        shuffle_rng=shuffle_seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hps=hps)
    return dataset


class OgbgMolpcbaTest(tf.test.TestCase):
  """Tests data loading for the ogbg-molpcba dataset."""

  def test_get_batch_pads_correctly(self):
    """Tests that data batches are padded correctly."""
    # Use validation batch to maintain the example order, since the train batch
    # will be shuffled.
    dataset = _get_dataset(jax.random.PRNGKey(0))

    batch = next(dataset.valid_epoch())
    inputs = batch['inputs']

    # The first two graphs are in the first batch
    self.assertNDArrayNear(inputs.n_node[:2], np.array(NUMS_NODES[:2]), 1e-3)

    # The graphs are padded to the right size
    self.assertEqual(inputs.n_node.shape[0], BATCH_SIZE + 1)
    self.assertEqual(
        np.sum(inputs.n_node), BATCH_SIZE * NODES_SIZE_MULTIPLIER + 1)
    self.assertEqual(np.sum(inputs.n_edge), BATCH_SIZE * EDGES_SIZE_MULTIPLIER)

    # Weights are zero at NaN labels and in padded examples
    self.assertNDArrayNear(batch['weights'],
                           np.array([[1, 1], [0, 1], [0, 0]]), 1e-3)
    self.assertFalse(np.any(np.isnan(batch['targets'])))

  def test_train_shuffle_is_deterministic(self):
    """Tests that shuffling of the train split is deterministic."""
    dataset = _get_dataset(jax.random.PRNGKey(0))
    dataset_same = _get_dataset(jax.random.PRNGKey(0))
    dataset_different = _get_dataset(jax.random.PRNGKey(1))

    batch = next(dataset.train_iterator_fn())
    batch_same = next(dataset_same.train_iterator_fn())
    batch_different = next(dataset_different.train_iterator_fn())

    self.assertAllClose(batch['inputs'], batch_same['inputs'])
    self.assertNotAllClose(batch['inputs'], batch_different['inputs'])

  def test_add_virtual_node(self):
    """Tests that adding a virtual node works correctly."""
    dataset = _get_dataset(jax.random.PRNGKey(0), {'add_virtual_node': True})

    batch = next(dataset.valid_epoch())
    inputs = batch['inputs']
    num_nodes = np.array(NUMS_NODES[0])
    num_edges = np.array(NUMS_EDGES[0])

    self.assertNDArrayNear(
        inputs.n_node[0], np.array(num_nodes + 1), 1e-3)
    self.assertNDArrayNear(
        inputs.n_edge[0], np.array(num_edges + num_nodes), 1e-3)
    self.assertNDArrayNear(
        inputs.edges[num_edges:num_edges + num_nodes],
        np.zeros_like(inputs.edges[num_edges:num_edges + num_nodes]), 1e-3)
    self.assertNDArrayNear(inputs.nodes[num_nodes],
                           np.zeros_like(inputs.nodes[num_nodes]), 1e-3)

  def test_add_bidirectional_edges(self):
    """Tests that adding bidirectional edges works correctly."""
    dataset = _get_dataset(
        jax.random.PRNGKey(0), {'add_bidirectional_edges': True})

    batch = next(dataset.valid_epoch())
    inputs = batch['inputs']
    num_nodes = np.array(NUMS_NODES[0])
    num_edges = np.array(NUMS_EDGES[0])

    self.assertNDArrayNear(
        inputs.n_node[0], np.array(num_nodes), 1e-3)
    self.assertNDArrayNear(
        inputs.n_edge[0], np.array(num_edges * 2), 1e-3)

  def test_add_self_loops(self):
    """Tests that adding self loops works correctly."""
    dataset = _get_dataset(jax.random.PRNGKey(0), {'add_self_loops': True})

    batch = next(dataset.valid_epoch())
    inputs = batch['inputs']
    num_nodes = np.array(NUMS_NODES[0])
    num_edges = np.array(NUMS_EDGES[0])

    self.assertNDArrayNear(
        inputs.n_node[0], np.array(num_nodes), 1e-3)
    self.assertNDArrayNear(
        inputs.n_edge[0], np.array(num_edges + num_nodes), 1e-3)
    self.assertNDArrayNear(
        inputs.edges[num_edges:num_edges + num_nodes],
        np.zeros_like(inputs.edges[num_edges:num_edges + num_nodes]), 1e-3)


if __name__ == '__main__':
  tf.test.main()
