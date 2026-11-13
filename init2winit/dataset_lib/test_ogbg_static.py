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

"""Verification test comparing dynamic and static batching in ogbg_molpcba.

Uses uniform-size graphs so that dynamically_batch packs exactly the same
number of graphs per shard as the static approach, making outputs directly
comparable.
"""

import itertools
from init2winit.dataset_lib import ogbg_molpcba
import jax
import jraph
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()


NUM_LABELS = 128
BATCH_SIZE = 4
NUM_SHARDS = 1
NUM_NODES = 10
NUM_EDGES = 12
NUM_GRAPHS = 16
BENCH_GRAPHS = 256

NORMAL_LABELS = np.random.RandomState(42).randn(NUM_LABELS).astype('float32')
NAN_LABELS = NORMAL_LABELS.copy()
NAN_LABELS[0] = np.nan


def _make_graph(seed, labels):
  rng = np.random.RandomState(seed)
  edge_pairs = list(
      itertools.islice(itertools.combinations(range(NUM_NODES), 2), NUM_EDGES)
  )
  return {
      'num_edges': np.array([NUM_EDGES]),
      'num_nodes': np.array([NUM_NODES]),
      'edge_index': np.array(edge_pairs, dtype=np.int64),
      'edge_feat': rng.randn(NUM_EDGES, 3).astype(np.float32),
      'node_feat': rng.randn(NUM_NODES, 9).astype(np.float32),
      'labels': labels,
  }


def _make_dataset():
  graphs = []
  for i in range(NUM_GRAPHS):
    lbl = NAN_LABELS if i % 3 == 1 else NORMAL_LABELS
    graphs.append(_make_graph(seed=i, labels=lbl))
  return graphs


def _as_dataset(*args, **kwargs):
  del args, kwargs
  dataset = _make_dataset()

  def gen():
    yield from dataset

  return tf.data.Dataset.from_generator(
      gen,
      output_signature=_OUTPUT_SIGNATURE,
  )


_OUTPUT_SIGNATURE = {
    'edge_feat': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
    'edge_index': tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
    'labels': tf.TensorSpec(shape=(NUM_LABELS,), dtype=tf.float32),
    'node_feat': tf.TensorSpec(shape=(None, 9), dtype=tf.float32),
    'num_edges': tf.TensorSpec(shape=(1,), dtype=tf.int64),
    'num_nodes': tf.TensorSpec(shape=(1,), dtype=tf.int64),
}


def _make_bench_dataset():
  graphs = []
  for i in range(BENCH_GRAPHS):
    lbl = NAN_LABELS if i % 3 == 1 else NORMAL_LABELS
    graphs.append(_make_graph(seed=i, labels=lbl))
  return graphs


def _as_bench_dataset(*args, **kwargs):
  del args, kwargs
  dataset = _make_bench_dataset()

  def gen():
    yield from dataset

  return tf.data.Dataset.from_generator(
      gen,
      output_signature=_OUTPUT_SIGNATURE,
  )


class VerifyStaticBatchingTest(tf.test.TestCase):
  """Tests that static batching produces the same results as dynamic batching.

  We test the following properties:
  - Same number of batches per epoch.
  - Same graph structure (number of nodes and edges).
  - Same node and edge features.
  - Same senders and receivers.
  - Same labels and weights.
  - Same padding mask.

  Note that static batching would only produce the same results as dynamic
  batching if the graphs all fit into the batch size. otherwise, the results
  will be different as static batching might drop some graphs that dynamic
  batching would split into multiple batches.
  """

  def _get_batches(self, hps):
    """Return batches from a mock dataset."""
    with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
      dataset = ogbg_molpcba.get_ogbg_molpcba(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=BATCH_SIZE,
          eval_batch_size=BATCH_SIZE,
          hps=hps,
      )
      return list(dataset.valid_epoch())

  def _make_hps(self, **overrides):
    """Creates a hyperparameter config for a test dataset."""

    hps_dict = ogbg_molpcba.DEFAULT_HPARAMS.to_dict()
    hps_dict.update({
        'train_size': NUM_GRAPHS,
        'valid_size': NUM_GRAPHS,
        'test_size': NUM_GRAPHS,
        'avg_nodes_per_graph': NUM_NODES,
        'avg_edges_per_graph': NUM_EDGES,
        'batch_nodes_multiplier': 1.0,
        'batch_edges_multiplier': 1.0,
        'input_node_shape': (9,),
        'input_edge_shape': (3,),
        'output_shape': (NUM_LABELS,),
    })
    hps_dict.update(overrides)
    return config_dict.ConfigDict(hps_dict)

  def test_same_batch_count(self):
    """Static & dynamic batching should produce the same number of batches."""

    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)
    self.assertLen(new_batches, len(old_batches))

  def test_same_graph_structure(self):
    """Static & dynamic batching should produce the same graph structure."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].n_node,
          new['inputs'].n_node,
          msg=f'n_node mismatch in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].n_edge,
          new['inputs'].n_edge,
          msg=f'n_edge mismatch in batch {i}',
      )

  def test_same_node_features(self):
    """Static & dynamic batching should produce the same node features."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllClose(
          old['inputs'].nodes,
          new['inputs'].nodes,
          atol=1e-6,
          msg=f'nodes mismatch in batch {i}',
      )

  def test_same_edge_features(self):
    """Static & dynamic batching should produce the same edge features."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllClose(
          old['inputs'].edges,
          new['inputs'].edges,
          atol=1e-6,
          msg=f'edges mismatch in batch {i}',
      )

  def test_same_senders_receivers(self):
    """Static & dynamic batching should produce the same senders & receivers."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].senders,
          new['inputs'].senders,
          msg=f'senders mismatch in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].receivers,
          new['inputs'].receivers,
          msg=f'receivers mismatch in batch {i}',
      )

  def test_same_labels_and_weights(self):
    """Static & dynamic batching should produce the same labels & weights."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllClose(
          old['targets'],
          new['targets'],
          atol=1e-6,
          msg=f'targets mismatch in batch {i}',
      )
      self.assertAllClose(
          old['weights'],
          new['weights'],
          atol=1e-6,
          msg=f'weights mismatch in batch {i}',
      )

  def test_same_padding_mask(self):
    """Static & dynamic batching should produce the same padding mask."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      old_mask = jraph.get_graph_padding_mask(old['inputs'])
      new_mask = jraph.get_graph_padding_mask(new['inputs'])
      self.assertAllEqual(
          old_mask, new_mask, msg=f'padding mask mismatch in batch {i}'
      )

  def test_with_bidirectional_edges(self):
    """Bidirectional edges should not affect static vs dynamic batching."""
    hps_dyn = self._make_hps(
        add_bidirectional_edges=True, eval_batching='dynamic'
    )
    hps_static = self._make_hps(
        add_bidirectional_edges=True, eval_batching='static'
    )
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    self.assertLen(new_batches, len(old_batches))
    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].n_node,
          new['inputs'].n_node,
          msg=f'n_node mismatch with bidir edges in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].n_edge,
          new['inputs'].n_edge,
          msg=f'n_edge mismatch with bidir edges in batch {i}',
      )
      self.assertAllClose(
          old['inputs'].nodes,
          new['inputs'].nodes,
          atol=1e-6,
          msg=f'nodes mismatch with bidir edges in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].senders,
          new['inputs'].senders,
          msg=f'senders mismatch with bidir edges in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].receivers,
          new['inputs'].receivers,
          msg=f'receivers mismatch with bidir edges in batch {i}',
      )

  def test_with_virtual_node(self):
    """Virtual nodes should not affect static vs dynamic batching."""
    hps_dyn = self._make_hps(add_virtual_node=True, eval_batching='dynamic')
    hps_static = self._make_hps(add_virtual_node=True, eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    self.assertLen(new_batches, len(old_batches))
    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].n_node,
          new['inputs'].n_node,
          msg=f'n_node mismatch with virtual node in batch {i}',
      )
      self.assertAllClose(
          old['inputs'].nodes,
          new['inputs'].nodes,
          atol=1e-6,
          msg=f'nodes mismatch with virtual node in batch {i}',
      )

  def test_with_self_loops(self):
    """Self loops should not affect static vs dynamic batching."""
    hps_dyn = self._make_hps(add_self_loops=True, eval_batching='dynamic')
    hps_static = self._make_hps(add_self_loops=True, eval_batching='static')
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    self.assertLen(new_batches, len(old_batches))
    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].n_edge,
          new['inputs'].n_edge,
          msg=f'n_edge mismatch with self loops in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].senders,
          new['inputs'].senders,
          msg=f'senders mismatch with self loops in batch {i}',
      )

  def test_all_augmentations(self):
    """Tests static vs dynamic batching with all augmentations.

    Verifies that static batching produces the same graph structure as dynamic
    batching when all augmentations are enabled.
    """
    hps_dyn = self._make_hps(
        add_bidirectional_edges=True,
        add_virtual_node=True,
        add_self_loops=True,
        eval_batching='dynamic',
    )
    hps_static = self._make_hps(
        add_bidirectional_edges=True,
        add_virtual_node=True,
        add_self_loops=True,
        eval_batching='static',
    )
    old_batches = self._get_batches(hps_dyn)
    new_batches = self._get_batches(hps_static)

    self.assertLen(new_batches, len(old_batches))
    for i, (old, new) in enumerate(zip(old_batches, new_batches)):
      self.assertAllEqual(
          old['inputs'].n_node,
          new['inputs'].n_node,
          msg=f'n_node mismatch with all augmentations in batch {i}',
      )
      self.assertAllEqual(
          old['inputs'].n_edge,
          new['inputs'].n_edge,
          msg=f'n_edge mismatch with all augmentations in batch {i}',
      )
      self.assertAllClose(
          old['inputs'].nodes,
          new['inputs'].nodes,
          atol=1e-6,
          msg=f'nodes mismatch with all augmentations in batch {i}',
      )
      self.assertAllClose(
          old['targets'],
          new['targets'],
          atol=1e-6,
          msg=f'targets mismatch with all augmentations in batch {i}',
      )
      self.assertAllClose(
          old['weights'],
          new['weights'],
          atol=1e-6,
          msg=f'weights mismatch with all augmentations in batch {i}',
      )

  def test_numpy_padding_mask_and_weights_parity_with_jraph(self):
    """Tests single-shard unskipped parity of targets and weights with jraph."""
    examples = _make_dataset()[:BATCH_SIZE]
    batch = ogbg_molpcba._build_full_batch(  # pylint: disable=protected-access
        examples=examples,
        graphs_per_shard=BATCH_SIZE,
        num_shards=1,
        max_nodes_per_shard=BATCH_SIZE * NUM_NODES + 1,
        max_edges_per_shard=BATCH_SIZE * NUM_EDGES,
        add_bidirectional_edges=False,
        add_virtual_node=False,
        add_self_loops=False,
        node_feat_dim=9,
        edge_feat_dim=3,
        num_labels=NUM_LABELS,
    )
    graph = batch['inputs']
    jraph_padding_mask = np.asarray(jraph.get_graph_padding_mask(graph))

    raw_labels = np.zeros((BATCH_SIZE + 1, NUM_LABELS), dtype=np.float32)
    for idx in range(BATCH_SIZE):
      raw_labels[idx] = examples[idx]['labels']
    nan_mask = np.isnan(raw_labels)
    ref_targets = np.where(nan_mask, 0.0, raw_labels).astype(np.float32)
    ref_weights = ((1.0 - nan_mask) * jraph_padding_mask[:, None]).astype(
        np.float32
    )

    self.assertEqual(batch['targets'].dtype, np.float32)
    self.assertEqual(batch['weights'].dtype, np.float32)
    self.assertAllEqual(batch['targets'], ref_targets)
    self.assertAllEqual(batch['weights'], ref_weights)

  def test_multishard_parity_with_dynamic_batching(self):
    """Static and dynamic batching match on targets and weights per shard."""
    num_shards = 4
    batch_size = 8
    graphs_per_shard = batch_size // num_shards

    dyn_batches = list(
        ogbg_molpcba._get_dynamic_batch_iterator(  # pylint: disable=protected-access
            iter(_make_dataset()),
            batch_size=batch_size,
            nodes_per_graph=NUM_NODES,
            edges_per_graph=NUM_EDGES,
            add_bidirectional_edges=False,
            add_self_loops=False,
            add_virtual_node=False,
            num_shards=num_shards,
        )
    )
    static_batches = list(
        ogbg_molpcba._get_static_batch_iterator(  # pylint: disable=protected-access
            iter(_make_dataset()),
            batch_size=batch_size,
            nodes_per_graph=NUM_NODES,
            edges_per_graph=NUM_EDGES,
            add_bidirectional_edges=False,
            add_self_loops=False,
            add_virtual_node=False,
            node_feat_dim=9,
            edge_feat_dim=3,
            num_labels=NUM_LABELS,
            num_shards=num_shards,
        )
    )

    self.assertLen(static_batches, len(dyn_batches))
    for i, (dyn_b, stat_b) in enumerate(zip(dyn_batches, static_batches)):
      self.assertAllEqual(
          dyn_b['inputs'].n_node,
          stat_b['inputs'].n_node,
          msg=f'n_node mismatch in multishard batch {i}',
      )
      self.assertAllClose(
          dyn_b['targets'],
          stat_b['targets'],
          atol=1e-6,
          msg=f'targets mismatch in multishard batch {i}',
      )
      self.assertAllClose(
          dyn_b['weights'],
          stat_b['weights'],
          atol=1e-6,
          msg=f'weights mismatch in multishard batch {i}',
      )
      for shard_idx in range(num_shards):
        pad_idx = shard_idx * (graphs_per_shard + 1) + graphs_per_shard
        self.assertAllEqual(
            stat_b['weights'][pad_idx],
            np.zeros(NUM_LABELS, dtype=np.float32),
        )

  def test_skipped_graphs_and_shard_padding_masked(self):
    """Skipped graphs and all shard padding graphs receive zero weight."""
    dataset = _make_dataset()
    for num_shards in [1, 3]:
      examples = dataset[: num_shards * BATCH_SIZE]
      for max_nodes_per_shard, max_edges_per_shard in [
          (2 * NUM_NODES + 1, BATCH_SIZE * NUM_EDGES),
          (NUM_NODES + 1, BATCH_SIZE * NUM_EDGES),
          (BATCH_SIZE * NUM_NODES + 1, 2 * NUM_EDGES),
      ]:
        batch = ogbg_molpcba._build_full_batch(  # pylint: disable=protected-access
            examples=examples,
            graphs_per_shard=BATCH_SIZE,
            num_shards=num_shards,
            max_nodes_per_shard=max_nodes_per_shard,
            max_edges_per_shard=max_edges_per_shard,
            add_bidirectional_edges=False,
            add_virtual_node=False,
            add_self_loops=False,
            node_feat_dim=9,
            edge_feat_dim=3,
            num_labels=NUM_LABELS,
        )
        n_node = batch['inputs'].n_node
        total_graphs = num_shards * (BATCH_SIZE + 1)
        ref_targets = np.zeros((total_graphs, NUM_LABELS), dtype=np.float32)
        ref_weights = np.zeros((total_graphs, NUM_LABELS), dtype=np.float32)

        for shard_idx in range(num_shards):
          graph_base = shard_idx * (BATCH_SIZE + 1)
          for local_idx in range(BATCH_SIZE):
            slot = graph_base + local_idx
            if n_node[slot] > 0:
              lbl = examples[shard_idx * BATCH_SIZE + local_idx]['labels']
              nan_mask = np.isnan(lbl)
              ref_targets[slot] = np.where(nan_mask, 0.0, lbl)
              ref_weights[slot] = np.where(nan_mask, 0.0, 1.0)

        self.assertAllEqual(batch['targets'], ref_targets)
        self.assertAllEqual(batch['weights'], ref_weights)


if __name__ == '__main__':
  tf.test.main()
