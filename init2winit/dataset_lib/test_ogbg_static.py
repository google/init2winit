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
import time
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


class BenchmarkStaticBatchingTest(tf.test.TestCase):
  """Tests that static batching is faster than dynamic batching.

  This should be true for any large batch size, because dynamic batching has a
  high overhead.
  """

  def _make_hps(self, **overrides):
    """Creates a hyperparameter config for a benchmark dataset."""
    hps_dict = ogbg_molpcba.DEFAULT_HPARAMS.to_dict()
    hps_dict.update({
        'train_size': BENCH_GRAPHS,
        'valid_size': BENCH_GRAPHS,
        'test_size': BENCH_GRAPHS,
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

  def _time_pipeline(self, hps, reps=3):
    """Times the pipeline for a given hyperparameter config.

    Takes the minimum over N=reps runs.
    Args:
      hps: The hyperparameters.
      reps: The number of repetitions.

    Returns:
      A tuple of (minimum time elapsed, number of batches).
    """
    times = []
    batches = []
    for _ in range(reps):
      with tfds.testing.mock_data(as_dataset_fn=_as_bench_dataset):
        dataset = ogbg_molpcba.get_ogbg_molpcba(
            shuffle_rng=jax.random.PRNGKey(0),
            batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            hps=hps,
        )
        start = time.perf_counter()
        batches = list(dataset.valid_epoch())
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times), len(batches)

  def test_static_is_faster(self):
    """Tests that static batching is faster than dynamic batching."""
    hps_dyn = self._make_hps(eval_batching='dynamic')
    hps_static = self._make_hps(eval_batching='static')
    old_time, old_count = self._time_pipeline(hps_dyn)
    new_time, new_count = self._time_pipeline(hps_static)

    self.assertEqual(old_count, new_count)

    print(f'\nDynamic batching: {old_time:.4f}s ({old_count} batches)')
    print(f'Static batching:  {new_time:.4f}s ({new_count} batches)')
    print(f'Speedup: {old_time / new_time:.2f}x')

    self.assertLess(
        new_time,
        old_time * 1.1,
        msg=(
            f'Static batching ({new_time:.4f}s) should not be slower than '
            f'dynamic batching ({old_time:.4f}s)'
        ),
    )


if __name__ == '__main__':
  tf.test.main()
