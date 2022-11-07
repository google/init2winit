# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""Input pipeline for the ogbg_molpcba graph dataset from TFDS.

See https://www.tensorflow.org/datasets/catalog/ogbg_molpcba and
https://ogb.stanford.edu/docs/graphprop/ for more details.

NOTE(dsuo): this dataset dynamically generates batches from example graphs that
represent different molecules. The core batching function,
`jraph.dynamically_batch`, takes a graph dataset iterator and batches examples
together until whichever of the specified maximum number of nodes, edges, or
graphs is reached first.

- max_n_nodes is computed as batch_size * avg_nodes_per_graph *
  batch_nodes_multiplier + 1.
- max_n_edges is computed as batch_size * avg_edges_per_graph *
  batch_edges_multiplier.
- max_n_graphs is computed as batch_size + 1.

These values may further be modified if any of `add_bidirectional_edges`,
`add_virtual_node`, or `add_self_loops` are true as they influence one or both
of `avg_nodes_per_graph` and `avg_edges_per_graph`.
"""

import functools
import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib.data_utils import Dataset
import jax
import jraph
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        output_shape=(128,),
        input_edge_shape=(3,),
        input_node_shape=(9,),
        train_size=350343,
        valid_size=43793,
        test_size=43793,
        # NOTE(dsuo): Max edges/nodes per batch will batch_size times the
        # multiplier times the average per graph. These max values are further
        # modified if any of `add_bidirectional_edges`, `add_virtual_node`, or
        # `add_self_loops` are true.
        avg_nodes_per_graph=26,
        avg_edges_per_graph=28,
        batch_nodes_multiplier=1.0,
        # NOTE(dsuo): We set this multiplier to 2.0 because it gives better
        # performance, empirically.
        batch_edges_multiplier=2.0,
        add_bidirectional_edges=False,
        add_virtual_node=False,
        add_self_loops=False,
    ))

METADATA = {
    'apply_one_hot_in_loss': False,
}


def _load_dataset(split,
                  should_shuffle=False,
                  shuffle_seed=None,
                  shuffle_buffer_size=None):
  """Loads a dataset split from TFDS."""
  if should_shuffle:
    assert shuffle_seed is not None and shuffle_buffer_size is not None
    file_shuffle_seed, dataset_shuffle_seed = jax.random.split(shuffle_seed)
    file_shuffle_seed = file_shuffle_seed[0]
    dataset_shuffle_seed = dataset_shuffle_seed[0]
  else:
    file_shuffle_seed = None
    dataset_shuffle_seed = None

  read_config = tfds.ReadConfig(
      add_tfds_id=True, shuffle_seed=file_shuffle_seed)
  dataset = tfds.load(
      'ogbg_molpcba',
      split=split,
      shuffle_files=should_shuffle,
      read_config=read_config)

  if should_shuffle:
    dataset = dataset.shuffle(
        seed=dataset_shuffle_seed, buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()

  return dataset


def _to_jraph(example, add_bidirectional_edges, add_virtual_node,
              add_self_loops):
  """Converts an example graph to jraph.GraphsTuple."""
  example = data_utils.tf_to_numpy(example)
  edge_feat = example['edge_feat']
  node_feat = example['node_feat']
  edge_index = example['edge_index']
  labels = example['labels']
  num_nodes = np.squeeze(example['num_nodes'])
  num_edges = len(edge_index)

  senders = edge_index[:, 0]
  receivers = edge_index[:, 1]

  new_senders, new_receivers = senders[:], receivers[:]

  if add_bidirectional_edges:
    new_senders = np.concatenate([senders, receivers])
    new_receivers = np.concatenate([receivers, senders])
    edge_feat = np.concatenate([edge_feat, edge_feat])
    num_edges *= 2

  if add_self_loops:
    new_senders = np.concatenate([new_senders, np.arange(num_nodes)])
    new_receivers = np.concatenate([new_receivers, np.arange(num_nodes)])
    edge_feat = np.concatenate(
        [edge_feat, np.zeros((num_nodes, edge_feat.shape[-1]))])
    num_edges += num_nodes

  if add_virtual_node:
    node_feat = np.concatenate([node_feat, np.zeros_like(node_feat[0, None])])
    new_senders = np.concatenate([new_senders, np.arange(num_nodes)])
    new_receivers = np.concatenate(
        [new_receivers, np.full((num_nodes,), num_nodes)])
    edge_feat = np.concatenate(
        [edge_feat, np.zeros((num_nodes, edge_feat.shape[-1]))])
    num_edges += num_nodes
    num_nodes += 1

  return jraph.GraphsTuple(
      n_node=np.array([num_nodes]),
      n_edge=np.array([num_edges]),
      nodes=node_feat,
      edges=edge_feat,
      senders=new_senders,
      receivers=new_receivers,
      # Keep the labels with the graph for batching. They will be removed
      # in the processed batch.
      globals=np.expand_dims(labels, axis=0))


def _get_weights_by_nan_and_padding(labels, padding_mask):
  """Handles NaNs and padding in labels.

  Sets all the weights from examples coming from padding to 0. Changes all NaNs
  in labels to 0s and sets the corresponding per-label weight to 0.

  Args:
    labels: Labels including labels from padded examples
    padding_mask: Binary array of which examples are padding

  Returns:
    tuple of (processed labels, corresponding weights)
  """
  nan_mask = np.isnan(labels)
  replaced_labels = np.copy(labels)
  np.place(replaced_labels, nan_mask, 0)

  weights = 1.0 - nan_mask
  # Weights for all labels of a padded element will be 0
  weights = weights * padding_mask[:, None]
  return replaced_labels, weights


def _get_batch_iterator(dataset_iter,
                        batch_size,
                        nodes_per_graph,
                        edges_per_graph,
                        add_bidirectional_edges,
                        add_self_loops,
                        add_virtual_node,
                        num_shards=None):
  """Turns a TFDS per-example iterator into a batched iterator in the init2winit format.

  Constructs the batch from num_shards smaller batches, so that we can easily
  shard the batch to multiple devices during training. We use
  dynamic batching, so we specify some max number of graphs/nodes/edges, add
  as many graphs as we can, and then pad to the max values.

  Args:
    dataset_iter: The TFDS dataset iterator.
    batch_size: How many average-sized graphs go into the batch.
    nodes_per_graph: How many nodes per graph there are on average. Max number
      of nodes in the batch will be nodes_per_graph * batch_size.
    edges_per_graph: How many edges per graph there are on average. Max number
      of edges in the batch will be edges_per_graph * batch_size.
    add_bidirectional_edges: If True, add edges with reversed sender and
      receiver.
    add_self_loops: If True, add a self-loop for each node.
    add_virtual_node: If True, add a new node connected to all nodes.
    num_shards: How many devices we should be able to shard the batch into.

  Yields:
    Batch in the init2winit format. Each field is a list of num_shards separate
    smaller batches.

  """
  if not num_shards:
    num_shards = jax.device_count()

  # We will construct num_shards smaller batches and then put them together.
  batch_size /= num_shards

  max_n_nodes = nodes_per_graph * batch_size
  max_n_edges = edges_per_graph * batch_size
  max_n_graphs = batch_size

  to_jraph_partial = functools.partial(
      _to_jraph,
      add_bidirectional_edges=add_bidirectional_edges,
      add_virtual_node=add_virtual_node,
      add_self_loops=add_self_loops)
  jraph_iter = map(to_jraph_partial, dataset_iter)
  batched_iter = jraph.dynamically_batch(jraph_iter, max_n_nodes + 1,
                                         max_n_edges, max_n_graphs + 1)

  count = 0
  graphs_shards = []
  labels_shards = []
  weights_shards = []

  for batched_graph in batched_iter:
    count += 1

    # Separate the labels from the graph
    labels = batched_graph.globals
    graph = batched_graph._replace(globals={})

    replaced_labels, weights = _get_weights_by_nan_and_padding(
        labels, jraph.get_graph_padding_mask(graph))

    graphs_shards.append(graph)
    labels_shards.append(replaced_labels)
    weights_shards.append(weights)

    if count == num_shards:
      yield {
          'inputs': graphs_shards,
          'targets': labels_shards,
          'weights': weights_shards
      }

      count = 0
      graphs_shards = []
      labels_shards = []
      weights_shards = []


def get_ogbg_molpcba(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Data generators for ogbg-molpcba."""
  shuffle_buffer_size = 2**15
  shuffle_rng_train, shuffle_rng_eval_train = jax.random.split(shuffle_rng)
  train_ds = _load_dataset(
      'train',
      should_shuffle=True,
      shuffle_seed=shuffle_rng_train,
      shuffle_buffer_size=shuffle_buffer_size)
  eval_train_ds = _load_dataset(
      'train',
      should_shuffle=True,
      shuffle_seed=shuffle_rng_eval_train,
      shuffle_buffer_size=shuffle_buffer_size)
  valid_ds = _load_dataset('validation')
  test_ds = _load_dataset('test')

  max_nodes_multiplier = hps.batch_nodes_multiplier * hps.avg_nodes_per_graph
  max_edges_multiplier = hps.batch_edges_multiplier * hps.avg_edges_per_graph

  if hps.add_bidirectional_edges:
    max_edges_multiplier *= 2

  if hps.add_self_loops:
    max_edges_multiplier += max_nodes_multiplier

  if hps.add_virtual_node:
    max_edges_multiplier += max_nodes_multiplier
    max_nodes_multiplier += 1

  iterator_from_ds = functools.partial(
      _get_batch_iterator,
      nodes_per_graph=int(max_nodes_multiplier),
      edges_per_graph=int(max_edges_multiplier),
      add_bidirectional_edges=hps.add_bidirectional_edges,
      add_virtual_node=hps.add_virtual_node,
      add_self_loops=hps.add_self_loops)

  def train_iterator_fn():
    return iterator_from_ds(dataset_iter=iter(train_ds), batch_size=batch_size)

  def eval_train_epoch(num_batches=None):
    return itertools.islice(
        iterator_from_ds(
            dataset_iter=iter(eval_train_ds), batch_size=eval_batch_size),
        num_batches)

  def valid_epoch(num_batches=None):
    return itertools.islice(
        iterator_from_ds(
            dataset_iter=iter(valid_ds), batch_size=eval_batch_size),
        num_batches)

  def test_epoch(num_batches=None):
    return itertools.islice(
        iterator_from_ds(
            dataset_iter=iter(test_ds), batch_size=eval_batch_size),
        num_batches)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)


def get_fake_batch(hps):
  """Get fake ogbg_molpcba batch."""
  # NOTE(dsuo): the number of edges / nodes are approximately normally
  # distributed with the following mean and standard deviation.
  num_nodes_mean = 25.6
  num_nodes_std = 5.9
  num_edges_mean = 27.6
  num_edges_std = 6.6

  def dataset_iterator():
    """Fake raw data iterator."""
    # NOTE(dsuo): fix the random seed locally.
    rng = np.random.default_rng(0)

    while True:
      num_nodes = int(rng.normal(loc=num_nodes_mean, scale=num_nodes_std))

      # NOTE(dsuo): we want at least as many edges as we have nodes.
      num_edges = max(num_nodes,
                      int(rng.normal(loc=num_edges_mean, scale=num_edges_std)))

      # NOTE(dsuo): create an edge between pair of consecutive nodes to have
      # a well-formed molecule.
      edge_index = np.zeros((num_edges, 2), dtype=np.int32)
      edge_index[:num_nodes, 0] = np.arange(num_nodes)
      edge_index[:num_nodes, 1] = np.roll(np.arange(num_nodes), 1)

      # NOTE(dsuo): create random edges for any remaining.
      if num_edges > num_nodes:
        edge_index[num_nodes:num_edges, :] = rng.choice(
            num_nodes, (num_edges - num_nodes, 2))

      yield {
          'edge_feat': tf.ones((num_edges, 3), dtype=tf.float32),
          'edge_index': tf.convert_to_tensor(edge_index, dtype=tf.int32),
          'node_feat': tf.ones((num_nodes, 9), dtype=tf.float32),
          'labels': tf.zeros((128,), dtype=tf.int32),
          'num_edges': tf.constant([num_edges], dtype=tf.int32),
          'num_nodes': tf.constant([num_nodes], dtype=tf.int32),
      }

  batch_iterator = _get_batch_iterator(
      dataset_iterator(),
      batch_size=hps.batch_size,
      nodes_per_graph=hps.batch_nodes_multiplier * hps.avg_nodes_per_graph,
      edges_per_graph=hps.batch_edges_multiplier * hps.avg_edges_per_graph,
      add_bidirectional_edges=hps.add_bidirectional_edges,
      add_virtual_node=hps.add_virtual_node,
      add_self_loops=hps.add_self_loops)

  return next(batch_iterator)
