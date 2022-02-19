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

"""Input pipeline for the ogbg_molpcba graph dataset from TFDS.

See https://www.tensorflow.org/datasets/catalog/ogbg_molpcba and
https://ogb.stanford.edu/docs/graphprop/ for more details.
"""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib.data_utils import Dataset
import jax
import jraph
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow_datasets as tfds

AVG_NODES_PER_GRAPH = 26
AVG_EDGES_PER_GRAPH = 56

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        output_shape=(128,),
        input_edge_shape=(3,),
        input_node_shape=(9,),
        train_size=350343,
        valid_size=43793,
        test_size=43793,
        # Max edges/nodes per batch will batch_size times the multiplier.
        # We set them to the average size of the graph in the dataset,
        # so that each batch contains batch_size graphs on average.
        max_edges_multiplier=AVG_EDGES_PER_GRAPH,
        max_nodes_multiplier=AVG_NODES_PER_GRAPH,
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


def _to_jraph(example):
  """Converts an example graph to jraph.GraphsTuple."""
  example = data_utils.tf_to_numpy(example)
  edge_feat = example['edge_feat']
  node_feat = example['node_feat']
  edge_index = example['edge_index']
  labels = example['labels']
  num_nodes = example['num_nodes']

  senders = edge_index[:, 0]
  receivers = edge_index[:, 1]

  return jraph.GraphsTuple(
      n_node=num_nodes,
      n_edge=np.array([len(edge_index) * 2]),
      nodes=node_feat,
      edges=np.concatenate([edge_feat, edge_feat]),
      # Make the edges bidirectional
      senders=np.concatenate([senders, receivers]),
      receivers=np.concatenate([receivers, senders]),
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

  jraph_iter = map(_to_jraph, dataset_iter)
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

  def train_iterator_fn():
    return _get_batch_iterator(
        iter(train_ds), batch_size, hps.max_nodes_multiplier,
        hps.max_edges_multiplier)

  def eval_train_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(
            iter(eval_train_ds), batch_size, hps.max_nodes_multiplier,
            hps.max_edges_multiplier), num_batches)

  def valid_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(
            iter(valid_ds), eval_batch_size, hps.max_nodes_multiplier,
            hps.max_edges_multiplier), num_batches)

  def test_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(
            iter(test_ds), eval_batch_size, hps.max_nodes_multiplier,
            hps.max_edges_multiplier), num_batches)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
