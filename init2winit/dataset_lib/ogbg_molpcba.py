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
import resource
from absl import logging
from init2winit.dataset_lib import data_utils
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
    )
)

METADATA = {
    'apply_one_hot_in_loss': False,
}


class _InMemoryDataset:
  """In-memory dataset that supports shuffling and repeating."""

  def __init__(self, data, should_shuffle=False, shuffle_seed=None):
    self.data = data
    self._should_shuffle = should_shuffle
    self._shuffle_seed = shuffle_seed

  def __iter__(self):
    if self._should_shuffle:
      rng = np.random.default_rng(int(self._shuffle_seed))
      while True:
        perm = rng.permutation(len(self.data))
        for i in perm:
          yield self.data[i]
    else:
      yield from self.data

  def __len__(self):
    return len(self.data)


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
  logging.info('Loading in memory dataset...')
  dataset = list(tfds.as_numpy(dataset))

  return _InMemoryDataset(dataset, should_shuffle, dataset_shuffle_seed)


def _to_jraph(example, add_bidirectional_edges, add_virtual_node,
              add_self_loops):
  """Converts an example graph to jraph.GraphsTuple."""
  if hasattr(example['edge_feat'], '_numpy'):
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


def _ensure_numpy(example):
  if hasattr(example.get('edge_feat'), '_numpy'):
    return data_utils.tf_to_numpy(example)
  return example


def _augmented_sizes(
    num_nodes,
    num_edges,
    add_bidirectional_edges,
    add_virtual_node,
    add_self_loops,
):
  """Fix the number of nodes and edges based on the hps."""

  if add_bidirectional_edges:
    num_edges *= 2
  if add_self_loops:
    num_edges += num_nodes
  if add_virtual_node:
    num_edges += num_nodes
    num_nodes += 1
  return num_nodes, num_edges


def _process_example(
    example: dict[str, np.ndarray],
    add_bidirectional_edges: bool,
    add_virtual_node: bool,
    add_self_loops: bool,
    nodes_buf: np.ndarray,
    edges_buf: np.ndarray,
    senders_buf: np.ndarray,
    receivers_buf: np.ndarray,
    node_offset: int,
    edge_offset: int,
) -> tuple[int, int]:
  """Processes a raw example and fills it into the pre-allocated buffers.

  Extracts node features, edge features, and connectivity from a single raw
  OGBG graph and writes them into the corresponding pre-allocated
  numpy buffers at the given offsets. Optionally adds bidirectional edges,
  self-loops, and a virtual node depending on the flags. The main goal is to
  batch multiple graphs into one large graph for the model.

  Args:
    example: A graph dict from ogbg_molpcba with the keys ``edge_feat``,
      ``node_feat``, ``edge_index``, and ``num_nodes``.
    add_bidirectional_edges: If True, duplicate every edge in the reverse
      direction so each original edge becomes bidirectional.
    add_virtual_node: If True, append an extra node connected to all original
      nodes (edges from every original node to the virtual node).
    add_self_loops: If True, add a self-loop edge for every node.
    nodes_buf: Pre-allocated numpy array for node features, written in-place.
    edges_buf: Pre-allocated numpy array for edge features, written in-place.
    senders_buf: Pre-allocated numpy array for sender indices, written in-place.
    receivers_buf: Pre-allocated numpy array for receiver indices, written
      in-place.
    node_offset: Starting index in the node buffers for this example.
    edge_offset: Starting index in the edge buffers for this example.

  Returns:
    A tuple (num_nodes, num_edges) giving the total number of nodes and edges
    written for this example (including any added by the augmentation flags).
  """

  edge_feat = example['edge_feat']
  node_feat = example['node_feat']
  edge_index = example['edge_index']
  num_nodes = int(np.squeeze(example['num_nodes']))
  num_edges = len(edge_index)

  nodes_buf[node_offset : node_offset + num_nodes] = node_feat

  eo = edge_offset
  senders_buf[eo : eo + num_edges] = edge_index[:, 0] + node_offset
  receivers_buf[eo : eo + num_edges] = edge_index[:, 1] + node_offset
  edges_buf[eo : eo + num_edges] = edge_feat
  eo += num_edges

  if add_bidirectional_edges:
    senders_buf[eo : eo + num_edges] = edge_index[:, 1] + node_offset
    receivers_buf[eo : eo + num_edges] = edge_index[:, 0] + node_offset
    edges_buf[eo : eo + num_edges] = edge_feat
    eo += num_edges

  if add_self_loops:
    self_range = np.arange(num_nodes, dtype=np.int32) + node_offset
    senders_buf[eo : eo + num_nodes] = self_range
    receivers_buf[eo : eo + num_nodes] = self_range
    eo += num_nodes

  if add_virtual_node:
    vn_senders = np.arange(num_nodes, dtype=np.int32) + node_offset
    senders_buf[eo : eo + num_nodes] = vn_senders
    receivers_buf[eo : eo + num_nodes] = node_offset + num_nodes
    eo += num_nodes
    num_nodes += 1

  return num_nodes, eo - edge_offset


def _build_full_batch(
    examples: list[dict[str, np.ndarray]],
    graphs_per_shard: int,
    num_shards: int,
    max_nodes_per_shard: int,
    max_edges_per_shard: int,
    add_bidirectional_edges: bool,
    add_virtual_node: bool,
    add_self_loops: bool,
    node_feat_dim: int,
    edge_feat_dim: int,
    num_labels: int,
) -> dict[str, jraph.GraphsTuple | np.ndarray]:
  """Builds a fully padded batch from raw examples in a single allocation.

  Pre-allocates node, edge, sender, and receiver buffers for the entire batch
  and fills them by calling _process_example on each graph sequentially.
  Graphs are packed into per-shard regions with a budget check; any graph that
  would exceed a shard's node or edge budget is skipped. Each shard graph is
  padded to ensure the batch size per shard is the same

  Args:
    examples: List of graph dicts from ogbg_molpcba.
    graphs_per_shard: Maximum number of real graphs packed into each shard.
    num_shards: Number of shards (typically number of devices).
    max_nodes_per_shard: Maximum number of nodes allocated per shard.
    max_edges_per_shard: Maximum number of edges allocated per shard.
    add_bidirectional_edges: If True, duplicate every edge in reverse.
    add_virtual_node: If True, add a virtual node connected to all real nodes.
    add_self_loops: If True, add a self-loop edge for every node.
    node_feat_dim: Dimensionality of node features.
    edge_feat_dim: Dimensionality of edge features.
    num_labels: Number of label columns per graph.

  Returns:
    A dict containing:
      inputs: a jraph.GraphsTuple for the entire batch
      targets: float32 label array with NaNs replaced by 0
      weights: a per-label mask that is 0 for NaN labels and padding graphs
  """
  total_graphs = num_shards * (graphs_per_shard + 1)
  total_nodes = num_shards * max_nodes_per_shard
  total_edges = num_shards * max_edges_per_shard

  nodes_buf = np.zeros((total_nodes, node_feat_dim), dtype=np.float32)
  edges_buf = np.zeros((total_edges, edge_feat_dim), dtype=np.float32)
  senders_buf = np.zeros(total_edges, dtype=np.int32)
  receivers_buf = np.zeros(total_edges, dtype=np.int32)
  n_node = np.zeros(total_graphs, dtype=np.int32)
  n_edge = np.zeros(total_graphs, dtype=np.int32)
  labels = np.zeros((total_graphs, num_labels), dtype=np.float32)

  global_node_offset = 0
  global_edge_offset = 0
  example_idx = 0

  for shard_idx in range(num_shards):
    graph_base = shard_idx * (graphs_per_shard + 1)
    shard_node_start = shard_idx * max_nodes_per_shard
    shard_edge_start = shard_idx * max_edges_per_shard
    node_budget = max_nodes_per_shard - 1
    edge_budget = max_edges_per_shard

    for local_idx in range(graphs_per_shard):
      ex = examples[example_idx]
      example_idx += 1

      nn, ne = _augmented_sizes(
          int(np.squeeze(ex['num_nodes'])),
          len(ex['edge_index']),
          add_bidirectional_edges,
          add_virtual_node,
          add_self_loops,
      )
      nodes_used = global_node_offset - shard_node_start
      edges_used = global_edge_offset - shard_edge_start
      if nodes_used + nn > node_budget or edges_used + ne > edge_budget:
        continue

      nn, ne = _process_example(
          ex,
          add_bidirectional_edges,
          add_virtual_node,
          add_self_loops,
          nodes_buf,
          edges_buf,
          senders_buf,
          receivers_buf,
          global_node_offset,
          global_edge_offset,
      )

      n_node[graph_base + local_idx] = nn
      n_edge[graph_base + local_idx] = ne
      labels[graph_base + local_idx] = ex['labels']

      global_node_offset += nn
      global_edge_offset += ne

    shard_node_end = shard_node_start + max_nodes_per_shard
    shard_edge_end = shard_edge_start + max_edges_per_shard
    pad_nodes = shard_node_end - global_node_offset
    pad_edges = shard_edge_end - global_edge_offset

    pad_graph_idx = graph_base + graphs_per_shard
    n_node[pad_graph_idx] = pad_nodes
    n_edge[pad_graph_idx] = pad_edges

    global_node_offset = shard_node_end
    global_edge_offset = shard_edge_end

  graph = jraph.GraphsTuple(
      n_node=n_node,
      n_edge=n_edge,
      nodes=nodes_buf,
      edges=edges_buf,
      senders=senders_buf,
      receivers=receivers_buf,
      globals={},
  )

  padding_mask = jraph.get_graph_padding_mask(graph)
  nan_mask = np.isnan(labels)
  replaced_labels = np.where(nan_mask, 0.0, labels)
  weights = (1.0 - nan_mask) * padding_mask[:, None]

  return {
      'inputs': graph,
      'targets': replaced_labels,
      'weights': weights,
  }


def _get_static_batch_iterator(
    dataset_iter,
    batch_size,
    nodes_per_graph,
    edges_per_graph,
    add_bidirectional_edges,
    add_self_loops,
    add_virtual_node,
    node_feat_dim,
    edge_feat_dim,
    num_labels,
    num_shards=None,
):
  """Construct a static batch iterator.

  Static batching just groups together N graphs based on the average number of
  nodes and edges per graph. This is in contrast to dynamic batching, which
  groups together graphs until the batch is full. Static batching is more
  efficient, but it may lead to some graphs being dropped if they don't fit
  into the batch.

  Args:
    dataset_iter: An iterator over the dataset.
    batch_size: The batch size.
    nodes_per_graph: The number of nodes per graph.
    edges_per_graph: The number of edges per graph.
    add_bidirectional_edges: Whether to add bidirectional edges.
    add_self_loops: Whether to add self loops.
    add_virtual_node: Whether to add a virtual node.
    node_feat_dim: The dimension of the node features.
    edge_feat_dim: The dimension of the edge features.
    num_labels: The number of labels.
    num_shards: The number of shards.

  Yields:
    A batch of graphs.
  """
  if not num_shards:
    num_shards = jax.local_device_count()

  graphs_per_shard = int(batch_size / num_shards)
  max_nodes_per_shard = int(nodes_per_graph * graphs_per_shard) + 1
  max_edges_per_shard = int(edges_per_graph * graphs_per_shard)

  total_graphs_needed = graphs_per_shard * num_shards

  while True:
    examples = [
        _ensure_numpy(ex)
        for ex in itertools.islice(dataset_iter, total_graphs_needed)
    ]
    if len(examples) < total_graphs_needed:
      break
    yield _build_full_batch(
        examples,
        graphs_per_shard,
        num_shards,
        max_nodes_per_shard,
        max_edges_per_shard,
        add_bidirectional_edges,
        add_virtual_node,
        add_self_loops,
        node_feat_dim,
        edge_feat_dim,
        num_labels,
    )


def _get_dynamic_batch_iterator(
    dataset_iter,
    batch_size,
    nodes_per_graph,
    edges_per_graph,
    add_bidirectional_edges,
    add_self_loops,
    add_virtual_node,
    num_shards=None,
):
  """Turns a TFDS per-example iterator into a batched iterator.

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
    Batch in the init2winit format.
  """
  if not num_shards:
    num_shards = jax.local_device_count()

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
          # Note: Use jraph.batch_np instead of jraph.batch (jnp) which leaks
          # memory due to the call to jnp.concatenate.
          # It is possible we may be leaking host memory with the np call.
          'inputs': jraph.batch_np(graphs_shards),
          'targets': np.vstack(labels_shards),
          'weights': np.vstack(weights_shards)
      }

      count = 0
      graphs_shards = []
      labels_shards = []
      weights_shards = []


def get_ogbg_molpcba(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Data generators for ogbg-molpcba."""

  process_count = jax.process_count()
  if batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide batch_size={}.'.format(
            process_count, batch_size
        )
    )
  if eval_batch_size % process_count != 0:
    raise ValueError(
        'process_count={} must divide eval_batch_size={}.'.format(
            process_count, eval_batch_size
        )
    )

  per_host_batch_size = int(batch_size / process_count)
  per_host_eval_batch_size = int(eval_batch_size / process_count)

  shuffle_buffer_size = 2**15
  shuffle_rng_train, shuffle_rng_eval_train = jax.random.split(shuffle_rng)
  def _log_mem(label):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    logging.info('[ogbg] %s — RSS: %.1f MB', label, rss_mb)

  _log_mem('Before loading any splits')

  train_ds = _load_dataset(
      'train',
      should_shuffle=True,
      shuffle_seed=shuffle_rng_train,
      shuffle_buffer_size=shuffle_buffer_size)
  _log_mem('After loading train split')
  eval_train_size = min(hps.valid_size, len(train_ds))
  # Use a random subset of the training data for eval_train.
  # This data is already loaded into memory, so this is cheap.
  # We just access it and wrap it in an _InMemoryDataset.
  eval_train_rng = np.random.default_rng(int(shuffle_rng_eval_train[0]))
  subset_indices = eval_train_rng.choice(
      len(train_ds), size=eval_train_size, replace=False
  )
  eval_train_data = [train_ds.data[i] for i in subset_indices]
  eval_train_ds = _InMemoryDataset(eval_train_data, should_shuffle=False)
  _log_mem('After creating eval_train subset')

  valid_ds = _load_dataset('validation')
  _log_mem('After loading validation split')

  test_ds = _load_dataset('test')
  _log_mem('After loading test split')

  max_nodes_multiplier = hps.batch_nodes_multiplier * hps.avg_nodes_per_graph
  max_edges_multiplier = hps.batch_edges_multiplier * hps.avg_edges_per_graph
  max_nodes_multiplier, max_edges_multiplier = _augmented_sizes(
      max_nodes_multiplier,
      max_edges_multiplier,
      hps.add_bidirectional_edges,
      hps.add_virtual_node,
      hps.add_self_loops,
  )

  common_kwargs = dict(
      nodes_per_graph=int(max_nodes_multiplier),
      edges_per_graph=int(max_edges_multiplier),
      add_bidirectional_edges=hps.add_bidirectional_edges,
      add_virtual_node=hps.add_virtual_node,
      add_self_loops=hps.add_self_loops,
  )
  static_kwargs = dict(
      node_feat_dim=hps.input_node_shape[0],
      edge_feat_dim=hps.input_edge_shape[0],
      num_labels=hps.output_shape[0],
  )

  def _make_iterator(strategy):
    if strategy == 'static':
      return functools.partial(
          _get_static_batch_iterator, **common_kwargs, **static_kwargs
      )
    else:
      return functools.partial(_get_dynamic_batch_iterator, **common_kwargs)

  train_iterator_from_ds = _make_iterator('static')
  eval_iterator_from_ds = _make_iterator('dynamic')

  def train_iterator_fn():
    return train_iterator_from_ds(
        dataset_iter=iter(train_ds), batch_size=per_host_batch_size
    )

  def _make_eval_epoch(ds):

    def epoch(num_batches=None):
      return itertools.islice(
          eval_iterator_from_ds(
              dataset_iter=iter(ds), batch_size=per_host_eval_batch_size
          ),
          num_batches,
      )

    return data_utils.CachedIteratorFactory(epoch(), split_name='eval')

  eval_train_epoch = _make_eval_epoch(eval_train_ds)
  valid_epoch = _make_eval_epoch(valid_ds)
  test_epoch = _make_eval_epoch(test_ds)
  return data_utils.Dataset(
      train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch
  )


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

  batching = 'static'
  if batching == 'static':
    batch_iterator = _get_static_batch_iterator(
        dataset_iterator(),
        batch_size=hps.batch_size,
        nodes_per_graph=hps.batch_nodes_multiplier * hps.avg_nodes_per_graph,
        edges_per_graph=hps.batch_edges_multiplier * hps.avg_edges_per_graph,
        add_bidirectional_edges=hps.add_bidirectional_edges,
        add_virtual_node=hps.add_virtual_node,
        add_self_loops=hps.add_self_loops,
        node_feat_dim=9,
        edge_feat_dim=3,
        num_labels=128,
    )
  else:
    batch_iterator = _get_dynamic_batch_iterator(
        dataset_iterator(),
        batch_size=hps.batch_size,
        nodes_per_graph=hps.batch_nodes_multiplier * hps.avg_nodes_per_graph,
        edges_per_graph=hps.batch_edges_multiplier * hps.avg_edges_per_graph,
        add_bidirectional_edges=hps.add_bidirectional_edges,
        add_virtual_node=hps.add_virtual_node,
        add_self_loops=hps.add_self_loops,
    )

  return next(batch_iterator)
