# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Graph neural network model.

Implements a basic GraphNetwork (Battaglia et al., 2018) with explicit node,
edge, and global state updates using a fully connected neural network as
described in https://arxiv.org/abs/1806.01261. The input node and edge features
are encoded using a linear layer.

On the OGBG leaderboard:
https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molpcba, the model is the
closest to the GIN+Virtual Node (Xu et al., 2018) model, and it reaches
equivalent performance on the ogbg-molpcba dataset.
"""
from typing import Tuple

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax.numpy as jnp
import jraph
from ml_collections.config_dict import config_dict

# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        rng_seed=-1,
        model_dtype='float32',
        latent_dim=256,
        optimizer='adam',
        hidden_dims=(256,),
        batch_size=256,
        lr_hparams={
            'base_lr': 0.01,
            'schedule': 'constant'
        },
        opt_hparams={
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0,
        },
        activation_function='relu',
        l2_decay_factor=.0005,
        l2_decay_rank_threshold=2,
        num_message_passing_steps=5,
        normalizer='layer_norm',
        dropout_rate=0.1,
        total_accumulated_batch_size=None,
        grad_clip=None,
        label_smoothing=0.0,
        use_shallue_label_smoothing=False,
    ))


def _make_embed(latent_dim):

  def make_fn(inputs):
    return nn.Dense(features=latent_dim)(inputs)

  return make_fn


def _make_mlp(hidden_dims, maybe_normalize_fn, dropout, activation=nn.relu):
  """Creates a MLP with specified dimensions."""

  @jraph.concatenated_args
  def make_fn(inputs):
    x = inputs
    for dim in hidden_dims:
      x = nn.Dense(features=dim)(x)
      x = maybe_normalize_fn()(x)
      x = activation(x)
      x = dropout(x)
    return x

  return make_fn


class GNN(nn.Module):
  """Defines a graph network.

  The model assumes the input data is a jraph.GraphsTuple without global
  variables. The final prediction will be encoded in the globals.
  """
  num_outputs: int
  latent_dim: int
  hidden_dims: Tuple[int]
  normalizer: str
  dropout_rate: float
  num_message_passing_steps: int
  activation_function: str

  @nn.compact
  def __call__(self, graph, train):
    maybe_normalize_fn = model_utils.get_normalizer(self.normalizer, train)
    dropout = nn.Dropout(rate=self.dropout_rate, deterministic=not train)
    activation = model_utils.ACTIVATIONS[self.activation_function]

    graph = graph._replace(
        globals=jnp.zeros([graph.n_node.shape[0], self.num_outputs]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=_make_embed(self.latent_dim),
        embed_edge_fn=_make_embed(self.latent_dim))
    graph = embedder(graph)

    for _ in range(self.num_message_passing_steps):
      net = jraph.GraphNetwork(
          update_edge_fn=_make_mlp(
              self.hidden_dims,
              maybe_normalize_fn=maybe_normalize_fn,
              dropout=dropout,
              activation=activation),
          update_node_fn=_make_mlp(
              self.hidden_dims,
              maybe_normalize_fn=maybe_normalize_fn,
              dropout=dropout,
              activation=activation),
          update_global_fn=_make_mlp(
              self.hidden_dims,
              maybe_normalize_fn=maybe_normalize_fn,
              dropout=dropout,
              activation=activation))

      graph = net(graph)

    # Map globals to represent the final result
    decoder = jraph.GraphMapFeatures(embed_global_fn=nn.Dense(self.num_outputs))
    graph = decoder(graph)

    return graph.globals


class GNNModel(base_model.BaseModel):
  """Defines the model for the graph network."""

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initializing the model."""
    assert 'input_node_shape' in hps and 'input_edge_shape' in hps
    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1,) + hps.input_node_shape),
        edges=jnp.ones((1,) + hps.input_edge_shape),
        globals=jnp.zeros((1,) + hps.output_shape),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))
    # We need to wrap the GraphsTuple in a list so that it can be passed as
    # *inputs to the model init function.
    return [graph]

  def build_flax_module(self):
    return GNN(
        num_outputs=self.hps['output_shape'][-1],
        latent_dim=self.hps['latent_dim'],
        hidden_dims=self.hps['hidden_dims'],
        normalizer=self.hps['normalizer'],
        dropout_rate=self.hps['dropout_rate'],
        num_message_passing_steps=self.hps['num_message_passing_steps'],
        activation_function=self.hps['activation_function'],)
