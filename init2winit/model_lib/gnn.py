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

"""Graph neural network model.

Implements a Graph Isomorphism Network (GIN) (https://arxiv.org/abs/1810.00826)
with edge features, also known as GINE (https://arxiv.org/abs/1905.12265).
The model replicates the GIN entry for the ogbg-molcpba benchmark
(https://ogb.stanford.edu/docs/leader_graphprop/).

"""

from flax import nn
from init2winit.model_lib import base_model
import jax.numpy as jnp
import jraph

from ml_collections.config_dict import config_dict

# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        rng_seed=-1,
        model_dtype='float32',
        latent_dim=128,
        optimizer='momentum',
        hidden_dims=(128, 128),
        lr_hparams={
            'base_lr': 0.0001,
            'schedule': 'constant'
        },
        opt_hparams={
            'momentum': 0.9,
        },
        activation_function='relu',
        l2_decay_factor=.0005,
        l2_decay_rank_threshold=2,
    ))


def _make_embed(latent_dim):

  def make_fn(inputs):
    return nn.Dense(inputs, features=latent_dim)

  return make_fn


def _make_mlp(hidden_dims):
  """Creates a MLP with specified dimensions."""
  @jraph.concatenated_args
  def make_fn(inputs):
    x = inputs
    # TODO(mbadura): Add batch normalization
    for dim in hidden_dims[:-1]:
      x = nn.Dense(x, features=dim)
      x = nn.relu(x)
    x = nn.Dense(x, features=hidden_dims[-1])
    return x

  return make_fn


class GNN(nn.Module):
  """Defines a graph network.

  The model assumes the input data is a jraph.GraphsTuple without global
  variables. The final prediction will be encoded in the globals.
  """

  def apply(self, graph, num_outputs, latent_dim, hidden_dims, train=True):
    graph = graph._replace(
        globals=jnp.zeros([graph.n_node.shape[0], num_outputs]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=_make_embed(latent_dim),
        embed_edge_fn=_make_embed(latent_dim))

    net = jraph.GraphNetwork(
        update_edge_fn=_make_mlp(hidden_dims),
        update_node_fn=_make_mlp(hidden_dims),
        update_global_fn=_make_mlp(hidden_dims + (num_outputs,)))

    # TODO(mbadura): Add multiple message passing steps
    embedded_graph = embedder(graph)
    result = net(embedded_graph)

    return result.globals


class GNNModel(base_model.BaseModel):
  """Defines the model for the graph network."""

  def get_fake_batch(self, hps):
    assert 'input_node_shape' in hps and 'input_edge_shape' in hps
    return jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1,) + hps.input_node_shape),
        edges=jnp.ones((1,) + hps.input_edge_shape),
        globals=jnp.zeros((1,) + hps.output_shape),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

  def build_flax_module(self):
    return GNN.partial(
        num_outputs=self.hps['output_shape'][-1],
        latent_dim=self.hps['latent_dim'],
        hidden_dims=self.hps['hidden_dims'])
