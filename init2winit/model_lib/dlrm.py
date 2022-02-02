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

"""A JAX implementation of DLRM."""

from typing import List

import flax.linen as nn
from jax import nn as jnn
import jax.numpy as jnp


def dot_interact(concat_features, keep_diags=True):
  """Performs feature interaction operation between dense or sparse features.

  Input tensors represent dense or sparse features.
  Pre-condition: The tensors have been stacked along dimension 1.

  Args:
    concat_features: Array of features with shape [B, n_features, feature_dim].
    keep_diags: Whether to keep the diagonal terms of x @ x.T.

  Returns:
    activations: Array representing interacted features.
  """
  batch_size = concat_features.shape[0]

  # Interact features, select upper or lower-triangular portion, and re-shape.
  xactions = jnp.matmul(
      concat_features, jnp.transpose(concat_features, [0, 2, 1]))
  ones = jnp.ones_like(xactions)
  upper_tri_mask = jnp.triu(ones)
  feature_dim = xactions.shape[-1]

  if keep_diags:
    activations = xactions[upper_tri_mask == 1.0]
    out_dim = feature_dim * (feature_dim + 1) // 2
  else:
    lower_tri_mask = ones - upper_tri_mask
    activations = xactions[lower_tri_mask == 1.0]
    out_dim = feature_dim * (feature_dim - 1) // 2

  activations = jnp.reshape(activations, [batch_size, out_dim])
  return activations


class DLRM(nn.Module):
  """DLRM Model.

  Parameters:
    vocab_sizes: list of vocab sizes of embedding tables.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
    keep_diags: whether to keep the diagonal terms in x @ x.T.
  """

  vocab_sizes: List[int]
  mlp_bottom_dims: List[int]
  mlp_top_dims: List[int]
  embed_dim: int = 128
  keep_diags: bool = True

  @nn.compact
  def __call__(self, x, train):
    del train
    bot_mlp_input = x['dense-features']
    cat_features = x['cat-features']

    # bottom mlp
    mlp_bottom_dims = self.mlp_bottom_dims
    for dense_dim in mlp_bottom_dims:
      bot_mlp_input = nn.Dense(
          dense_dim,
          kernel_init=jnn.initializers.glorot_uniform(),
          bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim)),
      )(bot_mlp_input)
      bot_mlp_input = nn.relu(bot_mlp_input)
    bot_mlp_output = bot_mlp_input
    batch_size = bot_mlp_output.shape[0]
    feature_stack = jnp.reshape(bot_mlp_output,
                                [batch_size, -1, self.embed_dim])

    # embedding table look-up
    vocab_sizes = jnp.asarray(self.vocab_sizes, dtype=jnp.int32)
    idx_offsets = jnp.asarray(
        [0] + list(jnp.cumsum(vocab_sizes[:-1])), dtype=jnp.int32)
    idx_offsets = jnp.tile(
        jnp.reshape(idx_offsets, [1, -1]), [batch_size, 1])
    idx_lookup = cat_features + idx_offsets
    embedding_table = jnp.concatenate(
        [
            self.param(  # pylint: disable=g-complex-comprehension
                'embedding_table_%d' % i,
                jnn.initializers.variance_scaling(
                    scale=1.0, mode='fan_in', distribution='uniform'),
                [size, self.embed_dim])
            for i, size in enumerate(vocab_sizes)
        ],
        0)
    idx_lookup = jnp.reshape(idx_lookup, [-1])
    embed_features = embedding_table[idx_lookup]
    embed_features = jnp.reshape(
        embed_features, [batch_size, -1, self.embed_dim])
    feature_stack = jnp.concatenate([feature_stack, embed_features], axis=1)
    dot_interact_output = dot_interact(
        concat_features=feature_stack, keep_diags=self.keep_diags)
    top_mlp_input = jnp.concatenate([bot_mlp_output, dot_interact_output],
                                    axis=-1)
    mlp_input_dim = top_mlp_input.shape[1]
    mlp_top_dims = self.mlp_top_dims
    num_layers_top = len(mlp_top_dims)
    sigmoid_layer_top = num_layers_top - 1
    for layer_idx, fan_out in enumerate(mlp_top_dims):
      fan_in = mlp_input_dim if layer_idx == 0 else mlp_top_dims[layer_idx - 1]
      top_mlp_input = nn.Dense(
          fan_out,
          kernel_init=jnn.initializers.normal(
              stddev=jnp.sqrt(2.0 / (fan_in + fan_out))),
          bias_init=jnn.initializers.normal(
              stddev=jnp.sqrt(1.0 / mlp_top_dims[layer_idx])))(
                  top_mlp_input)
      act_fn = nn.sigmoid if layer_idx == sigmoid_layer_top else nn.relu
      top_mlp_input = act_fn(top_mlp_input)
    predictions = top_mlp_input
    return predictions
