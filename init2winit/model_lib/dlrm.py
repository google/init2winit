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

from typing import List, Optional

import flax.linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax import nn as jnn
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        activation_function='relu',
        embedding_init_multiplier=None,
        rng_seed=-1,
        model_dtype='float32',
        vocab_size=32 * 128 * 1024,
        mlp_bottom_dims=[128, 128],
        mlp_top_dims=[256, 128, 1],
        output_shape=(1,),
        embed_dim=64,
        keep_diags=True,
        optimizer='adam',
        batch_size=128,
        num_dense_features=13,
        lr_hparams={
            'base_lr': 0.01,
            'schedule': 'constant'
        },
        opt_hparams={
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        },
        l2_decay_factor=1e-5,
        l2_decay_rank_threshold=2,
        total_accumulated_batch_size=None,
        grad_clip=None,
        dropout_rate=0.0,
        normalizer='none',
        # dropout will exist only if there are at least two top mlp layers
    ))

DEFAULT_RESNET_HPARAMS = DEFAULT_HPARAMS.copy_and_resolve_references()
DEFAULT_RESNET_HPARAMS.mlp_top_dims = [128, 128, 1]


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
  feature_dim = xactions.shape[-1]

  if keep_diags:
    indices = jnp.array(jnp.triu_indices(feature_dim))
  else:
    indices = jnp.array(jnp.tril_indices(feature_dim))
  num_elems = indices.shape[1]
  indices = jnp.tile(indices, [1, batch_size])
  indices0 = jnp.reshape(jnp.tile(jnp.reshape(
      jnp.arange(batch_size), [-1, 1]), [1, num_elems]), [1, -1])
  indices = tuple(jnp.concatenate((indices0, indices), 0))
  activations = xactions[indices]
  activations = jnp.reshape(activations, [batch_size, -1])
  return activations


class DLRM(nn.Module):
  """Define a DLRM model.

  Parameters:
    vocab_size: the size of a single unified embedding table.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    num_dense_features: number of dense features as the bottom mlp input.
    embed_dim: embedding dimension.
    keep_diags: whether to keep the diagonal terms in x @ x.T.
  """

  vocab_size: int
  mlp_bottom_dims: List[int]
  mlp_top_dims: List[int]
  num_dense_features: int
  embed_dim: int = 128
  keep_diags: bool = True
  dropout_rate: float = 0.0
  activation_function: str = 'relu'
  embedding_init_multiplier: Optional[float] = None
  normalizer: str = 'none'

  @nn.compact
  def __call__(self, x, train):
    bot_mlp_input, cat_features = jnp.split(x, [self.num_dense_features], 1)
    cat_features = jnp.asarray(cat_features, dtype=jnp.int32)

    activation_fn = model_utils.ACTIVATIONS[self.activation_function]
    normalizer_layer = model_utils.get_normalizer(self.normalizer, train)
    # bottom mlp
    mlp_bottom_dims = self.mlp_bottom_dims
    for dense_dim in mlp_bottom_dims:
      bot_mlp_input = nn.Dense(
          dense_dim,
          kernel_init=jnn.initializers.glorot_uniform(),
          bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim)),
      )(bot_mlp_input)
      bot_mlp_input = activation_fn(bot_mlp_input)
      bot_mlp_input = normalizer_layer()(bot_mlp_input)
    bot_mlp_output = bot_mlp_input
    batch_size = bot_mlp_output.shape[0]
    feature_stack = jnp.reshape(bot_mlp_output,
                                [batch_size, -1, self.embed_dim])

    base_init_fn = jnn.initializers.uniform(scale=1.0)
    if self.embedding_init_multiplier is None:
      embedding_init_multiplier = 1 / jnp.sqrt(self.vocab_size)
    else:
      embedding_init_multiplier = self.embedding_init_multiplier
    # Embedding table init and lookup for a single unified table.
    idx_lookup = jnp.reshape(cat_features, [-1]) % self.vocab_size
    def scaled_init(key, shape, dtype=jnp.float_):
      return base_init_fn(key, shape, dtype) * embedding_init_multiplier

    embedding_table = self.param(
        'embedding_table',
        scaled_init,
        [self.vocab_size, self.embed_dim])

    embed_features = embedding_table[idx_lookup]
    embed_features = jnp.reshape(
        embed_features, [batch_size, -1, self.embed_dim])
    embed_features = normalizer_layer()(embed_features)
    feature_stack = jnp.concatenate([feature_stack, embed_features], axis=1)
    dot_interact_output = dot_interact(
        concat_features=feature_stack, keep_diags=self.keep_diags)
    top_mlp_input = jnp.concatenate([bot_mlp_output, dot_interact_output],
                                    axis=-1)
    mlp_input_dim = top_mlp_input.shape[1]
    mlp_top_dims = self.mlp_top_dims
    num_layers_top = len(mlp_top_dims)
    for layer_idx, fan_out in enumerate(mlp_top_dims):
      fan_in = mlp_input_dim if layer_idx == 0 else mlp_top_dims[layer_idx - 1]
      top_mlp_input = nn.Dense(
          fan_out,
          kernel_init=jnn.initializers.normal(
              stddev=jnp.sqrt(2.0 / (fan_in + fan_out))),
          bias_init=jnn.initializers.normal(
              stddev=jnp.sqrt(1.0 / mlp_top_dims[layer_idx])))(
                  top_mlp_input)
      if layer_idx < (num_layers_top - 1):
        top_mlp_input = activation_fn(top_mlp_input)
        top_mlp_input = normalizer_layer()(top_mlp_input)
      if self.dropout_rate > 0.0 and layer_idx == num_layers_top - 2:
        top_mlp_input = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train)(
                top_mlp_input)
    logits = top_mlp_input
    return logits


class DLRMModel(base_model.BaseModel):
  """DLRMModel init2winit class."""

  def build_flax_module(self):
    """DLRM for ad click probability prediction."""
    return DLRM(
        activation_function=self.hps.activation_function,
        embedding_init_multiplier=self.hps.embedding_init_multiplier,
        vocab_size=self.hps.vocab_size,
        mlp_bottom_dims=self.hps.mlp_bottom_dims,
        mlp_top_dims=self.hps.mlp_top_dims,
        num_dense_features=self.hps.num_dense_features,
        embed_dim=self.hps.embed_dim,
        keep_diags=self.hps.keep_diags,
        dropout_rate=self.hps.dropout_rate,
        normalizer=self.hps.normalizer)

  def get_fake_inputs(self, hps):
    """Helper method solely for purpose of initalizing the model."""
    # NOTE(dsuo): hps.input_shape for `criteo_terabyte_input_pipeline` is
    # (39,)
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs


class DLRMResNet(nn.Module):
  """Define a DLRMResNet model.

  Parameters:
    vocab_size: the size of a single unified embedding table.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    num_dense_features: number of dense features as the bottom mlp input.
    embed_dim: embedding dimension.
    keep_diags: whether to keep the diagonal terms in x @ x.T.
  """

  vocab_size: int
  mlp_bottom_dims: List[int]
  mlp_top_dims: List[int]
  num_dense_features: int
  embed_dim: int = 128
  keep_diags: bool = True
  dropout_rate: float = 0.0
  activation_function: str = 'relu'
  embedding_init_multiplier: Optional[float] = None

  @nn.compact
  def __call__(self, x, train):
    bot_mlp_input, cat_features = jnp.split(x, [self.num_dense_features], 1)
    cat_features = jnp.asarray(cat_features, dtype=jnp.int32)

    activation_fn = model_utils.ACTIVATIONS[self.activation_function]

    # bottom mlp
    mlp_bottom_dims = self.mlp_bottom_dims

    bot_mlp_input = nn.Dense(
        mlp_bottom_dims[0],
        kernel_init=jnn.initializers.glorot_uniform(),
        bias_init=jnn.initializers.normal(
            stddev=jnp.sqrt(1.0 / mlp_bottom_dims[0])),
    )(bot_mlp_input)
    bot_mlp_input = activation_fn(bot_mlp_input)

    for dense_dim in mlp_bottom_dims[1:]:
      x = nn.Dense(
          dense_dim,
          kernel_init=jnn.initializers.glorot_uniform(),
          bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim)),
      )(bot_mlp_input)
      bot_mlp_input += activation_fn(x)

    base_init_fn = jnn.initializers.uniform(scale=1.0)
    if self.embedding_init_multiplier is None:
      embedding_init_multiplier = 1 / jnp.sqrt(self.vocab_size)
    else:
      embedding_init_multiplier = self.embedding_init_multiplier
    # Embedding table init and lookup for a single unified table.
    idx_lookup = jnp.reshape(cat_features, [-1]) % self.vocab_size
    def scaled_init(key, shape, dtype=jnp.float_):
      return base_init_fn(key, shape, dtype) * embedding_init_multiplier

    embedding_table = self.param(
        'embedding_table',
        scaled_init,
        [self.vocab_size, self.embed_dim])

    embed_features = embedding_table[idx_lookup]
    batch_size = bot_mlp_input.shape[0]
    embed_features = jnp.reshape(
        embed_features, (batch_size, 26 * self.embed_dim))
    top_mlp_input = jnp.concatenate([bot_mlp_input, embed_features], axis=1)
    mlp_input_dim = top_mlp_input.shape[1]
    mlp_top_dims = self.mlp_top_dims
    num_layers_top = len(mlp_top_dims)
    top_mlp_input = nn.Dense(
        mlp_top_dims[0],
        kernel_init=jnn.initializers.normal(
            stddev=jnp.sqrt(2.0 / (mlp_input_dim + mlp_top_dims[0]))),
        bias_init=jnn.initializers.normal(
            stddev=jnp.sqrt(1.0 / mlp_top_dims[0])))(
                top_mlp_input)
    top_mlp_input = activation_fn(top_mlp_input)
    for layer_idx, fan_out in list(enumerate(mlp_top_dims))[1:-1]:
      fan_in = mlp_top_dims[layer_idx - 1]
      x = nn.Dense(
          fan_out,
          kernel_init=jnn.initializers.normal(
              stddev=jnp.sqrt(2.0 / (fan_in + fan_out))),
          bias_init=jnn.initializers.normal(
              stddev=jnp.sqrt(1.0 / mlp_top_dims[layer_idx])))(
                  top_mlp_input)
      x = activation_fn(x)
      if self.dropout_rate > 0.0 and layer_idx == num_layers_top - 2:
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train)(x)
      top_mlp_input += x
    # In the DLRM model the last layer width is always 1. We can hardcode that
    # below.
    logits = nn.Dense(
        1,
        kernel_init=jnn.initializers.normal(
            stddev=jnp.sqrt(2.0 / (mlp_top_dims[-2] + 1))),
        bias_init=jnn.initializers.normal(
            stddev=jnp.sqrt(1.0)))(top_mlp_input)
    return logits


class DLRMResNetModel(base_model.BaseModel):
  """DLRMResNetModel init2winit class."""

  def build_flax_module(self):
    """DLRMResNet for ad click probability prediction."""
    return DLRMResNet(
        activation_function=self.hps.activation_function,
        embedding_init_multiplier=self.hps.embedding_init_multiplier,
        vocab_size=self.hps.vocab_size,
        mlp_bottom_dims=self.hps.mlp_bottom_dims,
        mlp_top_dims=self.hps.mlp_top_dims,
        num_dense_features=self.hps.num_dense_features,
        embed_dim=self.hps.embed_dim,
        keep_diags=self.hps.keep_diags,
        dropout_rate=self.hps.dropout_rate)

  def get_fake_inputs(self, hps):
    """Helper method solely for purpose of initalizing the model."""
    # NOTE(dsuo): hps.input_shape for `criteo_terabyte_input_pipeline` is
    # (39,)
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs

