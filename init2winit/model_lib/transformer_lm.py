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

"""Transformer-based language models.

Adapted from third_party/py/flax/examples/lm1b/models.py
"""
from typing import Any, Optional

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax
from jax import lax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


Array = Any

# These reproduce the flax example.
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        batch_size=512,
        emb_dim=128,
        num_heads=8,
        num_layers=6,
        qkv_dim=128,
        mlp_dim=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        optimizer='adam',
        opt_hparams={
            'beta1': .9,
            'beta2': .98,
            'epsilon': 1e-9,
            'weight_decay': 1e-1
        },
        layer_rescale_factors={},
        normalizer='layer_norm',
        lr_hparams={
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound'
        },
        label_smoothing=None,
        l2_decay_factor=None,
        l2_decay_rank_threshold=0,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
        decode=False,
    ))


def shift_right(x):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


class Embed(nn.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.

  num_embeddings: number of embedding
  features: size of the embedding dimension
  mode: either 'input' or 'output' -> to share input/output embedding
  emb_init: embedding initializer
  """
  num_embeddings: int
  features: int
  mode: str = 'input'
  emb_init: model_utils.Initializer = nn.initializers.normal(stddev=1.0)

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Embed module.

    Args:
      inputs: input data
      train: unused

    Returns:
      output which is embedded input data
    """
    del train
    embedding = self.param(
        'embedding', self.emb_init, (self.num_embeddings, self.features))
    if self.mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if self.mode == 'output':
      return jnp.einsum('bld,vd->blv', inputs, embedding)


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    max_len: maximum possible length for the input
    posemb_init: positional embedding initializer
    decode: whether to run in single-position autoregressive mode.
  """
  max_len: int = 2048
  posemb_init: model_utils.Initializer = nn.initializers.normal(stddev=1.0)
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               train,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      train: unused.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    del train
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=self.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP block."""
  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: model_utils.Initializer = nn.initializers.xavier_uniform()
  bias_init: model_utils.Initializer = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    output = nn.Dense(
        actual_out_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(x)
    output = nn.Dropout(
        rate=self.dropout_rate, deterministic=not train)(output)
    return output


class Transformer1DBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr).

    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
      'pre_layer_norm', 'none'
    attention_fn: Attention function to use. If None, defaults to
      nn.dot_product_attention.
  """
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  attention_fn: Optional[Any] = None
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Transformer1DBlock module.

    Args:
      inputs: input data
      train: bool: if model is training.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    if self.normalizer in [
        'batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
      maybe_pre_normalize = model_utils.get_normalizer(self.normalizer, train)
      maybe_post_normalize = model_utils.get_normalizer('none', train)
    elif self.normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer('none', train)
      maybe_post_normalize = model_utils.get_normalizer(self.normalizer, train)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))

    x = maybe_pre_normalize()(inputs)

    if self.attention_fn is None:
      attention_fn = nn.dot_product_attention
    else:
      attention_fn = self.attention_fn
    x = nn.SelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        decode=self.decode,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        attention_fn=attention_fn,
        dropout_rate=self.attention_dropout_rate,
        deterministic=not train)(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = x + inputs
    x = maybe_post_normalize()(x)

    # MLP block.
    y = maybe_pre_normalize()(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate)(y, train=train)
    res = x + y

    return maybe_post_normalize()(res)


class TransformerLM(nn.Module):
  """Transformer Model for language modeling.

    vocab_size: size of the vocabulary
    emb_dim: dimension of embedding
    num_heads: number of heads
    num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    train: bool: if model is training.
    causal: Whether to apply causal masking.
    shift: bool: if we right-shift input - this is only disabled for
      fast, looped single-token autoregressive decoding.
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    normalizer: One of 'batch_norm', 'layer_norm', 'none'
    attention_fn: Attention function to use. If None, defaults to
      nn.dot_product_attention.
    decode: whether to run in single-position autoregressive mode.
    pad_token: Indicates which input tokens are padded.
  """
  vocab_size: int
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  train: bool = False
  causal: bool = True
  shift: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  attention_fn: Optional[Any] = None
  decode: bool = False
  pad_token: int = 0

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      train: bool: if model is training.

    Returns:
      output of a transformer decoder.
    """
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if self.shift:
      if not self.causal:
        raise ValueError('Cannot have shift=True and causal=False')
      x = shift_right(x)
    x = x.astype('int32')
    x = Embed(
        num_embeddings=self.vocab_size,
        features=self.emb_dim,
        name='embed')(x, train=train)
    x = AddPositionEmbs(
        max_len=self.max_len,
        posemb_init=sinusoidal_init(max_len=self.max_len),
        decode=self.decode)(x, train=train)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    for _ in range(self.num_layers):
      x = Transformer1DBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          attention_fn=self.attention_fn,
          normalizer=self.normalizer)(x, train=train)
    if self.normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(self.normalizer, train)
      x = maybe_normalize()(x)
    logits = nn.Dense(
        self.vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(x)
    return logits


class TransformerLM1B(base_model.BaseModel):

  def build_flax_module(self):
    max_len = max(self.hps.max_target_length, self.hps.max_eval_target_length)
    return TransformerLM(
        vocab_size=self.hps['output_shape'][-1],
        emb_dim=self.hps.emb_dim,
        num_heads=self.hps.num_heads,
        num_layers=self.hps.num_layers,
        qkv_dim=self.hps.qkv_dim,
        mlp_dim=self.hps.mlp_dim,
        max_len=max_len,
        shift=self.dataset_meta_data['shift_inputs'],
        causal=self.dataset_meta_data['causal'],
        dropout_rate=self.hps.dropout_rate,
        attention_dropout_rate=self.hps.attention_dropout_rate,
        normalizer=self.hps.normalizer,
        decode=self.hps.decode,
        pad_token=self.dataset_meta_data.get('pad_token', 0),
    )


