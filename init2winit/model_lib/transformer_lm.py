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

from flax import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax
from jax import lax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


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
            'initial_value': 0.05,
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
  """

  def apply(self,
            inputs,
            num_embeddings,
            features,
            mode='input',
            emb_init=nn.initializers.normal(stddev=1.0)):
    """Applies Embed module.

    Args:
      inputs: input data
      num_embeddings: number of embedding
      features: size of the embedding dimension
      mode: either 'input' or 'output' -> to share input/output embedding
      emb_init: embedding initializer

    Returns:
      output which is embedded input data
    """
    embedding = self.param('embedding', (num_embeddings, features), emb_init)
    if mode == 'input':
      if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError('Input type must be an integer or unsigned integer.')
      return jnp.take(embedding, inputs, axis=0)
    if mode == 'output':
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
  """Adds learned positional embeddings to the inputs."""

  def apply(self,
            inputs,
            max_len=2048,
            posemb_init=nn.initializers.normal(stddev=1.0),
            cache=None):
    """Applies AddPositionEmbs module.

    Args:
      inputs: input data
      max_len: maximum possible length for the input
      posemb_init: positional embedding initializer
      cache: flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(np.array((4, 1, 1), dtype=np.int32))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        one = jnp.array(1, jnp.uint32)
        cache_entry = cache_entry.replace(i=cache_entry.i + one)
        cache.store(cache_entry)
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP block."""

  def apply(self,
            inputs,
            mlp_dim,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(inputs, mlp_dim, kernel_init=kernel_init, bias_init=bias_init)
    x = nn.gelu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x, actual_out_dim, kernel_init=kernel_init, bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Transformer1DBlock(nn.Module):
  """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            causal_mask=False,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            train=True,
            normalizer='layer_norm',
            attention_fn=None,
            cache=None):
    """Applies Transformer1DBlock module.

    Args:
      inputs: input data
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: number of heads
      causal_mask: bool, mask future or not
      padding_mask: bool, mask padding tokens
      dropout_rate: dropout rate
      attention_dropout_rate: dropout rate for attention weights
      train: bool: if model is training.
      normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
        'pre_layer_norm', 'none'
      attention_fn: Attention function to use. If None, defaults to
        nn.dot_product_attention.
      cache: flax autoregressive cache for fast decoding.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
      maybe_pre_normalize = model_utils.get_normalizer(normalizer, train)
      maybe_post_normalize = model_utils.get_normalizer('none', train)
    elif normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer('none', train)
      maybe_post_normalize = model_utils.get_normalizer(normalizer, train)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(normalizer))

    x = maybe_pre_normalize(inputs)

    if attention_fn is None:
      attention_fn = nn.dot_product_attention
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=causal_mask,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        attention_fn=attention_fn,
        dropout_rate=attention_dropout_rate,
        deterministic=not train,
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    x = x + inputs
    x = maybe_post_normalize(x)

    # MLP block.
    y = maybe_pre_normalize(x)
    y = MlpBlock(
        y, mlp_dim=mlp_dim, dropout_rate=dropout_rate, deterministic=not train)
    res = x + y

    return maybe_post_normalize(res)


class TransformerLM(nn.Module):
  """Transformer Model for language modeling."""

  def apply(self,
            inputs,
            vocab_size,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            causal=True,
            shift=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalizer='layer_norm',
            attention_fn=None,
            cache=None,
            pad_token=0):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
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
      cache: flax autoregressive cache for fast decoding.
      pad_token: Indicates which input tokens are padded.

    Returns:
      output of a transformer decoder.
    """
    padding_mask = jnp.where(inputs != pad_token, 1, 0).astype(jnp.float32)
    assert inputs.ndim == 2  # (batch, len)
    x = inputs
    if shift:
      if not causal:
        raise ValueError('Cannot have shift=True and causal=False')
      x = shift_right(x)
    x = x.astype('int32')
    x = Embed(x, num_embeddings=vocab_size, features=emb_dim, name='embed')
    x = AddPositionEmbs(
        x,
        max_len=max_len,
        posemb_init=sinusoidal_init(max_len=max_len),
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)
    for _ in range(num_layers):
      x = Transformer1DBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          causal_mask=causal,
          padding_mask=padding_mask,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          train=train,
          attention_fn=attention_fn,
          cache=cache,
          normalizer=normalizer,
      )
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(normalizer, train)
      x = maybe_normalize(x)
    logits = nn.Dense(
        x,
        vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    return logits


class TransformerLM1B(base_model.BaseModel):

  def build_flax_module(self):
    max_len = max(self.hps.max_target_length, self.hps.max_eval_target_length)
    return TransformerLM.partial(
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
        pad_token=self.dataset_meta_data.get('pad_token', 0),
    )


