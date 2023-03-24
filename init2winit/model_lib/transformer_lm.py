# coding=utf-8
# Copyright 2023 The init2winit Authors.
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

Adapted from
https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/examples/lm1b/models.py
"""
from typing import Any, Optional

from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import attention
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
            'weight_decay': 0.1
        },
        layer_rescale_factors={},
        normalizer='layer_norm',
        lr_hparams={
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup'
        },
        label_smoothing=None,
        l2_decay_factor=None,
        l2_decay_rank_threshold=0,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
        decode=False,
        normalize_attention=False,
    ))


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=dtype)
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
  def __call__(self, inputs, inputs_positions=None, dtype=np.float32):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      dtype: the dtype used for computation.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=self.max_len)(None, pos_emb_shape,
                                                            dtype)
    else:
      pos_embedding = self.param('pos_embedding', self.posemb_init,
                                 pos_emb_shape, dtype)
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
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
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
  dtype: model_utils.Dtype = np.float32
  kernel_init: model_utils.Initializer = nn.initializers.xavier_uniform()
  bias_init: model_utils.Initializer = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, train):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        param_dtype=self.dtype)(
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    output = nn.Dense(
        actual_out_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        param_dtype=self.dtype)(
            x)
    output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
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
  dtype: Any = jnp.float32
  decode: bool = False
  normalize_attention: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               train,
               decoder_mask=None,
               encoder_decoder_mask=None,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Transformer1DBlock module.

    Args:
      inputs: input data
      train: bool: if model is training.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output after transformer block.

    """

    # Attention block.
    assert inputs.ndim == 3
    if self.normalizer in [
        'batch_norm', 'layer_norm', 'pre_layer_norm', 'none'
    ]:
      maybe_pre_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=self.dtype)
      maybe_post_normalize = model_utils.get_normalizer(
          'none', train, dtype=self.dtype)
    elif self.normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer(
          'none', train, dtype=self.dtype)
      maybe_post_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=self.dtype)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))

    x = maybe_pre_normalize(param_dtype=self.dtype)(inputs)

    if self.attention_fn is None:
      attention_fn = attention.dot_product_attention
    else:
      attention_fn = self.attention_fn
    x = attention.SelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        decode=self.decode,
        dtype=self.dtype,
        param_dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        attention_fn=attention_fn,
        dropout_rate=self.attention_dropout_rate,
        normalize_attention=self.normalize_attention,
        deterministic=not train)(x, decoder_mask)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = x + inputs
    x = maybe_post_normalize(param_dtype=self.dtype)(x)

    # MLP block.
    y = maybe_pre_normalize(param_dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate, dtype=self.dtype)(
            y, train=train)
    res = x + y

    return maybe_post_normalize(param_dtype=self.dtype)(res)


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
  shared_embedding: Any = None
  logits_via_embedding: bool = False
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
  normalize_attention: bool = False
  normalizer: str = 'layer_norm'
  attention_fn: Optional[Any] = None
  model_dtype: str = None
  decode: bool = False
  pad_token: int = 0

  @nn.compact
  def __call__(self,
               inputs,
               train,
               inputs_positions=None,
               inputs_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      train: bool: if model is training.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      output of a transformer decoder.
    """
    assert inputs.ndim == 2  # (batch, len)
    dtype = utils.dtype_from_str(self.model_dtype)

    if self.decode:
      # for fast autoregressive decoding we use no decoder mask
      decoder_mask = None
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(inputs > 0, inputs > 0, dtype=dtype),
          nn.make_causal_mask(inputs, dtype=dtype))

    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              inputs_segmentation, inputs_segmentation, jnp.equal, dtype=dtype))

    y = inputs.astype('int32')
    if not self.decode:
      y = shift_inputs(y, segment_ids=inputs_segmentation)

    # TODO(gdahl,znado): this code appears to be accessing out-of-bounds
    # indices for dataset_lib:proteins_test. This will break when jnp.take() is
    # updated to return NaNs for out-of-bounds indices.
    # Debug why this is the case.
    y = jnp.clip(y, 0, self.vocab_size - 1)

    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          dtype=dtype,
          param_dtype=dtype,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = output_embed(y)

    y = AddPositionEmbs(
        max_len=self.max_len,
        posemb_init=sinusoidal_init(max_len=self.max_len),
        decode=self.decode,
        name='posembed_output')(
            y, inputs_positions=inputs_positions, dtype=dtype)
    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)

    y = y.astype(dtype)

    for _ in range(self.num_layers):
      y = Transformer1DBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          normalize_attention=self.normalize_attention,
          attention_fn=self.attention_fn,
          normalizer=self.normalizer,
          dtype=dtype)(
              inputs=y,
              train=train,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=None,
              inputs_positions=None,
              inputs_segmentation=None,)
    if self.normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=dtype)
      y = maybe_normalize(param_dtype=dtype)(y)

    if self.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          self.vocab_size,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          dtype=dtype,
          param_dtype=dtype,
          name='logits_dense')(
              y)

    return logits.astype(dtype)


class TransformerLM1B(base_model.BaseModel):
  """Model class for Transformer language model for LM1B dataset."""

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
        normalize_attention=self.hps.normalize_attention,
        normalizer=self.hps.normalizer,
        decode=self.hps.decode,
        model_dtype=self.hps.model_dtype,
        pad_token=self.dataset_meta_data.get('pad_token', 0),
    )

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs


