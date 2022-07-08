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

"""Transformer-based machine translation model.

Adapted from third_party/py/language/google/generation/tsukuyomi/models.py
"""
from typing import Any, Optional

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax import lax
from jax.nn import initializers
from jax.nn import one_hot
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        batch_size=64,
        share_embeddings=False,
        logits_via_embedding=False,
        emb_dim=512,
        num_heads=8,
        enc_num_layers=4,
        dec_num_layers=4,
        qkv_dim=512,
        mlp_dim=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        optimizer='adam',
        opt_hparams={
            'beta1': .9,
            'beta2': .98,
            'epsilon': 1e-9,
            'weight_decay': 0.0
        },
        layer_rescale_factors={},
        normalizer='layer_norm',
        lr_hparams={
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound'
        },
        label_smoothing=0.1,
        l2_decay_factor=None,
        l2_decay_rank_threshold=0,
        rng_seed=-1,
        use_shallue_label_smoothing=False,
        model_dtype='float32',
        grad_clip=None,
        enc_self_attn_kernel_init='xavier_uniform',
        dec_self_attn_kernel_init='xavier_uniform',
        dec_cross_attn_kernel_init='xavier_uniform',
        decode=False,
    ))


def _get_dtype(use_bfloat16):
  if use_bfloat16:
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32
  return dtype


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: Maximum possible length for the input.
      min_scale: <float> Minimum frequency-scale in sine grating.
      max_scale: <float> Maximum frequency-scale in sine grating.

  Returns:
      init: Init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2:2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  max_len: Maximum possible length for the input.
  posemb_init: Positional embedding initializer, if None, then use a fixed
    (non-learned) sinusoidal embedding table.
  decode: whether to use an autoregressive cache.
  """
  max_len: int = 512
  posemb_init: Optional[model_utils.Initializer] = None
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               train=True):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: <float>[batch_size, sequence_length, hidden_size] Input data.
      inputs_positions: [Same as above.] Position indices for packed sequences.
      train: if it is training.

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
      pos_embedding = self.param(
          'pos_embedding', pos_emb_shape, self.posemb_init)
    pe = pos_embedding[:, :length, :]
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
  """Transformer MLP / feed-forward block."""
  mlp_dim: int
  dtype: model_utils.Dtype = jnp.float32
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
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(x)
    output = nn.Dropout(
        rate=self.dropout_rate, deterministic=not train)(output)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    qkv_dim: <int> Dimension of the query/key/value.
    mlp_dim: <int> Dimension of the mlp on top of attention block.
    num_heads: <int> Number of heads.
    dtype: Dtype of the computation (default: float32).
    dropout_rate: <float> Dropout rate.
    attention_dropout_rate: <float> Dropout rate for attention weights.
    normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
      'pre_layer_norm', 'none'
    enc_self_attn_kernel_init_fn: initializer for encoder's
      self attention matrices.
  """
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: model_utils.Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  enc_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long

  @nn.compact
  def __call__(self,
               inputs,
               encoder_mask=None,
               train=True):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.
      train: if it is training.

    Returns:
      output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    if self.normalizer in [
        'batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
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

    x = maybe_pre_normalize()(inputs)
    x = nn.SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=self.enc_self_attn_kernel_init_fn,
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        name='EncoderSelfAttention')(
            x, mask=encoder_mask, deterministic=not train)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    x = x + inputs

    x = maybe_post_normalize()(x)
    # MLP block.
    y = maybe_pre_normalize()(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        name='MLPBlock')(y, train=train)

    res = x + y
    return maybe_post_normalize()(res)


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    dtype: Dtype of the computation (default: float32).
    dropout_rate: <float> Dropout rate.
    attention_dropout_rate: <float> Dropout rate for attention weights
    normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
      'pre_layer_norm', 'none'
    dec_self_attn_kernel_init_fn: initializer for decoder's
      self attention matrices.
    dec_cross_attn_kernel_init_fn: initializer for decoder's
      cross attention matrices.
    decode: whether to use an autoregressive cache.
  """
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: model_utils.Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  dec_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  dec_cross_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  decode: bool = False

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               train=True):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.
      train: if it is training.

    Returns:
      output after transformer encoder-decoder block.
    """
    # Decoder block.
    assert targets.ndim == 3
    if self.normalizer in [
        'batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
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

    x = maybe_pre_normalize()(targets)
    x = nn.SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=self.dec_self_attn_kernel_init_fn,
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        decode=self.decode,
        name='DecoderSelfAttention')(x, decoder_mask, deterministic=not train)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    x = x + targets

    x = maybe_post_normalize()(x)
    # Encoder-Decoder block.
    y = maybe_pre_normalize()(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        kernel_init=self.dec_cross_attn_kernel_init_fn,
        bias_init=nn.initializers.normal(stddev=1e-6),
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate)(
            y, encoded, encoder_decoder_mask, deterministic=not train)

    y = nn.Dropout(rate=self.dropout_rate)(
        y, deterministic=not train)
    y = y + x

    y = maybe_post_normalize()(y)
    # MLP block.
    z = maybe_pre_normalize()(y)
    z = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        name='MLPBlock')(z, train=train)

    res = y + z
    return maybe_post_normalize()(res)


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

    vocab_size: size of the vocabulary
    shared_embedding: a shared embedding layer to use.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding
    num_heads: number of heads
    enc_num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    dropout_rate: dropout rate
    normalizer: One of 'batch_norm', 'layer_norm', 'none'
    attention_dropout_rate: dropout rate for attention weights
    enc_self_attn_kernel_init_fn: initializer for encoder's
      self attention matrices.
  """
  vocab_size: int
  shared_embedding: Any = None
  use_bfloat16: bool = False
  emb_dim: int = 512
  num_heads: int = 8
  enc_num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  attention_dropout_rate: float = 0.1
  enc_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None,
               encoder_mask=None,
               train=True):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.
      train: if it is training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    # Input embedding.
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='input_vocab_embeddings')
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(
        max_len=self.max_len,
        decode=False,
        name='posembed_input')(
            x, inputs_positions=inputs_positions, train=train)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    if self.use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input encoder.
    for lyr in range(self.enc_num_layers):
      x = Encoder1DBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          normalizer=self.normalizer,
          enc_self_attn_kernel_init_fn=self.enc_self_attn_kernel_init_fn,
          name=f'encoderblock_{lyr}')(
              x,
              encoder_mask=encoder_mask,
              train=train)
    if self.normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=dtype)
      x = maybe_normalize()(x)
    return x


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

    output_vocab_size: size of the vocabulary.
    shared_embedding: a shared embedding layer to use.
    logits_via_embedding: bool: whether final logit transform shares embedding
      weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    dec_num_layers: number of layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    decode: whether to use an autoregressive cache.
    dropout_rate: dropout rate.
    normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
      'pre_layer_norm', 'none'
    attention_dropout_rate: dropout rate for attention weights.
    dec_self_attn_kernel_init_fn: initializer for decoder's
      self attention matrices.
    dec_cross_attn_kernel_init_fn: initializer for decoder's
      cross attention matrices.
  """
  output_vocab_size: int
  shared_embedding: Any = None
  logits_via_embedding: bool = False
  use_bfloat16: bool = False
  emb_dim: int = 512
  num_heads: int = 8
  dec_num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  decode: bool = False
  dropout_rate: float = 0.1
  normalizer: str = 'layer_norm'
  attention_dropout_rate: float = 0.1
  dec_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  dec_cross_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               targets_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               train=True):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

      train: whether it is training.

    Returns:
      output of a transformer decoder.
    """
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)
    dtype = _get_dtype(self.use_bfloat16)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=self.output_vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='output_vocab_embeddings')
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if not self.decode:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        max_len=self.max_len,
        decode=self.decode,
        name='posembed_output')(
            y, inputs_positions=targets_positions, train=train)
    y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)

    if self.use_bfloat16:
      y = y.astype(jnp.bfloat16)

    # Target-Input Decoder
    for lyr in range(self.dec_num_layers):
      y = EncoderDecoder1DBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          normalizer=self.normalizer,
          dec_self_attn_kernel_init_fn=self.dec_self_attn_kernel_init_fn,
          dec_cross_attn_kernel_init_fn=self.dec_cross_attn_kernel_init_fn,
          decode=self.decode,
          name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              train=train)
    if self.normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=dtype)
      y = maybe_normalize()(y)

    # Decoded Logits
    if self.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])

    else:
      logits = nn.Dense(
          self.output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')(y)
    return logits


# The following final model is simple but looks verbose due to all the
# repetitive keyword argument plumbing.  It just sticks the Encoder and
# Decoder in series for training, but allows running them separately for
# inference.


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

    vocab_size: size of the input vocabulary.
    output_vocab_size: size of the output vocabulary. If None, the output
      vocabulary size is assumed to be the same as vocab_size.
    share_embeddings: bool: share embedding layer for inputs and targets.
    logits_via_embedding: bool: whether final logit transform shares embedding
      weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    enc_num_layers: number of encoder layers.
    dec_num_layers: number of decoder layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.
    normalizer: One of 'batch_norm', 'layer_norm', 'none'
    enc_self_attn_kernel_init_fn: initializer for encoder's
      self attention matrices.
    dec_self_attn_kernel_init_fn: initializer for decoder's
      self attention matrices.
    dec_cross_attn_kernel_init_fn: initializer for decoder's
      cross attention matrices.
    decode: whether to use an autoregressive cache.
  """
  vocab_size: Optional[int] = None
  output_vocab_size: Optional[int] = None
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  use_bfloat16: bool = False
  emb_dim: int = 512
  num_heads: int = 8
  enc_num_layers: int = 6
  dec_num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.3
  attention_dropout_rate: float = 0.3
  normalizer: str = 'layer_norm'
  enc_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  dec_self_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  dec_cross_attn_kernel_init_fn: model_utils.Initializer = initializers.xavier_uniform()  # pylint: disable=line-too-long
  should_decode: bool = False

  def setup(self):
    if self.share_embeddings:
      if self.output_vocab_size is not None:
        assert self.output_vocab_size == self.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='VocabEmbeddings')
    else:
      self.shared_embedding = None

    self.encoder = Encoder(
        vocab_size=self.vocab_size,
        shared_embedding=self.shared_embedding,
        use_bfloat16=self.use_bfloat16,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        enc_num_layers=self.enc_num_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        max_len=self.max_len,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        normalizer=self.normalizer,
        enc_self_attn_kernel_init_fn=self.enc_self_attn_kernel_init_fn,
        name='encoder')
    self.decoder = Decoder(
        output_vocab_size=self.output_vocab_size,
        shared_embedding=self.shared_embedding,
        logits_via_embedding=self.logits_via_embedding,
        use_bfloat16=self.use_bfloat16,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        dec_num_layers=self.dec_num_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        max_len=self.max_len,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        normalizer=self.normalizer,
        dec_self_attn_kernel_init_fn=self.dec_self_attn_kernel_init_fn,
        dec_cross_attn_kernel_init_fn=self.dec_self_attn_kernel_init_fn,
        decode=self.should_decode,
        name='decoder')

  @nn.compact
  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None,
               train=False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      train: whether it is training.

    Returns:
      Output: <float>[batch_size, target_sequence_length, qkv_dim]
    """
    encoded = self.encode(inputs,
                          inputs_positions=inputs_positions,
                          inputs_segmentation=inputs_segmentation,
                          train=train)

    logits = self.decode(encoded,
                         inputs,  # only used for masks
                         targets,
                         targets_positions=targets_positions,
                         inputs_segmentation=inputs_segmentation,
                         targets_segmentation=targets_segmentation,
                         train=train)
    return logits.astype(jnp.float32) if self.use_bfloat16 else logits

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while using a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  def encode(self,
             inputs,
             inputs_positions=None,
             inputs_segmentation=None,
             train=False):
    # Make padding attention mask.
    dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
    encoder_mask = nn.make_attention_mask(
        inputs > 0, inputs > 0, dtype=dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if inputs_segmentation is not None:
      encoder_mask = nn.combine_masks(
          encoder_mask,
          nn.make_attention_mask(inputs_segmentation,
                                 inputs_segmentation,
                                 jnp.equal,
                                 dtype=dtype))
    encoded = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        encoder_mask=encoder_mask,
        train=train)

    return encoded

  def decode(self,
             encoded,
             inputs,
             targets,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             train=False):
    # Make padding attention masks.
    dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
    if self.should_decode:
      # For fast autoregressive decoding, only a special encoder-decoder mask is
      # used.
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0, dtype=dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=dtype),
          nn.make_causal_mask(targets, dtype=dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs > 0, dtype=dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 targets_segmentation,
                                 jnp.equal,
                                 dtype=dtype))
      encoder_decoder_mask = nn.combine_masks(
          encoder_decoder_mask,
          nn.make_attention_mask(targets_segmentation,
                                 inputs_segmentation,
                                 jnp.equal,
                                 dtype=dtype))

    logits = self.decoder(
        encoded,
        targets,
        targets_positions=targets_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        train=train)
    return logits


class TransformerTranslate(base_model.BaseModel):
  """Transformer Model for machine translation."""

  # pylint: disable=useless-super-delegation
  def __init__(self, hps, dataset_meta_data, loss_name, metrics_name):
    super().__init__(hps, dataset_meta_data, loss_name, metrics_name)
    # TODO(ankugarg): Initialize cache for fast auto-regressive decoding here.
    # Also, initilaize tokenizer here to de-tokenize predicted logits
    # from beach search to target language sequence.
  # pylint: disable=useless-super-delegation

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates cross_entopy on the given batch."""

    # TODO(ankugarg): Augment with other metrics like log-perplexity.
    logits = self.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        batch['inputs'],
        batch['targets'],
        inputs_positions=batch.get('inputs_positions'),
        targets_positions=batch.get('targets_positions'),
        inputs_segmentation=batch.get('inputs_segmentation'),
        targets_segmentation=batch.get('targets_segmentation'),
        train=False)

    weights = batch.get('weights')
    targets = batch['targets']
    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = one_hot(batch['targets'], logits.shape[-1])

    # Add log-perplexity metric.
    return self.metrics_bundle.gather_from_model_output(
        logits=logits, targets=targets, weights=weights, axis_name='batch')

  def apply_on_batch(self, params, batch_stats, batch, **apply_kwargs):
    """Wrapper around flax_module.apply."""
    if batch_stats is not None:
      variables = {'params': params, 'batch_stats': batch_stats}
    else:
      variables = {'params': params}

    return self.flax_module.apply(
        variables,
        batch['inputs'],
        batch['targets'],
        batch.get('inputs_positions'),
        batch.get('targets_positions'),
        batch.get('inputs_segmentation'),
        batch.get('targets_segmentation'),
        **apply_kwargs)

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    """Return cross entropy loss with (optional) L2 penalty on the weights."""

    # inputs/targets positions and segmentations are required when we have
    # packed examples.
    logits, new_batch_stats = self.apply_on_batch(
        params,
        batch_stats,
        batch,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng},
        train=True)

    weights = batch.get('weights')
    targets = batch['targets']

    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = one_hot(batch['targets'], logits.shape[-1])
    # Optionally apply label smoothing.
    if self.hps.get('label_smoothing') is not None:
      targets = model_utils.apply_label_smoothing(
          targets, self.hps.get('label_smoothing'))
    total_loss = self.loss_fn(logits, targets, weights)

    if self.hps.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(
          params, self.hps.l2_decay_rank_threshold)
      total_loss += 0.5 * self.hps.l2_decay_factor * l2_loss
    return total_loss, (new_batch_stats)

  def build_flax_module(self):
    max_len = max(self.hps.max_target_length, self.hps.max_eval_target_length,
                  self.hps.max_predict_length)
    enc_self_attn_kernel_init_fn = model_utils.INITIALIZERS[
        self.hps.enc_self_attn_kernel_init]()
    dec_self_attn_kernel_init_fn = model_utils.INITIALIZERS[
        self.hps.dec_self_attn_kernel_init]()
    dec_cross_attn_kernel_init_fn = model_utils.INITIALIZERS[
        self.hps.dec_cross_attn_kernel_init]()
    use_bfloat16 = self.hps.model_dtype == 'bfloat16'

    return Transformer(
        vocab_size=self.hps['output_shape'][-1],
        output_vocab_size=self.hps['output_shape'][-1],
        share_embeddings=self.hps.share_embeddings,
        logits_via_embedding=self.hps.logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=self.hps.emb_dim,
        num_heads=self.hps.num_heads,
        enc_num_layers=self.hps.enc_num_layers,
        dec_num_layers=self.hps.dec_num_layers,
        qkv_dim=self.hps.qkv_dim,
        mlp_dim=self.hps.mlp_dim,
        max_len=max_len,
        dropout_rate=self.hps.dropout_rate,
        normalizer=self.hps.normalizer,
        attention_dropout_rate=self.hps.attention_dropout_rate,
        enc_self_attn_kernel_init_fn=enc_self_attn_kernel_init_fn,
        dec_self_attn_kernel_init_fn=dec_self_attn_kernel_init_fn,
        dec_cross_attn_kernel_init_fn=dec_cross_attn_kernel_init_fn,
        should_decode=self.hps.decode,
    )
