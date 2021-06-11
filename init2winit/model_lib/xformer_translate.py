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

"""Transformer-based machine translation model.

Adapted from third_party/py/language/google/generation/tsukuyomi/models.py
"""

from flax import nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
from jax import lax
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
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=512,
            posemb_init=None,
            cache=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: <float>[batch_size, sequence_length, hidden_size] Input data.
      inputs_positions: [Same as above.] Position indices for packed sequences.
      max_len: Maximum possible length for the input.
      posemb_init: Positional embedding initializer, if None, then use a fixed
        (non-learned) sinusoidal embedding table.
      cache: Flax attention cache for fast decoding.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    if posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=max_len)(None, pos_emb_shape,
                                                       None)
    else:
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
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=False,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
    x = nn.Dense(
        inputs,
        mlp_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    x = nn.relu(x)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    output = nn.Dense(
        x,
        actual_out_dim,
        dtype=dtype,
        kernel_init=kernel_init,
        bias_init=bias_init)
    output = nn.dropout(output, rate=dropout_rate, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer decoder layer."""

  def apply(self,
            inputs,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalizer='layer_norm',
            deterministic=False):
    """Applies Encoder1DBlock module.

    Args:
      inputs: <float>[batch_size, input_sequence_length, qkv_dim]
      qkv_dim: <int> Dimension of the query/key/value.
      mlp_dim: <int> Dimension of the mlp on top of attention block.
      num_heads: <int> Number of heads.
      dtype: Dtype of the computation (default: float32).
      inputs_segmentation: input segmentation info for packed examples.
      padding_mask: <bool> Mask padding tokens.
      dropout_rate: <float> Dropout rate.
      attention_dropout_rate: <float> Dropout rate for attention weights.
      normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
        'pre_layer_norm', 'none'
      deterministic: <bool> Deterministic or not (to apply dropout).

    Returns:
      Output: <float>[batch_size, input_sequence_length, qkv_dim]
    """

    # Attention block.
    assert inputs.ndim == 3
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
      maybe_pre_normalize = model_utils.get_normalizer(normalizer,
                                                       not deterministic)
      maybe_post_normalize = model_utils.get_normalizer('none',
                                                        not deterministic)
    elif normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer('none',
                                                       not deterministic)
      maybe_post_normalize = model_utils.get_normalizer(normalizer,
                                                        not deterministic)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(normalizer))

    x = maybe_pre_normalize(inputs)
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        segmentation=inputs_segmentation,
        padding_mask=padding_mask,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + inputs
    x = maybe_post_normalize(x)

    # MLP block.
    y = maybe_pre_normalize(x)
    y = MlpBlock(
        y,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)
    res = x + y

    return maybe_post_normalize(res)


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer."""

  def apply(self,
            targets,
            encoded,
            qkv_dim,
            mlp_dim,
            num_heads,
            dtype=jnp.float32,
            inputs_segmentation=None,
            targets_segmentation=None,
            padding_mask=None,
            key_padding_mask=None,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            deterministic=False,
            normalizer='layer_norm',
            cache=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: <float>[batch_size, target_sequence_length, qkv_dim]
      encoded: <float>[batch_size, input_sequence_length, qkv_dim]
      qkv_dim: Dimension of the query/key/value.
      mlp_dim: Dimension of the mlp on top of attention block.
      num_heads: Number of heads.
      dtype: Dtype of the computation (default: float32).
      inputs_segmentation: Input segmentation info for packed examples.
      targets_segmentation: Iarget segmentation info for packed examples.
      padding_mask: <bool> Mask padding tokens.
      key_padding_mask: <bool> Mask padding tokens.
      dropout_rate: <float> Dropout rate.
      attention_dropout_rate: <float> Dropout rate for attention weights
      deterministic: <bool> Deterministic or not (to apply dropout)
      normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
        'pre_layer_norm', 'none'
      cache: Flax attention cache for fast decoding.

    Returns:
      output: <float>[batch_size, target_sequence_length, qkv_dim]
    """

    # Decoder block.
    assert targets.ndim == 3
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm', 'none']:
      maybe_pre_normalize = model_utils.get_normalizer(normalizer,
                                                       not deterministic)
      maybe_post_normalize = model_utils.get_normalizer('none',
                                                        not deterministic)
    elif normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer('none',
                                                       not deterministic)
      maybe_post_normalize = model_utils.get_normalizer(normalizer,
                                                        not deterministic)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(normalizer))

    x = maybe_pre_normalize(targets)
    x = nn.SelfAttention(
        x,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=x,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        padding_mask=padding_mask,
        segmentation=targets_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic,
        cache=cache)
    x = nn.dropout(x, rate=dropout_rate, deterministic=deterministic)
    x = x + targets
    x = maybe_post_normalize(x)

    # Encoder-Decoder block.
    # TODO(ankugarg): Support for confgurable pre vs post layernorm.
    y = maybe_pre_normalize(x)
    y = nn.SelfAttention(
        y,
        num_heads=num_heads,
        dtype=dtype,
        inputs_kv=encoded,
        qkv_features=qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        padding_mask=padding_mask,
        key_padding_mask=key_padding_mask,
        segmentation=targets_segmentation,
        key_segmentation=inputs_segmentation,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=deterministic)
    y = nn.dropout(y, rate=dropout_rate, deterministic=deterministic)
    y = y + x
    y = maybe_post_normalize(y)

    # MLP block.
    z = maybe_pre_normalize(y)
    z = MlpBlock(
        z,
        mlp_dim=mlp_dim,
        dtype=dtype,
        dropout_rate=dropout_rate,
        deterministic=deterministic)
    res = y + z

    return maybe_post_normalize(res)


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def apply(self,
            inputs,
            vocab_size,
            inputs_positions=None,
            inputs_segmentation=None,
            shared_embedding=None,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            enc_num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            dropout_rate=0.1,
            normalizer='layer_norm',
            attention_dropout_rate=0.1):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      vocab_size: size of the vocabulary
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      shared_embedding: a shared embedding layer to use.
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding
      num_heads: number of heads
      enc_num_layers: number of layers
      qkv_dim: dimension of the query/key/value
      mlp_dim: dimension of the mlp on top of attention block
      max_len: maximum length.
      train: if it is training,
      dropout_rate: dropout rate
      normalizer: One of 'batch_norm', 'layer_norm', 'none'
      attention_dropout_rate: dropout rate for attention weights

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 2  # (batch, len)

    src_padding_mask = (inputs > 0)[..., None]

    # Input Embedding
    if shared_embedding is None:
      input_embed = nn.Embed.partial(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='input_vocab_embeddings')
    else:
      input_embed = shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(
        x,
        inputs_positions=inputs_positions,
        max_len=max_len,
        name='posembed_input')
    x = nn.dropout(x, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(enc_num_layers):
      x = Encoder1DBlock(
          x,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          normalizer=normalizer,
          name=f'encoderblock_{lyr}')
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(normalizer, train)
      x = maybe_normalize(x)
    return x


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""

  def apply(self,
            encoded,
            src_padding_mask,
            targets,
            output_vocab_size,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None,
            shared_embedding=None,
            logits_via_embedding=False,
            shift=True,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            dec_num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=512,
            train=True,
            cache=None,
            dropout_rate=0.1,
            normalizer='layer_norm',
            attention_dropout_rate=0.1):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      src_padding_mask: padding mask for inputs.
      targets: target inputs.
      output_vocab_size: size of the vocabulary.
      targets_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
      tgt_padding_mask: target tokens padding mask.
      shared_embedding: a shared embedding layer to use.
      logits_via_embedding: bool: whether final logit transform shares embedding
        weights.
      shift: whether to shift or not (for fast decoding).
      use_bfloat16: bool: whether use bfloat16.
      emb_dim: dimension of embedding.
      num_heads: number of heads.
      dec_num_layers: number of layers.
      qkv_dim: dimension of the query/key/value.
      mlp_dim: dimension of the mlp on top of attention block.
      max_len: maximum length.
      train: whether it is training.
      cache: flax attention cache for fast decoding.
      dropout_rate: dropout rate.
      normalizer: One of 'batch_norm', 'layer_norm', 'post_layer_norm',
        'pre_layer_norm', 'none'
      attention_dropout_rate: dropout rate for attention weights.

    Returns:
      output of a transformer decoder.
    """
    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)
    dtype = _get_dtype(use_bfloat16)
    # Padding Masks
    if tgt_padding_mask is None:
      tgt_padding_mask = (targets > 0)[..., None]

    # Target Embedding
    if shared_embedding is None:
      output_embed = nn.Embed.shared(
          num_embeddings=output_vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='output_vocab_embeddings')
    else:
      output_embed = shared_embedding

    y = targets.astype('int32')
    if shift:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        y,
        inputs_positions=targets_positions,
        max_len=max_len,
        cache=cache,
        name='posembed_output')
    y = nn.dropout(y, rate=dropout_rate, deterministic=not train)

    if use_bfloat16:
      y = y.astype(jnp.bfloat16)

    # Target-Input Decoder
    for lyr in range(dec_num_layers):
      y = EncoderDecoder1DBlock(
          y,
          encoded,
          qkv_dim=qkv_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          dtype=dtype,
          padding_mask=tgt_padding_mask,
          key_padding_mask=src_padding_mask,
          inputs_segmentation=inputs_segmentation,
          targets_segmentation=targets_segmentation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          deterministic=not train,
          normalizer=normalizer,
          cache=cache,
          name=f'encoderdecoderblock_{lyr}')
    if normalizer in ['batch_norm', 'layer_norm', 'pre_layer_norm']:
      maybe_normalize = model_utils.get_normalizer(normalizer, train)
      y = maybe_normalize(y)

    # Decoded Logits
    if logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])

    else:
      logits = nn.Dense(
          y,
          output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')
    return logits


# The following final model is simple but looks verbose due to all the
# repetitive keyword argument plumbing.  It just sticks the Encoder and
# Decoder in series for training, but allows running them separately for
# inference.


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation."""

  def apply(self,
            inputs,
            targets,
            inputs_positions=None,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            vocab_size=None,
            output_vocab_size=None,
            share_embeddings=False,
            logits_via_embedding=False,
            use_bfloat16=False,
            emb_dim=512,
            num_heads=8,
            enc_num_layers=6,
            dec_num_layers=6,
            qkv_dim=512,
            mlp_dim=2048,
            max_len=2048,
            train=False,
            shift=True,
            dropout_rate=0.3,
            attention_dropout_rate=0.3,
            normalizer='layer_norm',
            cache=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.
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
      train: whether it is training.
      shift: whether to right-shift targets.
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate for attention weights.
      normalizer: One of 'batch_norm', 'layer_norm', 'none'
      cache: flax autoregressive cache for fast decoding.

    Returns:
      Output: <float>[batch_size, target_sequence_length, qkv_dim]
    """
    src_padding_mask = (inputs > 0)[..., None]
    if share_embeddings:
      if output_vocab_size is not None:
        assert output_vocab_size == vocab_size, (
            "can't share embedding with different vocab sizes.")
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='VocabEmbeddings')
    else:
      shared_embedding = None
    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        enc_num_layers=enc_num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalizer=normalizer,
        name='encoder')

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        dec_num_layers=dec_num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalizer=normalizer,
        cache=cache,
        name='decoder')
    return logits.astype(jnp.float32) if use_bfloat16 else logits

  # The following two methods allow us to run the trained Transformer in
  # two parts during fast decoding.  First, we call the encoder branch to
  # encode the inputs, then we call the decoder branch while providing a
  # cache object for iteratively storing keys and values during the decoding
  # process.

  @nn.module_method
  def encode(self,
             inputs,
             vocab_size=None,
             output_vocab_size=None,
             inputs_positions=None,
             inputs_segmentation=None,
             targets_positions=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             dec_num_layers=None,
             enc_num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             normalizer='layer_norm',
             cache=None):
    del (output_vocab_size, shift, targets_positions, targets_segmentation,
         tgt_padding_mask, logits_via_embedding, cache, dec_num_layers)
    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='VocabEmbeddings')
    else:
      shared_embedding = None
    encoded = Encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        enc_num_layers=enc_num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalizer=normalizer,
        name='encoder')

    return encoded

  @nn.module_method
  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             inputs_positions=None,
             vocab_size=None,
             output_vocab_size=None,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None,
             share_embeddings=False,
             logits_via_embedding=False,
             use_bfloat16=False,
             emb_dim=512,
             num_heads=8,
             enc_num_layers=None,
             dec_num_layers=6,
             qkv_dim=512,
             mlp_dim=2048,
             max_len=2048,
             train=True,
             shift=True,
             dropout_rate=0.1,
             attention_dropout_rate=0.1,
             normalizer='layer_norm',
             cache=None):
    del (inputs_positions, enc_num_layers)
    if share_embeddings:
      shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='VocabEmbeddings')
    else:
      shared_embedding = None

    logits = Decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask,
        output_vocab_size=output_vocab_size,
        shared_embedding=shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        dec_num_layers=dec_num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        shift=shift,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        normalizer=normalizer,
        cache=cache,
        name='decoder')

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

  def evaluate_batch(self, flax_module, batch_stats, batch):
    """Evaluates cross_entopy on the given batch."""

    # TODO(ankugarg): Augment with other metrics like log-perplexity.
    with nn.stateful(batch_stats, mutable=False):
      logits = flax_module(
          batch['inputs'],
          batch['targets'],
          batch.get('inputs_positions'),
          batch.get('targets_positions'),
          batch.get('inputs_segmentation'),
          batch.get('targets_segmentation'),
          train=False)

    weights = batch.get('weights')
    targets = batch['targets']
    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = one_hot(batch['targets'], logits.shape[-1])

    # Add log-perplexity metric.
    evaluated_metrics = {}
    for key in self.metrics_bundle:
      per_example_metrics = self.metrics_bundle[key](logits, targets, weights)
      evaluated_metrics[key] = jnp.sum(
          lax.psum(per_example_metrics, axis_name='batch'))

    return evaluated_metrics

  def training_cost(self, flax_module, batch_stats, batch, dropout_rng):
    """Return cross entropy loss with (optional) L2 penalty on the weights."""

    with nn.stateful(batch_stats) as new_batch_stats:
      with nn.stochastic(dropout_rng):
        # inputs/targets positions and segmentations are required
        # when we have packed examples.
        logits = flax_module(
            batch['inputs'],
            batch['targets'],
            batch.get('inputs_positions'),
            batch.get('targets_positions'),
            batch.get('inputs_segmentation'),
            batch.get('targets_segmentation'),
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
      l2_loss = model_utils.l2_regularization(flax_module.params,
                                              self.hps.l2_decay_rank_threshold)
      total_loss += 0.5 * self.hps.l2_decay_factor * l2_loss
    return total_loss, (new_batch_stats)

  def build_flax_module(self):
    max_len = max(self.hps.max_target_length, self.hps.max_eval_target_length,
                  self.hps.max_predict_length)
    return Transformer.partial(
        vocab_size=self.hps['output_shape'][-1],
        output_vocab_size=self.hps['output_shape'][-1],
        share_embeddings=self.hps.share_embeddings,
        logits_via_embedding=self.hps.logits_via_embedding,
        emb_dim=self.hps.emb_dim,
        num_heads=self.hps.num_heads,
        enc_num_layers=self.hps.enc_num_layers,
        dec_num_layers=self.hps.dec_num_layers,
        qkv_dim=self.hps.qkv_dim,
        mlp_dim=self.hps.mlp_dim,
        max_len=max_len,
        shift=self.dataset_meta_data['shift_outputs'],
        dropout_rate=self.hps.dropout_rate,
        normalizer=self.hps.normalizer,
        attention_dropout_rate=self.hps.attention_dropout_rate,
    )
