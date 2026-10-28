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

r"""Conformer.

This model uses a conformer network to convert speech to text.
paper : https://arxiv.org/abs/2005.08100

high-level overview of Conformer encoder layer.

  x = x + 0.5 * FeedForward(x)
  x = x + MHSA(x)
  x = x + ConvolutionBlock(x)
  x = x + 0.5 * FeedForward(x)
  y = layer_norm(x)
"""

import functools
import math
from typing import Any, List

from flax import linen as nn
from flax import struct
from init2winit import utils
from init2winit.model_lib import base_model
from init2winit.model_lib import librispeech_preprocessor as preprocessor
from init2winit.model_lib import lingvo_attention
from init2winit.model_lib import model_utils
from init2winit.model_lib import spectrum_augmenter
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np

MLCOMMONS_DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        activation_function='swish',
        eval_batch_size=128,
        computation_dtype='bfloat16',
        model_dtype='float32',
        encoder_dim=512,
        num_attention_heads=8,
        num_encoder_layers=16,
        convolution_kernel_size=5,
        freq_mask_count=2,
        freq_mask_max_bins=27,
        time_mask_count=10,
        time_mask_max_frames=40,
        time_mask_max_ratio=0.05,
        time_masks_per_frame=0.0,
        use_dynamic_time_mask_max_frames=True,
        use_specaug=True,
        dropout_rate=0.1,
        ## dropout_rate pipes to attention and feed_forward residual dropouts
        aux_dropout_rate=0.1,  ## This pipes to input_dropout
        tie_dropouts=False,
        enable_decoder_pre_layer_norm=True,
        enable_conformer_post_layer_norm=True,
        use_lingvo_attention=False,
        attn_temperature=1.0,
    )
)


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        activation_function='swish',
        eval_batch_size=128,
        computation_dtype='bfloat16',
        model_dtype='float32',
        encoder_dim=512,
        num_attention_heads=8,
        num_encoder_layers=16,
        convolution_kernel_size=5,
        freq_mask_count=2,
        freq_mask_max_bins=27,
        time_mask_count=10,
        time_mask_max_frames=40,
        time_mask_max_ratio=0.05,
        time_masks_per_frame=0.0,
        use_dynamic_time_mask_max_frames=True,
        use_specaug=True,
        residual_dropout_rate=0.1,
        input_dropout_rate=0.1,
        enable_decoder_pre_layer_norm=True,
        enable_conformer_post_layer_norm=True,
        use_lingvo_attention=False,
        attn_temperature=1.0,
    )
)


@struct.dataclass
class ConformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  vocab_size: int = 0
  dtype: Any = jnp.bfloat16
  param_dtype: Any = jnp.float32
  encoder_dim: int = 0
  num_attention_heads: int = 0
  num_encoder_layers: int = 0
  attention_dropout_rate: float = 0.0
  attention_residual_dropout_rate: float = 0.1
  input_dropout_rate: float = 0.0
  conv_residual_dropout_rate: float = 0.0
  feed_forward_dropout_rate: float = 0.0
  feed_forward_residual_dropout_rate: float = 0.1
  convolution_kernel_size: int = 5
  feed_forward_expansion_factor: int = 4
  conv_expansion_factor: int = 2
  conv_subsampling_factor: int = 2
  conv_subsampling_layers: int = 2
  train: bool = False
  use_specaug: bool = False
  freq_mask_count: int = 1
  freq_mask_max_bins: int = 15
  time_mask_count: int = 1
  time_mask_max_frames: int = 50
  time_mask_max_ratio: float = 1.0
  time_masks_per_frame: float = 0.0
  use_dynamic_time_mask_max_frames: bool = False
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  enable_conformer_post_layer_norm: bool = False
  enable_decoder_pre_layer_norm: bool = False
  use_lingvo_attention: bool = False
  attn_temperature: float = 1.0
  activation_function: str = 'swish'


class LayerNorm(nn.Module):
  """Module implementing layer normalization.

  This implementation is same as in this paper:
  https://arxiv.org/pdf/1607.06450.pdf.

  note: we multiply normalized inputs by (1 + scale) and initialize scale to
  zeros, this differs from default flax implementation of multiplying by scale
  and initializing to ones.
  """

  dim: int = 0
  epsilon: float = 1e-6
  param_dtype: Any = jnp.float32

  def setup(self):
    self.scale = self.param(
        'scale', nn.initializers.zeros, [self.dim], self.param_dtype
    )
    self.bias = self.param(
        'bias', nn.initializers.zeros, [self.dim], self.param_dtype
    )

  @nn.compact
  def __call__(self, inputs):
    inputs_f32 = inputs.astype(jnp.float32)
    mean = jnp.mean(inputs_f32, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(inputs_f32 - mean), axis=[-1], keepdims=True)

    normed_inputs = (inputs_f32 - mean) * jax.lax.rsqrt(var + self.epsilon)
    normed_inputs = normed_inputs * (
        1.0 + self.scale.astype(jnp.float32)
    ) + self.bias.astype(jnp.float32)

    return normed_inputs.astype(inputs.dtype)


class Subsample(nn.Module):
  """Module to perform strided convolution in order to subsample inputs.

  Attributes:
    encoder_dim: model dimension of conformer.
    input_dropout_rate: dropout rate for inputs.
    dtype: activation compute dtype.
    param_dtype: parameter dtype.
  """

  encoder_dim: int = 0
  input_dropout_rate: float = 0.0
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    work_dtype = self.dtype if self.dtype is not None else inputs.dtype
    output_paddings = input_paddings
    outputs = jnp.expand_dims(inputs.astype(work_dtype), axis=-1)

    outputs, output_paddings = Conv2dSubsampling(
        input_channels=1,
        output_channels=self.encoder_dim,
        dtype=work_dtype,
        param_dtype=self.param_dtype,
    )(outputs, output_paddings)

    outputs, output_paddings = Conv2dSubsampling(
        input_channels=self.encoder_dim,
        output_channels=self.encoder_dim,
        dtype=work_dtype,
        param_dtype=self.param_dtype,
    )(outputs, output_paddings)

    batch_size, subsampled_lengths, subsampled_dims, channels = outputs.shape

    outputs = jnp.reshape(
        outputs, (batch_size, subsampled_lengths, subsampled_dims * channels)
    )
    outputs = nn.Dense(
        self.encoder_dim,
        use_bias=True,
        dtype=work_dtype,
        param_dtype=self.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
    )(outputs)

    pos_emb = AddPositionalEmbedding(embedding_dim=self.encoder_dim)(
        seq_length=outputs.shape[1]
    ).astype(work_dtype)
    outputs = outputs + pos_emb

    outputs = nn.Dropout(rate=self.input_dropout_rate, deterministic=not train)(
        outputs
    )

    return outputs, output_paddings


class Conv2dSubsampling(nn.Module):
  """Helper module used in Subsample layer.

  1) Performs strided convolution over inputs and then applies non-linearity.
  2) Also performs strided convolution over input_paddings to return the correct
  paddings for downstream layers.
  """

  input_channels: int = 0
  output_channels: int = 0
  filter_stride: List[int] = (2, 2)
  padding: str = 'SAME'
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32

  def setup(self):
    self.filter_shape = (3, 3, self.input_channels, self.output_channels)
    self.kernel = self.param(
        'kernel',
        nn.initializers.xavier_uniform(),
        self.filter_shape,
        self.param_dtype,
    )
    self.bias = self.param(
        'bias',
        nn.initializers.zeros,
        self.output_channels,
        self.param_dtype,
    )

  @nn.compact
  def __call__(self, inputs, paddings):
    compute_dtype = self.dtype if self.dtype is not None else inputs.dtype
    inputs = inputs.astype(compute_dtype)
    kernel = self.kernel.astype(compute_dtype)
    bias = self.bias.astype(compute_dtype)

    # Computing strided convolution to subsample inputs.
    feature_group_count = inputs.shape[3] // self.filter_shape[2]
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=self.filter_stride,
        padding=self.padding,
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count,
    )

    outputs += jnp.reshape(bias, (1,) * (outputs.ndim - 1) + (-1,))
    outputs = nn.relu(outputs)

    # Computing correct paddings post input convolution.
    stride = self.filter_stride[0]
    out_padding = paddings[:, ::stride].astype(
        jnp.result_type(paddings, jnp.float32)
    )

    # Mask outputs by correct paddings to ensure padded elements in inputs map
    # to padded value in outputs.
    mask = (
        1.0 - jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1)
    ).astype(compute_dtype)
    outputs = outputs * mask
    return outputs, out_padding


class FeedForwardModule(nn.Module):
  """Feedforward block of conformer layer."""

  config: ConformerConfig

  @nn.compact
  def __call__(self, inputs, padding_mask=None, train=False):
    config = self.config
    work_dtype = config.dtype if config.dtype is not None else inputs.dtype

    inputs = LayerNorm(
        dim=config.encoder_dim, param_dtype=config.param_dtype
    )(inputs.astype(work_dtype))

    inputs = nn.Dense(
        config.encoder_dim * config.feed_forward_expansion_factor,
        use_bias=True,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
    )(inputs)
    inputs = model_utils.ACTIVATIONS[self.config.activation_function](inputs)
    inputs = nn.Dropout(rate=config.feed_forward_dropout_rate)(
        inputs, deterministic=not train
    )

    if padding_mask is not None:
      padding_mask = padding_mask.astype(work_dtype)
      inputs = inputs * padding_mask

    inputs = nn.Dense(
        config.encoder_dim,
        use_bias=True,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
    )(inputs)
    if padding_mask is not None:
      inputs = inputs * padding_mask

    inputs = nn.Dropout(rate=config.feed_forward_residual_dropout_rate)(
        inputs, deterministic=not train
    )

    return inputs


class AddPositionalEmbedding(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    max_len: maximum possible length for the input
    posemb_init: positional embedding initializer
  """

  min_timescale: int = 1
  max_timescale: int = 10_000
  embedding_dim: int = 512

  @nn.compact
  def __call__(self, seq_length):
    position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    num_timescales = self.embedding_dim // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)
    ) / jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1)
    inv_timescales = self.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
    ).astype(jnp.float32)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = jnp.pad(
        signal, [[0, 0], [0, 0], [0, np.mod(self.embedding_dim, 2)]]
    )
    return signal


# Adapted from lingvo attention layer for query scaling
# https://github.com/tensorflow/lingvo/blob/7de4ca8fff3cb28c2ecb21bbd7b02a964ce727f7/lingvo/jax/layers/attentions.py#L201
class QueryScaler(nn.Module):
  """A layer to scale individual dims of the query attention matrix."""

  dim: int = 0
  param_dtype: Any = jnp.float32

  def setup(self):
    self.scale = self.param(
        'scale', nn.initializers.zeros, [self.dim], self.param_dtype
    )

  @nn.compact
  def __call__(self, inputs):
    inputs_shape = inputs.shape
    if inputs_shape[-1] != self.dim:
      raise ValueError(
          'QueryScaler expects inputs to have'
          ' same last dimension as scaling param.'
      )

    # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
    # can avoid unnecessary XLA op fusion mess on TPU.
    r_softplus_0 = 1.442695041

    scale = (
        r_softplus_0 * jax.nn.softplus(self.scale.astype(jnp.float32))
    ).astype(inputs.dtype)

    return inputs * scale


# Modifying flax linen default dot product attention function to add
# query scaling, reference to original function here :
# https://github.com/google/flax/blob/a9af38085a7a49b571cf37d375060fd683e74972/flax/linen/attention.py#L121
def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    broadcast_dropout=True,
    dropout_rng=None,
    dropout_rate=0.0,
    deterministic=False,
    dtype=jnp.float32,
    param_dtype=jnp.float32,
    precision=None,
    temperature=1.0,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It's slightly modified to add query scaling.
  It calculates the attention weights given query and key and combines the
  values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    param_dtype: the dtype of the query scaling parameter (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    temperature: Constant factor to multiply logits by before computing softmax.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
      query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  work_dtype = dtype if dtype is not None else query.dtype
  query = QueryScaler(dim=query.shape[-1], param_dtype=param_dtype)(
      query.astype(work_dtype)
  )
  key = key.astype(work_dtype)
  value = value.astype(work_dtype)

  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(work_dtype)
  logits = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision
  ).astype(jnp.float32)
  if bias is not None:
    logits = logits + bias.astype(jnp.float32)
  if mask is not None:
    logits = jnp.where(mask, logits, jnp.asarray(-1e9, dtype=jnp.float32))
  attn_weights = jax.nn.softmax(logits, axis=-1).astype(work_dtype)

  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(work_dtype) / jnp.asarray(
        keep_prob, dtype=work_dtype
    )
    attn_weights = attn_weights * multiplier

  # return weighted sum over values for each query position
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  ) * jnp.asarray(temperature, dtype=work_dtype)


class MultiHeadedSelfAttention(nn.Module):
  """Self attention sub-layer used in the Conformer layer.

  Input is first normalized using layer norm. Output is processed using
  multi-headed attention.

  Note: this attention implementation uses a learned scale parameter to scale
  query matrix before passing it to flax attention module.
  """

  config: ConformerConfig = None

  def setup(self):
    dim_per_head = self.config.encoder_dim // self.config.num_attention_heads
    self.self_attention = lingvo_attention.DotProductAttention(
        num_heads=self.config.num_attention_heads,
        hidden_dim=self.config.encoder_dim,
        input_dim=self.config.encoder_dim,
        dim_per_head=dim_per_head,
    )

  def _get_large_negative_number(self, dtype):
    if jnp.issubdtype(dtype, jnp.inexact):
      dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
      dtype_max = jnp.iinfo(dtype).max
    else:
      raise ValueError('Unsupported dtype for inputs.')

    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)

  def convert_paddings_to_mask(self, paddings, dtype=jnp.float32):
    attention_mask = paddings[:, jnp.newaxis, jnp.newaxis, :]
    attention_mask *= self._get_large_negative_number(dtype)

    return attention_mask

  @nn.compact
  def __call__(self, inputs, paddings, train, attention_mask=None):
    config = self.config
    work_dtype = config.dtype if config.dtype is not None else inputs.dtype
    if not self.config.use_lingvo_attention and attention_mask is None:
      valid = (1 - paddings) > 0
      attention_mask = jnp.logical_and(
          valid[:, None, :, None], valid[:, None, None, :]
      )

    inputs = LayerNorm(
        dim=config.encoder_dim, param_dtype=config.param_dtype
    )(inputs.astype(work_dtype))

    if self.config.use_lingvo_attention:
      atten_mask = self.convert_paddings_to_mask(paddings, jnp.float32)

      result = self.self_attention(
          query_vec=inputs,
          key_vec=inputs,
          value_vec=inputs,
          atten_mask=atten_mask,
      )[0].astype(work_dtype)
    else:
      attn_fn = functools.partial(
          dot_product_attention,
          temperature=config.attn_temperature,
          param_dtype=config.param_dtype,
      )
      result = nn.SelfAttention(
          num_heads=config.num_attention_heads,
          qkv_features=config.encoder_dim,
          decode=False,
          dtype=work_dtype,
          param_dtype=config.param_dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.zeros,
          use_bias=True,
          broadcast_dropout=False,
          attention_fn=attn_fn,
          dropout_rate=config.attention_dropout_rate,
          deterministic=not train,
      )(inputs, mask=attention_mask)

    result = nn.Dropout(
        rate=config.attention_residual_dropout_rate, deterministic=not train
    )(result)

    return result


class BatchNorm(nn.Module):
  """Implements batch norm respecting input paddings.

  This implementation takes into account input padding by masking inputs before
  computing mean and variance.

  This is inspired by lingvo jax implementation of BatchNorm:
  https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/jax/layers/normalizations.py#L92

  and the corresponding defaults for momentum and epsilon have been copied over
  from lingvo.
  """

  config: ConformerConfig

  def setup(self):
    dim = self.config.encoder_dim

    self.ra_mean = self.variable(
        'batch_stats', 'mean', lambda s: jnp.zeros(s, jnp.float32), dim
    )
    self.ra_var = self.variable(
        'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), dim
    )

    self.gamma = self.param(
        'scale', nn.initializers.zeros, dim, self.config.param_dtype
    )
    self.beta = self.param(
        'bias', nn.initializers.zeros, dim, self.config.param_dtype
    )

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    out_dtype = (
        self.config.dtype if self.config.dtype is not None else inputs.dtype
    )
    rank = inputs.ndim
    reduce_over_dims = list(range(0, rank - 1))

    padding = jnp.expand_dims(input_paddings, -1).astype(jnp.float32)
    mask = 1.0 - padding
    inputs_f32 = inputs.astype(jnp.float32)
    momentum = self.config.batch_norm_momentum
    epsilon = self.config.batch_norm_epsilon

    if train:
      sum_v = jnp.sum(inputs_f32 * mask, axis=reduce_over_dims, keepdims=False)
      count_v = jnp.maximum(
          jnp.sum(mask, axis=reduce_over_dims, keepdims=False), 1.0
      )
      mean = sum_v / count_v

      centered = inputs_f32 - mean
      sum_vv = jnp.sum(
          jnp.square(centered) * mask,
          axis=reduce_over_dims,
          keepdims=False,
      )

      var = sum_vv / count_v

      self.ra_mean.value = momentum * self.ra_mean.value + (1 - momentum) * mean
      self.ra_var.value = momentum * self.ra_var.value + (1 - momentum) * var
    else:
      mean = self.ra_mean.value.astype(jnp.float32)
      var = self.ra_var.value.astype(jnp.float32)
      centered = inputs_f32 - mean

    inv = (1.0 + self.gamma.astype(jnp.float32)) * jax.lax.rsqrt(var + epsilon)

    bn_output = (centered * inv + self.beta.astype(jnp.float32)) * mask

    return bn_output.astype(out_dtype)


class ConvolutionBlock(nn.Module):
  r"""Convolution block in conformer layer.

  architecture:

   input                   # (batch, time, hidden_dim)
     |
   layer_norm(.)           # (batch, time, hidden_dim)
   dense(.), dense(.)      # (batch, time, 2 * hidden_dim)
     |      /
    glu(.)                 # (batch, time, hidden_dim)
    depthwise_conv1d(.)
    batch_norm(.)
    act(.)
     |
    dense(.)
    dropout(.)
     |
    output
  """

  config: ConformerConfig

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config
    work_dtype = config.dtype if config.dtype is not None else inputs.dtype
    inputs = LayerNorm(
        dim=config.encoder_dim,
        param_dtype=config.param_dtype,
        name='LayerNorm_0',
    )(inputs.astype(work_dtype))

    input_gated1 = nn.Dense(
        config.encoder_dim,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        use_bias=True,
    )(inputs)

    input_gated2 = nn.Dense(
        config.encoder_dim,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        use_bias=True,
    )(inputs)

    inputs = input_gated1 * jax.nn.sigmoid(input_gated2)
    inputs = inputs * (1 - jnp.expand_dims(input_paddings, -1)).astype(
        work_dtype
    )

    inputs = nn.Conv(
        features=config.encoder_dim,
        kernel_size=(config.convolution_kernel_size,),
        strides=(1,),
        padding='SAME',
        feature_group_count=config.encoder_dim,
        use_bias=False,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        name='Conv_0',
    )(inputs)

    inputs = BatchNorm(config, name='BatchNorm_0')(
        inputs, input_paddings, train
    )

    inputs = model_utils.ACTIVATIONS[self.config.activation_function](inputs)
    inputs = nn.Dense(
        config.encoder_dim,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        name='Dense_2',
    )(inputs)

    inputs = nn.Dropout(
        rate=config.conv_residual_dropout_rate,
        deterministic=not train,
        name='Dropout_0',
    )(inputs)
    return inputs


class ConformerBlock(nn.Module):
  """Implements a single conformer encoder layer.

  High level overview:

    x = x + 0.5 * FeedForward(x)
    x = x + MHSA(x)
    x = x + ConvolutionBlock(x)
    x = x + 0.5 * FeedForward(x)

    y = layer_norm(x)
  """

  config: ConformerConfig

  @nn.compact
  def __call__(self, inputs, input_paddings, train, attention_mask=None):
    config = self.config
    work_dtype = config.dtype if config.dtype is not None else inputs.dtype
    inputs = inputs.astype(work_dtype)
    padding_mask = jnp.expand_dims(1 - input_paddings, -1).astype(work_dtype)

    inputs = inputs + 0.5 * FeedForwardModule(config=self.config)(
        inputs, padding_mask, train
    )

    inputs = inputs + MultiHeadedSelfAttention(config=self.config)(
        inputs, input_paddings, train, attention_mask=attention_mask
    )

    inputs = inputs + ConvolutionBlock(config)(inputs, input_paddings, train)

    inputs = inputs + 0.5 * FeedForwardModule(config=self.config)(
        inputs, padding_mask, train
    )

    if config.enable_conformer_post_layer_norm:
      inputs = LayerNorm(
          dim=config.encoder_dim, param_dtype=config.param_dtype
      )(inputs)

    return inputs


class ConformerEncoderDecoder(nn.Module):
  """Conformer (encoder + decoder) block.

  Takes audio input signals and outputs probability distribution over vocab size
  for each time step. The output is then fed into a CTC loss which eliminates
  the need for alignment with targets.
  """

  config: ConformerConfig

  def setup(self):
    config = self.config
    self.specaug = spectrum_augmenter.SpecAug(
        freq_mask_count=config.freq_mask_count,
        freq_mask_max_bins=config.freq_mask_max_bins,
        time_mask_count=config.time_mask_count,
        time_mask_max_frames=config.time_mask_max_frames,
        time_mask_max_ratio=config.time_mask_max_ratio,
        time_masks_per_frame=config.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=config.use_dynamic_time_mask_max_frames,
    )

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config

    outputs = inputs
    output_paddings = input_paddings

    # Compute normalized log mel spectrograms from input audio signal.
    preprocessing_config = preprocessor.LibrispeechPreprocessingConfig()
    outputs, output_paddings = preprocessor.MelFilterbankFrontend(
        preprocessing_config,
        per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
        per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR,
    )(outputs, output_paddings)

    # Ablate random parts of input along temporal and frequency dimension
    # following the specaug procedure in https://arxiv.org/abs/1904.08779.
    if config.use_specaug and train:
      outputs, output_paddings = self.specaug(outputs, output_paddings)

    work_dtype = config.dtype if config.dtype is not None else jnp.float32
    outputs = outputs.astype(work_dtype)

    # Subsample input by a factor of 4 by performing strided convolutions.
    outputs, output_paddings = Subsample(
        encoder_dim=config.encoder_dim,
        input_dropout_rate=config.input_dropout_rate,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
    )(outputs, output_paddings, train)

    # Precompute the boolean self-attention mask once across all encoder layers.
    if config.use_lingvo_attention:
      attention_mask = None
    else:
      valid = (1 - output_paddings) > 0
      attention_mask = jnp.logical_and(
          valid[:, None, :, None], valid[:, None, None, :]
      )

    # Run the conformer encoder layers.
    for _ in range(config.num_encoder_layers):
      outputs = ConformerBlock(config)(
          outputs, output_paddings, train, attention_mask=attention_mask
      )

    if config.enable_decoder_pre_layer_norm:
      outputs = LayerNorm(
          config.encoder_dim, param_dtype=config.param_dtype
      )(outputs)
    # Run the decoder which in this case is a trivial projection layer.
    outputs = nn.Dense(
        config.vocab_size,
        use_bias=True,
        dtype=work_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nn.initializers.xavier_uniform(),
    )(outputs)
    outputs = outputs.astype(jnp.float32)

    return outputs, output_paddings


class ConformerModel(base_model.BaseModel):
  """Conformer model that takes in log mel spectrograms as inputs.

  outputs probability distribution over vocab size for each time step.
  """

  # Adapted from lingvo's greedy decoding logic here:
  # https://github.com/tensorflow/lingvo/blob/2ee26814c57b7dcead3f0382170f2f3da006f810/lingvo/jax/layers/ctc_objectives.py#L138
  def sequence_mask(self, lengths, maxlen):
    batch_size = lengths.shape[0]
    a = jnp.ones([batch_size, maxlen])
    b = jnp.cumsum(a, axis=-1)
    c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(lengths.dtype)
    return c

  def collapse_and_remove_blanks(self, labels, seq_length, blank_id: int = 0):
    b, t = labels.shape
    # Zap out blank
    blank_mask = 1 - jnp.equal(labels, blank_id)
    labels = (labels * blank_mask).astype(labels.dtype)

    # Mask labels that don't equal previous label.
    label_mask = jnp.concatenate(
        [
            jnp.ones_like(labels[:, :1], dtype=jnp.int32),
            jnp.not_equal(labels[:, 1:], labels[:, :-1]),
        ],
        axis=1,
    )

    # Filter labels that aren't in the original sequence.
    maxlen = labels.shape[1]
    seq_mask = self.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = label_mask * seq_mask

    # remove repetitions from the labels
    ulabels = label_mask * labels

    # Count masks for new sequence lengths.
    label_mask = jnp.not_equal(ulabels, 0).astype(labels.dtype)
    new_seq_len = jnp.sum(label_mask, axis=1)

    # Mask indexes based on sequence length mask.
    new_maxlen = maxlen
    idx_mask = self.sequence_mask(new_seq_len, maxlen=new_maxlen)

    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = jnp.reshape(ulabels, [-1])
    flat_idx_mask = jnp.reshape(idx_mask, [-1])

    indices = jnp.nonzero(flat_idx_mask, size=b * t)[0]
    values = jnp.nonzero(flat_labels, size=b * t)[0]
    updates = jnp.take_along_axis(flat_labels, values, axis=-1)

    # Scatter to flat shape.
    flat = jnp.zeros(flat_idx_mask.shape).astype(labels.dtype)
    flat = flat.at[indices].set(updates)
    # 0'th position in the flat array gets clobbered by later padded updates,
    # so reset it here to its original value
    flat = flat.at[0].set(updates[0])

    # Reshape back to square batch.
    batch_size = labels.shape[0]
    new_shape = [batch_size, new_maxlen]
    return (
        jnp.reshape(flat, new_shape).astype(labels.dtype),
        new_seq_len.astype(seq_length.dtype),
    )

  def greedy_decode(self, logits, logit_paddings):
    per_frame_max = jnp.argmax(logits, axis=-1)
    seqlen = jnp.sum(1.0 - logit_paddings, axis=-1)
    hyp, _ = self.collapse_and_remove_blanks(per_frame_max, seqlen, blank_id=0)
    hyp_paddings = jnp.equal(hyp, 0).astype(jnp.int32)
    return hyp, hyp_paddings

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates cross_entopy on the given batch."""

    logits, logit_paddings = self.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        batch['inputs'],
        batch['input_paddings'],
        train=False,
        mutable=False,
    )

    labels = batch['targets']
    label_paddings = batch['target_paddings']

    objective_numerator, objective_denominator = self.loss_fn(
        logits, logit_paddings, labels, label_paddings
    )

    normalized_loss = objective_numerator / (objective_denominator)
    hyps, hyp_paddings = self.greedy_decode(logits, logit_paddings)

    return self.metrics_bundle.single_from_model_output(
        normalized_loss=normalized_loss,
        hyps=hyps,
        hyp_paddings=hyp_paddings,
        targets=labels,
        target_paddings=label_paddings,
    )

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    """Return CTC loss."""

    # For more information on flax.linen.Module.apply, see the docs at
    # https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply.
    (outputs, output_paddings), new_batch_stats = self.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        batch['inputs'],
        batch['input_paddings'],
        rngs={'dropout': dropout_rng},
        mutable=['batch_stats'],
        train=True,
    )

    labels = batch['targets']
    label_paddings = batch['target_paddings']

    objective_numerator, objective_denominator = self.loss_fn(
        outputs, output_paddings, labels, label_paddings
    )

    # epsilon added to handle empty batch case if we encounter one.
    objective_value = objective_numerator / (objective_denominator + 1e-9)
    return objective_value, new_batch_stats

  def apply_on_batch(self, params, batch_stats, batch, **apply_kwargs):
    """Wrapper around flax_module.apply."""
    if batch_stats is not None:
      variables = {'params': params, 'batch_stats': batch_stats}
    else:
      variables = {'params': params}

    return self.flax_module.apply(
        variables, batch['inputs'], batch['input_paddings'], **apply_kwargs
    )

  def build_flax_module(self):
    config = ConformerConfig(
        vocab_size=self.hps.output_shape[1],
        dtype=utils.dtype_from_str(self.hps.computation_dtype),
        param_dtype=utils.dtype_from_str(self.hps.model_dtype),
        encoder_dim=self.hps.encoder_dim,
        num_attention_heads=self.hps.num_attention_heads,
        num_encoder_layers=self.hps.num_encoder_layers,
        convolution_kernel_size=self.hps.convolution_kernel_size,
        freq_mask_count=self.hps.freq_mask_count,
        freq_mask_max_bins=self.hps.freq_mask_max_bins,
        time_mask_count=self.hps.time_mask_count,
        time_mask_max_frames=self.hps.time_mask_max_frames,
        time_mask_max_ratio=self.hps.time_mask_max_ratio,
        time_masks_per_frame=self.hps.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=self.hps.use_dynamic_time_mask_max_frames,
        use_specaug=self.hps.use_specaug,
        attention_residual_dropout_rate=self.hps.residual_dropout_rate,
        feed_forward_residual_dropout_rate=self.hps.residual_dropout_rate,
        input_dropout_rate=self.hps.input_dropout_rate,
        enable_conformer_post_layer_norm=self.hps.enable_conformer_post_layer_norm,
        enable_decoder_pre_layer_norm=self.hps.enable_decoder_pre_layer_norm,
        use_lingvo_attention=self.hps.use_lingvo_attention,
        activation_function=self.hps.activation_function,
    )
    module = ConformerEncoderDecoder(config)

    return module

  def get_fake_inputs(self, hps):
    """Helper method solely for purpose of initializing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *x), dtype=hps.model_dtype)
        for x in hps.input_shape
    ]
    return dummy_inputs


class MLCommonsConformerModel(ConformerModel):
  """Uses dropout_rate and aux_dropout_rate as hps.

  Dropouts are tied if tie_dropouts is True.
  Otherwise intended to be the same as ConformerModel.
  """

  def build_flax_module(self):

    aux_dropout_rate = (
        self.hps.dropout_rate
        if self.hps.tie_dropouts
        else self.hps.aux_dropout_rate
    )

    config = ConformerConfig(
        vocab_size=self.hps.output_shape[1],
        dtype=utils.dtype_from_str(self.hps.computation_dtype),
        param_dtype=utils.dtype_from_str(self.hps.model_dtype),
        encoder_dim=self.hps.encoder_dim,
        num_attention_heads=self.hps.num_attention_heads,
        num_encoder_layers=self.hps.num_encoder_layers,
        convolution_kernel_size=self.hps.convolution_kernel_size,
        freq_mask_count=self.hps.freq_mask_count,
        freq_mask_max_bins=self.hps.freq_mask_max_bins,
        time_mask_count=self.hps.time_mask_count,
        time_mask_max_frames=self.hps.time_mask_max_frames,
        time_mask_max_ratio=self.hps.time_mask_max_ratio,
        time_masks_per_frame=self.hps.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=self.hps.use_dynamic_time_mask_max_frames,
        use_specaug=self.hps.use_specaug,
        attention_residual_dropout_rate=self.hps.dropout_rate,
        feed_forward_residual_dropout_rate=self.hps.dropout_rate,
        input_dropout_rate=aux_dropout_rate,
        enable_conformer_post_layer_norm=self.hps.enable_conformer_post_layer_norm,
        enable_decoder_pre_layer_norm=self.hps.enable_decoder_pre_layer_norm,
        use_lingvo_attention=self.hps.use_lingvo_attention,
        attn_temperature=self.hps.attn_temperature,
        activation_function=self.hps.activation_function,
    )
    module = ConformerEncoderDecoder(config)

    return module
