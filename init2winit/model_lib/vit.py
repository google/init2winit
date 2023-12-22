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

"""A refactored and simplified ViT.

NOTE: Taken from https://github.com/google/big_vision with modifications noted.
"""

import functools
from typing import Optional, Sequence, Union

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


# NOTE(dsuo): could be useful to have a `base_config` for models as well.
DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        num_classes=1000,
        variant=None,
        width=384,
        depth=12,
        mlp_dim=1536,
        num_heads=6,
        patch_size=(16, 16),
        rep_size=True,
        pool_type='gap',
        posemb='sincos2d',
        head_zeroinit=True,
        lr_hparams={
            'base_lr': 1e-3,
            'schedule': 'cosine_warmup',
            'warmup_steps': 10_000
        },
        optimizer='adam',
        opt_hparams={
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 1e-1,
        },
        l2_decay_factor=None,
        l2_decay_rank_threshold=2,
        batch_size=1024,
        rng_seed=-1,
        model_dtype='float32',
        grad_clip=None,
        total_accumulated_batch_size=None,
        dropout_rate=0.0,
        label_smoothing=0.0,
        use_shallue_label_smoothing=False,
        normalizer='pre_layer_norm',
        activation='gelu',
        resnet_style_residual=False,
        residual_alpha=0.5,
        scale_attention_init=1.0,
        layer_norm_struct=None,
        attn_temperature=1.0,
        use_glu=False,
    ))


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  if width % 4 != 0:
    raise ValueError('Width must be mult of 4 for sincos posemb.')
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature ** omega)
  y = jnp.einsum('m,d->md', y.flatten(), omega)
  x = jnp.einsum('m,d->md', x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, emb_type, seqshape, width, name, dtype=jnp.float32):
  if emb_type == 'learn':
    return self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif emb_type == 'sincos2d':
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f'Unknown posemb type: {emb_type}')


def dot_product_attention(query,
                          key,
                          value,
                          bias=None,
                          mask=None,
                          broadcast_dropout=True,
                          dropout_rng=None,
                          dropout_rate=0.,
                          dtype=jnp.float32,
                          precision=None,
                          temperature=1.0):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    temperature: Constant factor to multiply logits by before computing softmax.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = nn.dot_product_attention_weights(
      query,
      key,
      bias,
      mask,
      broadcast_dropout,
      dropout_rng,
      dropout_rate,
      dtype,
      precision,
  )

  # return weighted sum over values for each query position
  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value,
                    precision=precision) * temperature


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  activation: str = 'gelu'
  use_glu: bool = False

  @nn.compact
  def __call__(self, x, train=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[2]
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    if self.activation in model_utils.ACTIVATIONS:
      x = model_utils.ACTIVATIONS[self.activation](x)
    else:
      raise ValueError('Unsupported activation: {}'.format(self.activation))

    if self.use_glu:
      y = nn.Dense(
          self.mlp_dim,
          **inits)(x)
      x = x * y

    x = nn.Dropout(rate=self.dropout)(x, train)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  normalizer: str = 'pre_layer_norm'
  activation: str = 'gelu'
  resnet_style_residual: bool = False
  residual_alpha: float = 0.5
  scale_attention_init: float = 1.0
  attn_temperature: float = 1.0
  use_glu: bool = False

  @nn.compact
  def __call__(self, x, train=True):
    out = {}
    if self.normalizer == 'pre_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=None
      )
      maybe_post_normalize = model_utils.get_normalizer(
          'none', train, dtype=None
      )
    elif self.normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer(
          'none', train, dtype=None
      )
      maybe_post_normalize = model_utils.get_normalizer(
          self.normalizer, train, dtype=None
      )
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))

    y = maybe_pre_normalize()(x)
    attn_fn = functools.partial(
        dot_product_attention, temperature=self.attn_temperature
    )
    y = out['sa'] = nn.SelfAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.variance_scaling(
            self.scale_attention_init,
            'fan_avg',
            'uniform',
            in_axis=-2,
            out_axis=-1,
            batch_axis=(),
            dtype=jnp.float_),
        deterministic=train,
        attention_fn=attn_fn,
        name='MultiHeadDotProductAttention_1',
    )(y)
    y = nn.Dropout(rate=self.dropout)(y, train)
    x = 2 * (self.residual_alpha * x + (1-self.residual_alpha) * y)
    x = maybe_post_normalize()(x)
    out['+sa'] = x

    y = maybe_pre_normalize()(x)
    y = out['mlp'] = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        name='MlpBlock_3',
        activation=self.activation,
        use_glu=self.use_glu
    )(y, train)
    y = nn.Dropout(rate=self.dropout)(y, train)
    x = 2*(self.residual_alpha*x+(1-self.residual_alpha)*y)
    x = maybe_post_normalize()(x)
    if self.resnet_style_residual:
      activation_fn = model_utils.ACTIVATIONS[self.activation]
      x = activation_fn(x)
    out['+mlp'] = x

    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  normalizer: str = 'pre_layer_norm'
  activation: str = 'gelu'
  resnet_style_residual: bool = False
  residual_alpha: float = 0.5
  scale_attention_init: float = 1.0
  layer_norm_struct: Sequence[str] = None
  attn_temperature: float = 1.0
  use_glu: bool = False

  @nn.compact
  def __call__(self, x, train=True):
    out = {}

    # Input Encoder
    for lyr in range(self.depth):
      if not self.layer_norm_struct:
        block = Encoder1DBlock(
            name=f'encoderblock_{lyr}',
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            normalizer=self.normalizer,
            activation=self.activation,
            resnet_style_residual=self.resnet_style_residual,
            residual_alpha=self.residual_alpha,
            scale_attention_init=self.scale_attention_init,
            attn_temperature=self.attn_temperature,
            use_glu=self.use_glu)
      else:
        assert(len(self.layer_norm_struct)) == self.depth
        block = Encoder1DBlock(
            name=f'encoderblock_{lyr}',
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            normalizer=self.layer_norm_struct[lyr],
            activation=self.activation,
            resnet_style_residual=self.resnet_style_residual,
            residual_alpha=self.residual_alpha,
            scale_attention_init=self.scale_attention_init,
            attn_temperature=self.attn_temperature,
            use_glu=self.use_glu)
      x, out[f'block{lyr:02d}'] = block(x, train)
    out['pre_ln'] = x  # Alias for last block, but without the number in it.

    if self.normalizer == 'pre_layer_norm':
      return nn.LayerNorm(name='encoder_norm')(x), out
    else:
      return x, out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  normalizer: str = 'pre_layer_norm'
  @nn.compact
  def __call__(self, x):
    # TODO(lbeyer): condition on GAP(x)
    n, _, d = x.shape
    probe = self.param('probe', nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO(lbeyer): dropout on head?
    if self.normalizer == 'pre_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer(
          self.normalizer, train=True, dtype=None)
      maybe_post_normalize = model_utils.get_normalizer(
          'none', train=True, dtype=None)
    elif self.normalizer == 'post_layer_norm':
      maybe_pre_normalize = model_utils.get_normalizer(
          'none', train=True, dtype=None)
      maybe_post_normalize = model_utils.get_normalizer(
          self.normalizer, train=True, dtype=None)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))
    y = maybe_pre_normalize()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    x = maybe_post_normalize()(x)
    return x[:, 0]


class ViT(nn.Module):
  """ViT model."""

  num_classes: int
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = 'sincos2d'  # Can also be "learn"
  rep_size: Union[int, bool] = False
  dropout_rate: float = 0.0
  pool_type: str = 'gap'  # Can also be 'map' or 'tok'
  reinit: Optional[Sequence[str]] = None
  head_zeroinit: bool = True
  normalizer: str = 'pre_layer_norm'
  activation: str = 'gelu'
  resnet_style_residual: bool = False
  residual_alpha: float = 0.5
  scale_attention_init: float = 1.0
  layer_norm_struct: Sequence[str] = None
  attn_temperature: float = 1.0
  use_glu: bool = False
  @nn.compact
  def __call__(self, x, *, train=False):
    out = {}

    # Patch extraction
    x = out['stem'] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding='VALID', name='conv_patch_extract')(x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out['with_posemb'] = x + get_posemb(
        self, self.posemb, (h, w), c, 'pos_embedding', x.dtype)

    if self.pool_type == 'tok':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout_rate)(x, not train)

    x, out['encoder'] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout_rate,
        normalizer=self.normalizer,
        activation=self.activation,
        resnet_style_residual=self.resnet_style_residual,
        residual_alpha=self.residual_alpha,
        scale_attention_init=self.scale_attention_init,
        layer_norm_struct=self.layer_norm_struct,
        attn_temperature=self.attn_temperature,
        use_glu=self.use_glu,
        name='Transformer')(
            x, train=not train)
    encoded = out['encoded'] = x

    if self.pool_type == 'map':
      x = out['head_input'] = MAPHead(
          num_heads=self.num_heads,
          mlp_dim=self.mlp_dim,
          normalizer=self.normalizer)(
              x)
    elif self.pool_type == 'gap':
      x = out['head_input'] = jnp.mean(x, axis=1)
    elif self.pool_type == '0':
      x = out['head_input'] = x[:, 0]
    elif self.pool_type == 'tok':
      x = out['head_input'] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f'Unknown pool type: "{self.pool_type}"')

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      hid = nn.Dense(rep_size, name='pre_logits')
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out['pre_logits_2d'] = x_2d
    out['pre_logits'] = x

    if self.num_classes:
      kw = {'kernel_init': nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name='head', **kw)
      x_2d = out['logits_2d'] = head(x_2d)
      x = out['logits'] = head(x)

    # TODO(dsuo): this used to be `return x, out`. Do we need out?
    return x


class ViTModel(base_model.BaseModel):
  """ViT model."""

  def build_flax_module(self):
    """Vision transformer."""

    keys = [
        'num_classes', 'rep_size', 'pool_type', 'posemb', 'width', 'depth',
        'mlp_dim', 'num_heads', 'patch_size', 'dropout_rate', 'head_zeroinit',
        'normalizer', 'activation', 'resnet_style_residual', 'residual_alpha',
        'scale_attention_init', 'layer_norm_struct', 'attn_temperature',
        'use_glu'
    ]

    args = {k: self.hps[k] for k in keys}

    if self.hps.variant is not None:
      args.update(decode_variant(self.hps.variant))
    return ViT(**args)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs


def decode_variant(variant):
  """Converts a string like 'B/32' into a params dict.

  NOTE(dsuo): modified to expect variant + patch size only.

  Args:
    variant: a string of `model_size`/`patch_size`.

  Returns:
    dict: an expanded dictionary of model hps.
  """
  v, patch = variant.split('/')

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      'width': {'Ti': 192, 'S': 384, 'V': 384, 'M': 512, 'B': 768, 'L': 1024, 'H': 1280, 'g': 1408, 'G': 1664}[v],
      'depth': {'Ti': 12, 'S': 12, 'V': 12, 'M': 12, 'B': 12, 'L': 24, 'H': 32, 'g': 40, 'G': 48}[v],
      'mlp_dim': {'Ti': 768, 'S': 1536, 'V': 1152, 'M': 2048, 'B': 3072, 'L': 4096, 'H': 5120, 'g': 6144, 'G': 8192}[v],
      'num_heads': {'Ti': 3, 'S': 6, 'V': 6, 'M': 8, 'B': 12, 'L': 16, 'H': 16, 'g': 16, 'G': 16}[v],
      # pylint:enable=line-too-long
      'patch_size': (int(patch), int(patch))
  }
