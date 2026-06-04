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

"""Nanodo decoder-only model for next-token prediction task.

Implementation from
https://github.com/google-deepmind/nanodo/blob/main/nanodo/model.py
"""

# pylint: disable=invalid-name
import dataclasses
import functools

from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import base_model
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

partial = functools.partial


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        emb_dim=512,  # model/embed dim  = qkv dim
        num_heads=8,  # num attention heads
        num_layers=12,  # number of transformer block layers
        mlp_dim=2048,  # FF inner dimension
        computation_dtype='bfloat16',
        model_dtype='float32',
        normalization='rmsnorm',
        mlp_activation='glu',
        qk_norm=True,
        tie_embeddings=True,
        use_residual_scaling=False,
        initializer='xavier',
    )
)


@dataclasses.dataclass
class DoConfig:
  """Hyper-parameters for Transformer decoder-only."""

  D: int  # embed dim
  H: int  # num attention heads
  N: int  # number of transformer block layers
  V: int  # vocab size
  F: int  # FF inner dimension
  L: int  # sequence length
  dtype: jnp.dtype = jnp.bfloat16
  param_dtype: jnp.dtype = jnp.float32
  rmsnorm_epsilon: float = 1e-6
  multiple_of: int = 256
  tie_embeddings: bool = True  # Whether to tie input and output embeddings
  mlp_activation: str = 'glu'
  normalization: str = 'rmsnorm'
  qk_norm: bool = True
  is_causal: bool = True
  eps: float = 1e-6
  use_residual_scaling: bool = False
  initializer: str = 'xavier'  # 'xavier' or 'std0.02'

  # Derived initializers (set in __post_init__)
  attention_init: nn.initializers.Initializer = dataclasses.field(init=False)
  linear_init: nn.initializers.Initializer = dataclasses.field(init=False)
  embed_init: nn.initializers.Initializer = dataclasses.field(init=False)
  residual_init: nn.initializers.Initializer = dataclasses.field(init=False)

  def __post_init__(self):
    if self.initializer == 'xavier':
      self.attention_init = nn.initializers.xavier_uniform()
      self.linear_init = nn.initializers.xavier_uniform()
      self.embed_init = nn.initializers.variance_scaling(
          1.0, 'fan_in', 'normal', out_axis=0
      )
    elif self.initializer == 'std0.02':
      self.attention_init = nn.initializers.normal(stddev=0.02)
      self.linear_init = nn.initializers.normal(stddev=0.02)
      self.embed_init = nn.initializers.normal(stddev=0.02)
    else:
      raise ValueError(f'Unknown initializer: {self.initializer}')

    if self.use_residual_scaling and self.initializer == 'std0.02':
      self.residual_init = nn.initializers.normal(
          stddev=0.02 / jnp.sqrt(2 * self.N)
      )
    else:
      self.residual_init = self.linear_init

  def make_norm(self):
    """Returns a normalization layer based on config."""
    if self.normalization == 'layernorm':
      return nn.LayerNorm(
          dtype=self.dtype, param_dtype=self.param_dtype, use_bias=False
      )
    elif self.normalization == 'rmsnorm':
      return nn.RMSNorm(
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          epsilon=self.rmsnorm_epsilon,
      )
    else:
      raise ValueError(f'Unknown normalization: {self.normalization}')


class Mlp(nn.Module):
  """Multilayer perceptron with GLU activation."""

  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    linear = partial(
        nn.Dense,
        kernel_init=cfg.linear_init,
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    if cfg.mlp_activation == 'gelu':
      mlp_activation = nn.gelu
      x_BxLxF = linear(cfg.F)(x_BxLxD)
    elif cfg.mlp_activation == 'glu':
      mlp_activation = nn.glu
      #  Adjust hidden dimension to keep the number of parameters invariant to
      # the activation function used since the GLU MLP has 3 * hidden_dim * D
      # parameters instead of 2 * hidden_dim * D parameters.
      hidden_dim = cfg.F * 2 / 3
      # Round up to the nearest multiple of cfg.multiple_of
      hidden_dim = int(cfg.multiple_of * (
          (hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of
      ))
      # Double the hidden dimension for GLU
      x_BxLxF = linear(2 * hidden_dim)(x_BxLxD)
    else:
      raise ValueError(f'Unknown activation: {cfg.mlp_activation}')
    x_BxLxF = mlp_activation(x_BxLxF)
    x_BxLxD = nn.Dense(
        cfg.D,
        kernel_init=cfg.residual_init
        if cfg.use_residual_scaling
        else cfg.linear_init,
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )(x_BxLxF)
    return x_BxLxD


@partial(jax.jit, static_argnums=(0, 1, 2))
def init_rope(dim=256, seq_len=128, n_heads=4):
  """Initialize rotary embeddings."""

  def precompute_freqs_cis_jax(dim, end, theta=10000.0):
    inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(end) / 1.0
    freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)
    return jnp.stack(
        [jnp.cos(freqs)[None, :, None, :], jnp.sin(freqs)[None, :, None, :]],
        axis=3,
    )

  freqs_cis = precompute_freqs_cis_jax(dim // n_heads, seq_len, theta=500000)
  return freqs_cis.transpose(0, 1, 2, 4, 3)


@jax.jit
def apply_rope(q, k, freqs_cis):
  """Apply rotary embeddings to Q and K."""
  input_dtype = q.dtype

  def rotate_tensor(x):
    x_r2 = x.reshape(*x.shape[:-1], -1, 2)
    L = x.shape[1]
    freqs = freqs_cis[:, :L, :, :, :]

    rotated_x_r2 = jnp.stack(
        [
            x_r2[..., 0] * freqs[..., 0] - x_r2[..., 1] * freqs[..., 1],
            x_r2[..., 1] * freqs[..., 0] + x_r2[..., 0] * freqs[..., 1],
        ],
        axis=-1,
    )

    return rotated_x_r2.reshape(*x.shape)

  rotated_q = rotate_tensor(q).astype(input_dtype)
  rotated_k = rotate_tensor(k).astype(input_dtype)

  return rotated_q, rotated_k


class Attention(nn.Module):
  """Causal attention layer with rotary embeddings."""

  cfg: DoConfig

  def setup(self):
    cfg = self.cfg

    assert cfg.D % cfg.H == 0, f'D {cfg.D} not divisible by H {cfg.H}'
    self.Dh = cfg.D // cfg.H

    # Initialize rotary embeddings
    self.freqs_cis = init_rope(cfg.D, cfg.L, cfg.H)

    # Maps D -> (H, Dh)
    self.multilinear = partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, self.Dh),
        kernel_init=cfg.attention_init,
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )

    self.multilinear_query = self.multilinear(name='query')
    self.multilinear_key = self.multilinear(name='key')
    self.multilinear_value = self.multilinear(name='value')
    self.output_projection = nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        # axis=(-2, -1),      #
        kernel_init=cfg.residual_init
        if cfg.use_residual_scaling
        else cfg.linear_init,
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    if cfg.qk_norm:
      self.eps = cfg.eps
    attn_scale0 = jnp.log2(cfg.L**2 - cfg.L).astype(cfg.param_dtype)
    self.attn_scale = self.param(
        'attn_scale',
        nn.initializers.constant(attn_scale0, dtype=cfg.param_dtype),
        (),
    )

  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg

    q_BxLxHxDh = self.multilinear_query(x_BxLxD)
    k_BxLxHxDh = self.multilinear_key(x_BxLxD)
    v_BxLxHxDh = self.multilinear_value(x_BxLxD)

    if cfg.qk_norm:
      q_BxLxHxDh /= (
          jnp.linalg.norm(q_BxLxHxDh, axis=-1, keepdims=True) + self.eps
      )
      k_BxLxHxDh /= (
          jnp.linalg.norm(k_BxLxHxDh, axis=-1, keepdims=True) + self.eps
      )
    q_BxLxHxDh = q_BxLxHxDh * self.attn_scale.astype(cfg.dtype)

    q_BxLxHxDh, k_BxLxHxDh = apply_rope(q_BxLxHxDh, k_BxLxHxDh, self.freqs_cis)

    out_BxLxHxDh = jax.nn.dot_product_attention(
        q_BxLxHxDh.astype(cfg.dtype),
        k_BxLxHxDh.astype(cfg.dtype),
        v_BxLxHxDh.astype(cfg.dtype),
        is_causal=cfg.is_causal,
        scale=1.0,
    )

    out_BxLxD = out_BxLxHxDh.reshape(*x_BxLxD.shape)
    out_BxLxD = self.output_projection(out_BxLxD)

    return out_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""

  docfg: DoConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array):
    cfg = self.docfg

    x_BxLxD = cfg.make_norm()(in_BxLxD)

    x_BxLxD = Attention(cfg)(x_BxLxD)
    x_BxLxD += in_BxLxD

    z_BxLxD = cfg.make_norm()(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)

    return x_BxLxD + z_BxLxD


class TransformerDo(nn.Module):
  """Transformer decoder-only."""

  docfg: DoConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V,
        features=cfg.D,
        embedding_init=cfg.embed_init,
        dtype=cfg.dtype,
        param_dtype=cfg.param_dtype,
    )
    self.blocks = [TBlock(cfg) for _ in range(cfg.N)]
    self.out_ln = cfg.make_norm()

    # Output projection - tied to input embeddings if configured
    if cfg.tie_embeddings:
      self.output_proj = lambda x: self.embed.attend(x.astype(jnp.float32))
    else:
      self.output_proj = nn.Dense(
          cfg.V,
          kernel_init=cfg.embed_init,
          dtype=cfg.dtype,
          param_dtype=cfg.param_dtype,
          name='output_proj',
      )

  def __call__(self, y_BxL: jax.Array, train: bool):
    del train
    # For training on concatenated examples.
    y_BxLxD = self.embed(y_BxL)
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD)
    y_BxLxD = self.out_ln(y_BxLxD)
    logits_BxLxV = self.output_proj(y_BxLxD)
    return logits_BxLxV


class RoPENanodoModel(base_model.BaseModel):
  """Defines the model."""

  def build_flax_module(self):
    config = DoConfig(
        D=self.hps['emb_dim'],
        H=self.hps['num_heads'],
        N=self.hps['num_layers'],
        V=self.hps['vocab_size'],
        L=self.hps['input_shape'][0],
        F=self.hps['mlp_dim'],
        dtype=utils.dtype_from_str(self.hps['computation_dtype']),
        param_dtype=utils.dtype_from_str(self.hps['model_dtype']),
        mlp_activation=self.hps['mlp_activation'],
        normalization=self.hps['normalization'],
        qk_norm=self.hps['qk_norm'],
        tie_embeddings=self.hps['tie_embeddings'],
        use_residual_scaling=self.hps['use_residual_scaling'],
        initializer=self.hps['initializer'],
    )
    return TransformerDo(config)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=jnp.int32)
    ]
    return dummy_inputs
