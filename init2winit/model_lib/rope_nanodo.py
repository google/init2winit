# coding=utf-8
# Copyright 2025 The init2winit Authors.
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
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

partial = functools.partial
ParameterType = model_utils.ParameterType
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        emb_dim=1024,  # model/embed dim  = qkv dim
        num_heads=8,  # num attention heads
        num_layers=12,  # number of transformer block layers
        mlp_dim=1024,  # FF inner dimension
        rng_seed=-1,
        computation_dtype='bfloat16',
        model_dtype='float32',
        optimizer='adam',
        batch_size=256,
        lr_hparams={'base_lr': 0.01, 'schedule': 'constant'},
        opt_hparams={
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0,
        },
        l2_decay_factor=0.0005,
        l2_decay_rank_threshold=2,
        grad_clip=None,
        label_smoothing=0.0,
        use_shallue_label_smoothing=False,
        normalization='layernorm',
        mlp_activation='glu',
    )
)


@dataclasses.dataclass
class DoConfig:
  """Hyper-parameters for Transformer decoder-only."""

  D: int  # model/embed dim  = qkv dim
  H: int  # num attention heads
  N: int  # number of transformer block layers
  V: int  # vocab size
  F: int  # FF inner dimension
  L: int  # sequence length
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  embed_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal', out_axis=0
  )
  dtype: jnp.dtype = jnp.float32
  rmsnorm_epsilon: float = 1e-6
  multiple_of: int = 256
  tie_embeddings: bool = True  # Whether to tie input and output embeddings
  mlp_activation: str = 'glu'
  normalization: str = 'layernorm'


class Mlp(nn.Module):
  """Multilayer perceptron with GLU activation."""

  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    # Use Xavier uniform initialization explicitly
    linear = partial(
        nn.Dense, kernel_init=cfg.kernel_init, use_bias=False, dtype=cfg.dtype
    )
    if cfg.mlp_activation == 'glu':
      mlp_activation = nn.glu
      x_BxLxF = linear(cfg.F)(x_BxLxD)
    elif cfg.mlp_activation == 'gelu':
      mlp_activation = nn.gelu
      hidden_dim = cfg.multiple_of * (
          (cfg.F + cfg.multiple_of - 1) // cfg.multiple_of
      )
      # Double the hidden dimension for GLU
      x_BxLxF = linear(2 * hidden_dim)(x_BxLxD)
    else:
      raise ValueError(f'Unknown activation: {cfg.mlp_activation}')
    x_BxLxF = mlp_activation(x_BxLxF)
    x_BxLxD = linear(cfg.D)(x_BxLxF)
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

  def rotate_tensor(x):
    # Split into real and imaginary parts
    x_r2 = x.reshape(*x.shape[:-1], -1, 2)
    L = x.shape[1]
    freqs = freqs_cis[:, :L, :, :, :]

    # Apply rotation
    rotated_x_r2 = jnp.stack(
        [
            x_r2[..., 0] * freqs[..., 0] - x_r2[..., 1] * freqs[..., 1],
            x_r2[..., 1] * freqs[..., 0] + x_r2[..., 0] * freqs[..., 1],
        ],
        axis=-1,
    )

    return rotated_x_r2.reshape(*x.shape)

  # Apply rotation to Q and K separately
  rotated_q = rotate_tensor(q)
  rotated_k = rotate_tensor(k)

  return rotated_q, rotated_k


class CausalAttn(nn.Module):
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
        kernel_init=cfg.kernel_init,
        use_bias=False,
        dtype=cfg.dtype,
    )

    self.multilinear_query = self.multilinear(name='query')
    self.multilinear_key = self.multilinear(name='key')
    self.multilinear_value = self.multilinear(name='value')
    self.output_projection = nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        # axis=(-2, -1),      #
        kernel_init=cfg.kernel_init,
        use_bias=False,
        dtype=cfg.dtype,
    )

  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg

    # Project inputs to Q, K, V
    q_BxLxHxDh = self.multilinear_query(x_BxLxD)
    k_BxLxHxDh = self.multilinear_key(x_BxLxD)
    v_BxLxHxDh = self.multilinear_value(x_BxLxD)

    # Apply rotary embeddings to Q and K
    q_BxLxHxDh, k_BxLxHxDh = apply_rope(q_BxLxHxDh, k_BxLxHxDh, self.freqs_cis)

    # Scale queries
    q_BxLxHxDh /= self.Dh**0.5

    # Compute attention scores
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
    # TODO(kasimbeg): Remove this.
    # # cast to fp32 for softmax
    # att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)

    # Causal attention mask
    L = x_BxLxD.shape[1]
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    # Apply mask and softmax
    _NEG_INF = jnp.finfo(cfg.dtype).min
    att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)

    # Compute attention output
    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)

    # Reshape and project output
    out_BxLxD = out_BxLxHxDh.reshape(*x_BxLxD.shape)

    # Output projection
    out_BxLxD = self.output_projection(out_BxLxD)

    return out_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""

  docfg: DoConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array):
    cfg = self.docfg

    # "pre-layernorm"
    if cfg.normalization == 'layernorm':
      x_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(in_BxLxD)
    else:
      raise ValueError(f'Unknown normalization: {cfg.normalization}')

    x_BxLxD = CausalAttn(cfg)(x_BxLxD)
    x_BxLxD += in_BxLxD

    z_BxLxD = Mlp(cfg)(x_BxLxD)

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
    )
    self.pos_embed = nn.Embed(
        num_embeddings=cfg.L,
        features=cfg.D,
        embedding_init=cfg.embed_init,
    )

    self.blocks = [TBlock(cfg) for _ in range(cfg.N)]
    if cfg.normalization == 'layernorm':
      self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)
    elif cfg.normalization == 'rmsnorm':
      self.out_ln = nn.RMSNorm(dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)
    else:
      raise ValueError(f'Unknown normalization: {cfg.normalization}')

    # Output projection - tied to input embeddings if configured
    if cfg.tie_embeddings:
      self.output_proj = lambda x: self.embed.attend(x.astype(jnp.float32))
    else:
      self.output_proj = nn.Dense(
          cfg.V, kernel_init=cfg.embed_init, dtype=cfg.dtype, name='output_proj'
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
        mlp_activation=self.hps['mlp_activation'],
        normalization=self.hps['normalization'],
    )
    return TransformerDo(config)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=jnp.int32)
    ]
    return dummy_inputs
