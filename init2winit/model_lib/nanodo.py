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

ParameterType = model_utils.ParameterType
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        emb_dim=512,  # model/embed dim  = qkv dim
        num_heads=8,  # num attention heads
        num_layers=6,  # number of transformer block layers
        mlp_dim=2048,  # FF inner dimension
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
      1.0, 'fan_in', 'normal', out_axis=0)
  dtype: jnp.dtype = jnp.float32


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
    self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)

  def __call__(self, y_BxL: jax.Array, train: bool):
    del train
    # For training on concatenated examples.
    y_BxLxD = self.embed(y_BxL)
    y_BxLxD += self.pos_embed(jnp.arange(0, y_BxL.shape[1])[None, ...])
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD)
    y_BxLxD = self.out_ln(y_BxLxD)
    logits_BxLxV = self.embed.attend(y_BxLxD.astype(jnp.float32))
    return logits_BxLxV


class Mlp(nn.Module):
  """Multilayer perceptron."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    linear = functools.partial(
        nn.Dense, kernel_init=cfg.kernel_init, use_bias=False,
        dtype=cfg.dtype
    )
    x_BxLxF = linear(cfg.F)(x_BxLxD)
    x_BxLxF = jax.nn.gelu(x_BxLxF)
    x_BxLxD = linear(cfg.D)(x_BxLxF)
    return x_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""
  docfg: DoConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array):
    cfg = self.docfg

    # "pre-layernorm"
    x_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(in_BxLxD)
    x_BxLxD = CausalAttn(cfg)(x_BxLxD)
    x_BxLxD += in_BxLxD

    z_BxLxD = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)

    return x_BxLxD + z_BxLxD


class CausalAttn(nn.Module):
  """Causal attention layer."""
  cfg: DoConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg

    assert cfg.D % cfg.H == 0, f'D {cfg.D} not divisible by H {cfg.H}'
    Dh = cfg.D // cfg.H

    # Maps D -> (H, Dh)
    multilinear = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(cfg.H, Dh),
        kernel_init=cfg.kernel_init,
        use_bias=False,
        dtype=cfg.dtype,
    )

    q_BxLxHxDh, k_BxLxHxDh, v_BxLxHxDh = (
        multilinear(name='query')(x_BxLxD),
        multilinear(name='key')(x_BxLxD),
        multilinear(name='value')(x_BxLxD),
    )
    q_BxLxHxDh /= Dh**0.5
    att_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', q_BxLxHxDh, k_BxLxHxDh)
    # cast to fp32 for softmax
    att_BxHxLxL = att_BxHxLxL.astype(jnp.float32)

    # causal attention mask
    L = x_BxLxD.shape[1]
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

    _NEG_INF = jnp.finfo(cfg.dtype).min
    att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
    att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
    att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)
    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', att_BxHxLxL, v_BxLxHxDh)
    # Output projection followed by contraction back to original dims
    out_BxLxD = nn.DenseGeneral(
        features=cfg.D,
        name='attn_out_proj',
        axis=(-2, -1),
        kernel_init=cfg.kernel_init,
        use_bias=False,
        dtype=cfg.dtype,
    )(out_BxLxHxDh)
    return out_BxLxD


class NanodoModel(base_model.BaseModel):
  """Defines the model for the graph network."""

  def get_sharding_overrides(self, mesh):
    type_to_sharding = super().get_sharding_overrides(mesh)
    overrides = {
        ParameterType.EMBEDDING: NamedSharding(mesh, P(None, 'devices')),
        ParameterType.WEIGHT: NamedSharding(mesh, P('devices', None)),
        ParameterType.ATTENTION_K: NamedSharding(
            mesh, P('devices', None, None)
        ),
        ParameterType.ATTENTION_Q: NamedSharding(
            mesh, P('devices', None, None)
        ),
        ParameterType.ATTENTION_V: NamedSharding(
            mesh, P('devices', None, None)
        ),
        ParameterType.ATTENTION_OUT: NamedSharding(
            mesh, P(None, None, 'devices')
        ),
    }

    type_to_sharding.update(overrides)
    return type_to_sharding

  def build_flax_module(self):
    config = DoConfig(
        D=self.hps['emb_dim'],
        H=self.hps['num_heads'],
        N=self.hps['num_layers'],
        V=self.hps['vocab_size'],
        L=self.hps['input_shape'][0],
        F=self.hps['mlp_dim'],
        dtype=utils.dtype_from_str(self.hps['computation_dtype']),
    )
    return TransformerDo(config)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=jnp.int32)
    ]
    return dummy_inputs
