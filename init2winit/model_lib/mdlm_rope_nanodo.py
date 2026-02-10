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

"""MDLM (Masked Diffusion Language Model) with RoPE transformer.

Bidirectional transformer for masked diffusion, reusing building blocks from
rope_nanodo.py. Implements the MDLM training objective from
"Simple and Effective Masked Diffusion Language Models" (Sahoo et al., 2024).
"""

# pylint: disable=invalid-name
from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import base_model
from init2winit.model_lib import mdlm_schedules
from init2winit.model_lib import model_utils
from init2winit.model_lib import rope_nanodo
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        emb_dim=512,
        num_heads=8,
        num_layers=12,
        mlp_dim=2048,
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
        normalization='rmsnorm',
        mlp_activation='glu',
        qk_norm=True,
        tie_embeddings=True,
        noise_schedule='log_linear',
        epsilon=1e-7,
    )
)


class TimestepEmbedding(nn.Module):
  """Embeds scalar timestep into D-dimensional vector via MLP."""

  D: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, t_B: jax.Array):
    half_d = self.D // 2
    freq = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half_d, dtype=jnp.float32) / half_d
    )
    angles = t_B[:, None] * freq[None, :]
    sincos = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    h = nn.Dense(self.D, dtype=self.dtype)(sincos)
    h = nn.gelu(h)
    h = nn.Dense(self.D, dtype=self.dtype)(h)
    return h


class MDLMTransformer(nn.Module):
  """Bidirectional transformer for MDLM."""

  docfg: rope_nanodo.DoConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
        num_embeddings=cfg.V + 1,
        features=cfg.D,
        embedding_init=cfg.embed_init,
    )

    self.time_embed = TimestepEmbedding(D=cfg.D, dtype=cfg.dtype)

    self.blocks = [rope_nanodo.TBlock(cfg) for _ in range(cfg.N)]
    if cfg.normalization == 'layernorm':
      self.out_ln = nn.LayerNorm(dtype=cfg.dtype, use_bias=False)
    elif cfg.normalization == 'rmsnorm':
      self.out_ln = nn.RMSNorm(dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)
    else:
      raise ValueError(f'Unknown normalization: {cfg.normalization}')

    if cfg.tie_embeddings:
      self.output_proj = None
    else:
      self.output_proj = nn.Dense(
          cfg.V, kernel_init=cfg.embed_init, dtype=cfg.dtype, name='output_proj'
      )

  def __call__(self, z_BxL: jax.Array, t_B: jax.Array, train: bool):
    del train
    cfg = self.docfg

    z_BxLxD = self.embed(z_BxL)

    t_BxD = self.time_embed(t_B)
    z_BxLxD = z_BxLxD + t_BxD[:, None, :]

    for block in self.blocks:
      z_BxLxD = block(z_BxLxD)
    z_BxLxD = self.out_ln(z_BxLxD)

    if self.output_proj is not None:
      logits_BxLxV = self.output_proj(z_BxLxD)
    else:
      embed_matrix = self.embed.embedding[: cfg.V]
      logits_BxLxV = z_BxLxD.astype(jnp.float32) @ embed_matrix.T

    return logits_BxLxV


class MDLMModel(base_model.BaseModel):
  """MDLM model with diffusion training objective."""

  def build_flax_module(self):
    config = rope_nanodo.DoConfig(
        D=self.hps['emb_dim'],
        H=self.hps['num_heads'],
        N=self.hps['num_layers'],
        V=self.hps['vocab_size'],
        L=self.hps['input_shape'][0],
        F=self.hps['mlp_dim'],
        dtype=utils.dtype_from_str(self.hps['computation_dtype']),
        mlp_activation=self.hps['mlp_activation'],
        normalization=self.hps['normalization'],
        qk_norm=self.hps['qk_norm'],
        tie_embeddings=self.hps['tie_embeddings'],
        is_causal=False,
    )
    return MDLMTransformer(config)

  def get_fake_inputs(self, hps):
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=jnp.int32),
        jnp.zeros((hps.batch_size,), dtype=jnp.float32),  # t_B timestep
    ]
    return dummy_inputs

  def _mask_and_forward(self, params, batch, rng, train):
    vocab_size = self.hps['vocab_size']
    mask_id = vocab_size

    get_alpha, _ = mdlm_schedules.get_schedule(self.hps['noise_schedule'])

    rng_t, rng_mask = jax.random.split(rng)

    x_BxL = batch['inputs']
    B = x_BxL.shape[0]

    t_B = jax.random.uniform(rng_t, shape=(B,), minval=1e-5, maxval=1.0)

    alpha_t_B = get_alpha(t_B)

    mask_prob_BxL = jnp.broadcast_to((1.0 - alpha_t_B)[:, None], x_BxL.shape)
    mask_draws_BxL = jax.random.uniform(rng_mask, shape=x_BxL.shape)
    is_masked_BxL = mask_draws_BxL < mask_prob_BxL

    z_BxL = jnp.where(is_masked_BxL, mask_id, x_BxL)

    variables = {'params': params}
    logits_BxLxV = self.flax_module.apply(variables, z_BxL, t_B, train=train)

    _NEG_INF = jnp.finfo(logits_BxLxV.dtype).min
    logits_BxLxV = jnp.where(
        is_masked_BxL[:, :, None],
        logits_BxLxV,
        _NEG_INF,
    )
    unmasked_targets = jax.nn.one_hot(x_BxL, vocab_size)
    logits_BxLxV = jnp.where(
        is_masked_BxL[:, :, None],
        logits_BxLxV,
        unmasked_targets * 1e6,
    )

    return logits_BxLxV, z_BxL, is_masked_BxL, t_B, alpha_t_B

  def _compute_elbo(self, params, batch, rng, train):
    logits_BxLxV, _, is_masked_BxL, t_B, alpha_t_B = self._mask_and_forward(
        params, batch, rng, train
    )

    _, get_alpha_deriv = mdlm_schedules.get_schedule(self.hps['noise_schedule'])

    x_BxL = batch['inputs']
    B = x_BxL.shape[0]

    log_probs_BxLxV = jax.nn.log_softmax(logits_BxLxV, axis=-1)

    targets_BxL = x_BxL
    log_prob_true_BxL = log_probs_BxLxV[
        jnp.arange(B)[:, None],
        jnp.arange(x_BxL.shape[1])[None, :],
        targets_BxL,
    ]

    alpha_deriv_B = get_alpha_deriv(t_B)
    weight_B = -alpha_deriv_B / (1.0 - alpha_t_B + self.hps['epsilon'])

    loss_BxL = -log_prob_true_BxL * is_masked_BxL.astype(
        log_prob_true_BxL.dtype
    )

    weighted_loss_BxL = weight_B[:, None] * loss_BxL

    pad_weights = batch.get('weights')
    if pad_weights is not None:
      weighted_loss_BxL = weighted_loss_BxL * pad_weights

    total_loss = jnp.sum(weighted_loss_BxL)
    if pad_weights is not None:
      num_tokens = jnp.sum(pad_weights)
    else:
      num_tokens = jnp.array(B * x_BxL.shape[1], dtype=jnp.float32)
    return total_loss / num_tokens

  def inference(self, params, batch, rng):
    logits_BxLxV, z_BxL, is_masked_BxL, t_B, _ = self._mask_and_forward(
        params, batch, rng, train=False
    )
    predictions_BxL = jnp.argmax(logits_BxLxV, axis=-1)
    return predictions_BxL, z_BxL, is_masked_BxL, t_B

  def evaluate_batch(self, params, batch_stats, batch):
    rng = batch['eval_rng']
    loss = self._compute_elbo(params, batch, rng, train=False)
    return self.metrics_bundle.single_from_model_output(normalized_loss=loss)

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    loss = self._compute_elbo(params, batch, dropout_rng, train=True)

    if self.hps.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(
          params, self.hps.l2_decay_rank_threshold
      )
      loss += 0.5 * self.hps.l2_decay_factor * l2_loss

    return loss, {}
