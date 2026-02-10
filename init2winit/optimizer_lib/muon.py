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

"""Muon optimizer implementation in JAX.

blogpost for optimizer explanation:
https://kellerjordan.github.io/posts/muon/

reference torch code compared against:
1) https://github.com/KellerJordan/Muon
2) https://github.com/KellerJordan/modded-nanogpt/blob/d700b8724cbda3e7b1e5bcadbc0957f6ad1738fd/train_gpt.py#L135  # pylint: disable=line-too-long

"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


@jax.jit
def orthogonalize_via_newton_schulz(
    updates: jax.Array,
    newton_schulz_coeffs: jax.Array,
    newton_schulz_steps: int = 1,
    eps: float = 1e-7,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    updates: A matrix to orthogonalize.
    newton_schulz_coeffs: Coefficients for the Newton-Schulz method.
    newton_schulz_steps: Number of Newton-schulz iterations.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix.
  """
  was_reshaped = False
  original_shape = updates.shape

  if updates.ndim == 3:
    updates = updates.reshape(updates.shape[0], -1)
    was_reshaped = True

  if updates.ndim != 2:
    raise ValueError(f'Input must be 2D, got {updates.shape}')

  # Ensure spectral norm is at most 1
  updates /= (jnp.linalg.norm(updates) + eps)

  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x

  # Transpose the 2-D matrix so bigger dimension is first.
  # If the matrix is transposed, we need to transpose the final output.
  transposed = False

  if updates.shape[0] > updates.shape[1]:
    updates = updates.T
    transposed = True

  orthogonalized_updates = jax.lax.fori_loop(
      0,
      newton_schulz_steps,
      lambda _, x: newton_schulz_iterator(x, newton_schulz_coeffs),
      updates
  )

  if transposed:
    orthogonalized_updates = orthogonalized_updates.T

  fan_out = orthogonalized_updates.shape[1]
  fan_in = orthogonalized_updates.shape[0]

  # Scaling factor taken from https://jeremybernste.in/writing/deriving-muon
  # and https://github.com/KellerJordan/modded-nanogpt/blame/822ab2dd79140ed34ae43a20450f0bdc36457a24/train_gpt.py#L184 # pylint: disable=line-too-long
  scale_factor = jnp.maximum(1.0, jnp.sqrt(fan_out / fan_in))
  orthogonalized_updates *= scale_factor

  if was_reshaped:
    orthogonalized_updates = orthogonalized_updates.reshape(original_shape)

  return orthogonalized_updates


class MuonState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  momentum: optax.Updates


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree.map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
  )


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree.map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_muon(
    learning_rate: float = 0.0,
    beta: float = 0.95,
    weight_decay: float = 0.01,
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
    nesterov: bool = True,
    bias_correction: bool = False,
) -> optax.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Args:
    learning_rate: Learning rate.
    beta: Decay rate for the gradient momentum.
    weight_decay: Weight decay coefficient.
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
    eps: Term added to denominators to improve numerical stability.
    nesterov: Whether to use Nesterov momentum.
    bias_correction: Whether to perform bias correction.

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  ns_coeffs = jnp.asarray(ns_coeffs)

  def init_fn(params):
    momentum = jax.tree_util.tree_map(
        lambda p: jnp.zeros_like(p, jnp.float32), params
    )  # First moment

    return MuonState(
        count=jnp.zeros([], jnp.int32),
        momentum=momentum,
    )

  def update_fn(updates, state, params=None):
    running_momentum = _update_moment(updates, state.momentum, beta, 1)
    new_count = optax.safe_int32_increment(state.count)

    if nesterov:
      momentum = _update_moment(updates, running_momentum, beta, 1)
    else:
      momentum = running_momentum

    if bias_correction:
      momentum = _bias_correction(momentum, beta, new_count)

    # Apply Newton-schulz orthogonalization.
    scaled_orthogonalized_momentum = jax.tree_util.tree_map(
        lambda x: orthogonalize_via_newton_schulz(
            x, ns_coeffs, ns_steps, eps
        ),
        momentum,
    )

    # Apply weight decay similar to how it's being applied here :
    # https://github.com/KellerJordan/Muon/commit/e0ffefd4f7ea88f2db724caa2c7cfe859155995d#diff-ff0575a769b2390ce3256edb1c20e4d741d514a77c4f0697c2fa628f810f46b1R60-R80
    new_updates = jax.tree_util.tree_map(
        lambda u, p: -learning_rate * (u + weight_decay * p),
        scaled_orthogonalized_momentum, params
    )

    return new_updates, MuonState(
        count=new_count,
        momentum=running_momentum,
    )
  return optax.GradientTransformation(init_fn, update_fn)
