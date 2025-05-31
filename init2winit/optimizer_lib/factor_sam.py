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

"""Efficient implementation of Sharpness Aware Minimization (SAM).

Applies SAM learning rule every k steps, and factorizes the perturbation
radius and the regularization strength.
"""

import functools
from typing import Optional

from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
import optax

_GRAD_CLIP_EPS = 1e-6


def normalize_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns unit norm version of original pytree.

  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(
      sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree.map(lambda x: x / gradient_norm, y)
  return normalized_gradient


def clean_update(updates, state, unused_grad_fn_params_tuple):
  """Returns the clean update function."""
  return updates, state


def sam_update(
    updates,
    state,
    grad_fn_params_tuple,
    rho=0.1,
    alpha=1.0,
):
  """SAM update function."""
  (grad_fn, params) = grad_fn_params_tuple
  updates = normalize_vector(updates)
  noised_params = jax.tree_util.tree_map(
      lambda p, u: p + rho * u, params, updates
  )
  _, sam_updates = grad_fn(noised_params)

  # Regularizer gradient - difference between SAM and clean updates.
  sam_updates = jax.tree.map(lambda x, y: x - y, sam_updates, updates)

  # Rescale and apply regularizer
  updates = jax.tree.map(lambda x, y: x + alpha * y, updates, sam_updates)
  return updates, state


def sharpness_aware_minimization(
    rho: float,
    alpha: float,
    k: int,
    grad_clip: Optional[float],
    base_opt_init_fn,
    base_opt_update_fn,
) -> optax.GradientTransformation:
  """Implementation of Sharpness Aware Minimization (SAM).

  Paper: https://arxiv.org/abs/2010.01412
  Code: https://github.com/google-research/sam

  References:
    Foret et al, 2021: https://arxiv.org/abs/2010.01412
  Args:
    rho: The size of the neighborhood for the sharpness aware minimization
      gradient updates. Defaults to 0.1.
    alpha: Additional scaling factor for regularization strength.
    k: Period on which to apply SAM. Regularization strength is scaled by k.
    grad_clip: The optional value to clip the updates by. Defaults to None.
    base_opt_init_fn: The initialization function for the base optimizer used to
      generate updates given the total gradient.
    base_opt_update_fn: The update function for the base optimizer used to
      generate updates given the total gradient.

  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params):
    return base_opt_init_fn(params)

  # TODO(thetish): Implement version which applies SAM before averaging over
  # devices.
  def update_fn(updates, state, grad_fn_params_tuple):
    # Updates here have been averaged across devices in Trainer before being
    # sent to the optimizer.
    (_, params) = grad_fn_params_tuple

    # Update function in between SAM steps.
    intermediate_update_fn = clean_update

    # Sam update. Scale alpha by k to keep optimal rho independent of k.
    alpha_eff = alpha * k
    sam_update_fn = functools.partial(sam_update, rho=rho, alpha=alpha_eff)
    updates, state = jax.lax.cond(  # Apply SAM every k steps.
        state.count % k == 0,
        sam_update_fn,
        intermediate_update_fn,
        updates,
        state,
        grad_fn_params_tuple,
    )
    # Clipping
    if grad_clip:
      updates_norm = jnp.sqrt(model_utils.l2_regularization(updates, 0))
      scaled_updates = jax.tree.map(
          lambda x: x / (updates_norm + _GRAD_CLIP_EPS) * grad_clip, updates)
      updates = jax.lax.cond(updates_norm > grad_clip, lambda _: scaled_updates,
                             lambda _: updates, None)
    # TODO(thetish): Explore different order for base optimizer and SAM. For
    # example, in Adam preconditioning the SAM perturbation is helpful.
    return base_opt_update_fn(updates, state, params)  # Apply base optimizer

  return optax.GradientTransformation(init_fn, update_fn)
