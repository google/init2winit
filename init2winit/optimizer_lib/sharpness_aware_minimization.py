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

"""Implementation of Sharpness Aware Minimization (SAM).

This implementation is still being actively evaluated by the MLCommons Training
Algorithms Benchmark, so it should not be used (yet).

  Paper: https://arxiv.org/abs/2010.01412
  Code: https://github.com/google-research/sam
"""

from typing import Optional

from init2winit.model_lib import model_utils

import jax
import jax.numpy as jnp
import optax

_GRAD_CLIP_EPS = 1e-6


# Copied from the official SAM GitHub repository. Note how it doesn't add an
# epsilon to the gradient norm before normalizing the gradients.
def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t.

  ||x||_2 <= 1.

  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(
      sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree.map(lambda x: x / gradient_norm, y)
  return normalized_gradient


def sharpness_aware_minimization(
    rho: float,
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

  def update_fn(updates, state, grad_fn_params_tuple):
    (grad_fn, params) = grad_fn_params_tuple

    # Updates here have been averaged across devices in Trainer before being
    # sent to the optimizer. We obtain gradients computed on the noised
    # parameters in the same order as how Trainer does on the original
    # gradients and with the same 1e-6 epsilon that is used when clipping the
    # gradients.
    updates = dual_vector(updates)
    noised_params = jax.tree_util.tree_map(lambda p, u: p + rho * u, params,
                                           updates)
    _, updates = grad_fn(noised_params)

    updates_norm = jnp.sqrt(model_utils.l2_regularization(updates, 0))
    if grad_clip:
      scaled_updates = jax.tree.map(
          lambda x: x / (updates_norm + _GRAD_CLIP_EPS) * grad_clip, updates)
      updates = jax.lax.cond(updates_norm > grad_clip, lambda _: scaled_updates,
                             lambda _: updates, None)
    updates, state = base_opt_update_fn(updates, state, params)

    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
