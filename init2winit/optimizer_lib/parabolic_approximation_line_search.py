# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Implementation of Parabolic Approximation Line Search (PAL).

  Paper: https://arxiv.org/abs/1903.11991
  Code: https://github.com/cogsys-tuebingen/PAL
"""

from typing import NamedTuple

from init2winit.model_lib import model_utils

import jax
from jax import lax
import jax.numpy as jnp
import optax


class ParabolicApproximationLineSearchState(NamedTuple):
  step: jnp.ndarray  # shape=(), dtype=jnp.int32.
  base_state: NamedTuple  # The state of the base optimizer.
  hyperparams: dict[str, jnp.ndarray]  # The base optimizer's hyperparams.


def parabolic_approximation_line_search(
    mu: float,
    alpha: float,
    s_max: float,
    start_step: int,
    stop_step: int,
    batch_axis_name: str,
    base_opt_init_fn,
    base_opt_update_fn,
) -> optax.GradientTransformation:
  """Implementation of Parabolic Approximation Line Search (PAL).

  Paper: https://arxiv.org/abs/1903.11991
  Code: https://github.com/cogsys-tuebingen/PAL

  References:
    Mutschler and Zell, 2021: https://arxiv.org/abs/1903.11991
  Args:
    mu: The measuring step size to use when computing the loss as the projected
      point.
    alpha: The update step adaptation used when computing the update.
    s_max: The upper bound for the maximum step size that we can take.
    start_step: The step to start using PAL at.
    stop_step: The step to stop using PAL at.
    batch_axis_name: the name of the axis to pmap over. Used to run a pmean
      before applying the optimizer update.
    base_opt_init_fn: The initialization function for the base optimizer used to
      generate updates given the total gradient.
    base_opt_update_fn: The update function for the base optimizer used to
      generate updates given the total gradient.

  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params):
    base_state = base_opt_init_fn(params)
    return ParabolicApproximationLineSearchState(
        step=jnp.zeros([], dtype=jnp.int32),
        base_state=base_state,
        hyperparams=base_state.hyperparams)

  def update_fn(updates, state, cost_fn_params_tuple):
    (cost_fn, params) = cost_fn_params_tuple

    def pal_update(updates, state, params):

      def loss_fn(params):
        loss, _ = cost_fn(params)
        return lax.pmean(loss, axis_name=batch_axis_name)

      loss = loss_fn(params)

      grad = updates
      updates, state = base_opt_update_fn(updates, state, params)

      updates_norm = jnp.sqrt(model_utils.l2_regularization(updates, 0))
      updates = jax.tree_util.tree_map(lambda u: u / updates_norm, updates)
      new_params = optax.apply_updates(
          params, jax.tree_util.tree_map(lambda u: mu * u, updates))
      new_loss = loss_fn(new_params)

      b = jax.tree_util.tree_reduce(
          lambda a, b: a + b,
          jax.tree_util.tree_map(lambda g, u: jnp.sum(g * u), grad, updates))
      a = (new_loss - loss - b * mu) / (mu**2)

      def line_search_update(mu, alpha, a, b):
        del mu
        return (-1.0 * alpha * b) / (2.0 * a)

      def mu_update(mu, alpha, a, b):
        del alpha, a, b
        return mu

      def noop_update(mu, alpha, a, b):
        del mu, alpha, a, b
        return 0.0

      s_upd_1 = lax.cond(
          jnp.logical_and(jnp.greater(a, 0), jnp.less(b, 0)),
          line_search_update, noop_update, mu, alpha, a, b)

      s_upd_2 = lax.cond(
          jnp.logical_and(jnp.less_equal(a, 0), jnp.less(b, 0)), mu_update,
          noop_update, mu, alpha, a, b)

      s_upd = jnp.maximum(s_upd_1, s_upd_2)
      s_upd = lax.cond(jnp.greater(s_upd, s_max), lambda: s_max, lambda: s_upd)
      state.hyperparams['learning_rate'] = s_upd

      def scale_update(updates, lr):
        return jax.tree_util.tree_map(lambda u: u * lr, updates)

      def scale_by_zeros_update(updates, lr):
        del lr
        return jax.tree_util.tree_map(jnp.zeros_like, updates)

      updates = lax.cond(
          jnp.greater(s_upd, 0.0), scale_update, scale_by_zeros_update, updates,
          s_upd)

      return updates, state

    def base_optimizer_update(updates, state, params):
      return base_opt_update_fn(updates, state, params)

    updates, base_state = lax.cond(
        jnp.logical_and(
            jnp.greater_equal(state.step, start_step),
            jnp.less_equal(state.step, stop_step)),
        pal_update,
        base_optimizer_update,
        updates,
        state.base_state,
        params)

    step = state.step + jnp.ones([], dtype=jnp.int32)
    state = ParabolicApproximationLineSearchState(
        step=step, base_state=base_state, hyperparams=base_state.hyperparams)

    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
