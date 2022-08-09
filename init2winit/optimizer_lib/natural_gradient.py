# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""Optimizers for natural gradient algorithms."""

from typing import NamedTuple
import flax
import jax
import jax.numpy as jnp
import optax


def unfreeze_wrapper(init_fn, update_fn):
  """Freeze/unfreeze params."""

  # NOTE(dsuo): We use plain dicts internally due to this issue
  # https://github.com/deepmind/optax/issues/160.
  def wrapped_init_fn(params):
    return init_fn(flax.core.unfreeze(params))

  def wrapped_update_fn(updates, state, params=None):
    new_updates, state = update_fn(
        flax.core.unfreeze(updates), state,
        None if params is None else flax.core.unfreeze(params))

    if isinstance(updates, flax.core.FrozenDict):
      new_updates = flax.core.freeze(new_updates)

    return new_updates, state

  return optax.GradientTransformation(wrapped_init_fn, wrapped_update_fn)


class ScaleBySSState(NamedTuple):
  """State holding the sum of gradient squares to date."""
  sum_of_squares: optax.Updates


def last_layer_transformation(last_layer_optimizer,
                              base_lr,
                              last_layer_base_lr):
  """Use an optimizer while scaling by a different learning rate."""
  def init_fn(params):
    return last_layer_optimizer.init(params)

  def update_fn(updates, state, params=None):
    updates, state = last_layer_optimizer.update(updates, state, params)
    updates = jax.tree_util.tree_map(lambda g: last_layer_base_lr / base_lr * g,
                                     updates)

    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_ss(initial_accumulator_value: float = 0.1,
                eps: float = 1e-7) -> optax.GradientTransformation:
  r"""Scale by sum of squares."""

  def init_fn(params):
    sum_of_squares = jax.tree_util.tree_map(
        lambda t: jnp.full_like(t, initial_accumulator_value), params)
    return ScaleBySSState(sum_of_squares=sum_of_squares)

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jax.tree_util.tree_map(
        lambda g, t: jax.lax.cumsum(jax.lax.square(g)) + t, updates,
        state.sum_of_squares)
    inv_g_square = jax.tree_util.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.reciprocal(t + eps), 0.0),
        sum_of_squares)
    updates = jax.tree_util.tree_map(lambda scale, g: scale * g, inv_g_square,
                                     updates)
    return updates, ScaleBySSState(sum_of_squares=sum_of_squares)

  return optax.GradientTransformation(init_fn, update_fn)


def online_diag_newton_step(learning_rate,
                            weight_decay):
  r"""An optimizer that does diagonal newton preconditioning on a single specified layer."""

  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      scale_by_ss(),
      optax.sgd(learning_rate))


def sherman_morrison(a_inv, u, alpha):
  """Given A^-1, compute (A + alpha * u u^T)^-1 using Sherman-Morrison."""
  denom = 1 + alpha * u.T @ a_inv @ u
  numer = alpha * jnp.outer(a_inv @ u, u) @ a_inv

  return a_inv - numer / denom


class OnlineNewtonState(NamedTuple):
  """State holding the sum of gradient squares to date."""
  inv_hessian: optax.Updates


def full_matrix_ons(alpha, initial_accumulator_value=0.1):
  """A full Online Newton Step transformation."""

  def init_fn(params):
    raveled_params, _ = jax.flatten_util.ravel_pytree(params)
    initial_hessian = jnp.diag(
        jnp.full_like(raveled_params, 1. / initial_accumulator_value))

    return OnlineNewtonState(inv_hessian=initial_hessian)

  def update_fn(updates, state, params=None):
    del params

    raveled_updates, unravel = jax.flatten_util.ravel_pytree(updates)
    new_hessian = sherman_morrison(state.inv_hessian, raveled_updates, alpha)
    new_updates = unravel(new_hessian @ raveled_updates)

    return new_updates, OnlineNewtonState(inv_hessian=new_hessian)

  return optax.GradientTransformation(init_fn, update_fn)


def online_newton_step(learning_rate,
                       alpha,
                       weight_decay):
  r"""An optimizer that does full matrix preconditioning."""

  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      full_matrix_ons(alpha),
      optax.sgd(learning_rate))


def nice_function_optimizer(last_layer_name,
                            network_optimizer,
                            last_layer_optimizer,
                            last_layer_base_lr,
                            base_lr):
  """Use a different optimizer for the last layer."""

  def get_select_fn(layer_name):
    """Get a function that selects the specified layer as last layer."""

    def select_layer(tree):
      return {k: ('ll' if k == layer_name else 'net') for k, v in tree.items()}

    return select_layer

  return unfreeze_wrapper(*optax.multi_transform(
      {
          'net': network_optimizer,
          # Scale the learning rate of the last layer according to match
          # last_layer_base_lr
          'll': last_layer_transformation(last_layer_optimizer,
                                          base_lr,
                                          last_layer_base_lr),
      }, get_select_fn(last_layer_name)))
