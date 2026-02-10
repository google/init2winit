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

"""Optimizers for online newton step algorithms."""

from typing import NamedTuple

from init2winit.optimizer_lib import kitchen_sink
from init2winit.optimizer_lib import utils
import jax
import jax.numpy as jnp
import optax


def diag_ons(learning_rate,
             weight_decay: float = 0.0,
             b1: float = 0.9,
             b2: float = 0.999,
             eps: float = 1e-8):
  """The diagonal version of Online Newton Step with flexible updates.

  Args:
    learning_rate: A fixed global scaling factor.
    weight_decay: weight decay.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root (as
      in the Adam paper) to avoid dividing by zero when rescaling.

  Returns:
    The corresponding `GradientTransformation`.
  """
  if b1 == 1.0 and b2 == 1.0:
    # Diag ONS without momentum and second moment decay
    return optax.chain(
        kitchen_sink.precondition_by_rss(eps=eps, power=1.0),
        optax.add_decayed_weights(weight_decay), optax.scale(learning_rate))
  elif b1 == 1.0 and b2 != 1.0:
    # Diag ONS without momentum but with second moment decay
    return optax.chain(
        kitchen_sink.precondition_by_rms(
            decay=b2, eps=eps, eps_root=0.0, power=1.0),
        optax.add_decayed_weights(weight_decay), optax.scale(learning_rate))
  elif b1 != 1.0 and b2 != 1.0:
    # Diag ONS with momentum and second moment decay
    return optax.chain(
        kitchen_sink.scale_by_adam(b1, b2, eps, eps_root=0.0, power=1.0),
        optax.add_decayed_weights(weight_decay), optax.scale(learning_rate))


def last_layer_transformation(last_layer_optimizer, base_lr,
                              last_layer_base_lr, learning_rate):
  """Use an optimizer while scaling by a different learning rate."""

  return optax.chain(last_layer_optimizer,
                     optax.scale(learning_rate * last_layer_base_lr / base_lr))


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


def online_newton_step(learning_rate, alpha, weight_decay):
  r"""An optimizer that does full matrix preconditioning."""

  return optax.chain(
      optax.add_decayed_weights(weight_decay), full_matrix_ons(alpha),
      optax.sgd(learning_rate))


def multiple_optimizer(last_layer_name, network_optimizer, last_layer_optimizer,
                       last_layer_base_lr, base_lr):
  """Use a different optimizer for the last layer."""

  def get_select_fn(layer_name):
    """Get a function that selects the specified layer as last layer."""

    def select_layer(tree):
      return {k: ('ll' if k == layer_name else 'net') for k, v in tree.items()}

    return select_layer

  return kitchen_sink.unfreeze_wrapper(*optax.multi_transform(
      {
          'net':
              network_optimizer,
          # Scale the learning rate of the last layer according to match
          # last_layer_base_lr
          'll':
              utils.static_inject_hyperparams(last_layer_transformation)
              (last_layer_optimizer,
               base_lr,
               last_layer_base_lr,
               learning_rate=0.0),
      },
      get_select_fn(last_layer_name)))
