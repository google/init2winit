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

"""Implementation of the Adan optimizer.

Paper: https://arxiv.org/abs/2208.06677
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import wrappers
from optax._src.alias import _scale_by_learning_rate
from optax._src.utils import canonicalize_dtype
from optax._src.utils import cast_tree


class ScaleByAdanState(NamedTuple):
  """State for the Adan algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  vu: optax.Updates
  nu: optax.Updates
  prev_updates: Optional[optax.Updates] = None


def scale_by_adan(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  """The Adan optimizer.

  Adan is an Adam variant with Nesterov acceleration.
  References:
    Xie et al, 2022: https://arxiv.org/abs/2208.06677
  Args:
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the first moment of the difference of
      the gradients over the last two steps.
    b3: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root to
      avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling.
    mu_dtype: Optional `dtype` to be used for the accumulator for both the first
      moment of the gradients and the difference of gradients over the last two
      steps; if `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  mu_dtype = canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    vu = jax.tree_util.tree_map(jnp.zeros_like,
                                params)  # First moment of gradient differences
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    prev_updates = None
    return ScaleByAdanState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        vu=vu,
        nu=nu,
        prev_updates=prev_updates)

  def update_fn(updates, state, params=None):
    del params
    prev_updates = state.prev_updates
    if not prev_updates:
      prev_updates = updates

    mu = optax.update_moment(updates, state.mu, b1, 1)
    updates_diff = jax.tree_util.tree_map(lambda a, b: a - b, updates,
                                          prev_updates)
    vu = optax.update_moment(updates_diff, state.vu, b2, 1)
    nu = optax.update_moment_per_elem_norm(
        jax.tree_util.tree_map(lambda u, ud: u + b2 * ud, updates,
                               updates_diff), state.nu, b3, 2)
    count_inc = optax.safe_int32_increment(state.count)
    mu_hat = optax.bias_correction(mu, b1, count_inc)
    vu_hat = optax.bias_correction(vu, b2, count_inc)
    nu_hat = optax.bias_correction(nu, b3, count_inc)
    updates_new = jax.tree_util.tree_map(
        lambda m, v, n: (m + b2 * v) / (jnp.sqrt(n + eps_root) + eps), mu_hat,
        vu_hat, nu_hat)
    mu = cast_tree(mu, mu_dtype)
    return updates_new, ScaleByAdanState(
        count=count_inc, mu=mu, vu=vu, nu=nu, prev_updates=updates)

  return optax.GradientTransformation(init_fn, update_fn)


AddAdanDecayedWeights = optax.EmptyState


def add_adan_decayed_weights(
    learning_rate: float,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None
) -> optax.GradientTransformation:
  """Adan-style weight decay.

    Applies weight decay after updating the parameters by the learning rate.
    Implements the no_prox=False variant of
    Adan in the original implementation.

  Args:
    learning_rate: A fixed global scaling factor.
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params):
    del params
    return AddAdanDecayedWeights()

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def wd_func(p, u):
      return -p + ((p - learning_rate * u) / (1 + learning_rate * weight_decay))

    updates = jax.tree_util.tree_map(wd_func, params, updates)

    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        optax.GradientTransformation(init_fn, update_fn), mask)
  return optax.GradientTransformation(init_fn, update_fn)


def adan(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
    use_adan_wd: Optional[bool] = False,
) -> optax.GradientTransformation:
  """The Adan optimizer.

  Adan is an Adam variant with Nesterov acceleration.
  References:
    Xie et al, 2022: https://arxiv.org/abs/2208.06677
  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the first moment of the difference of
      the gradients over the last two steps.
    b3: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root to
      avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling.
    mu_dtype: Optional `dtype` to be used for the accumulator for both the first
      moment of the gradients and the difference of gradients over the last two
      steps; if `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adan gradient transformations are applied to all parameters.
    use_adan_wd: flag to use adan-style weight decay instead of standard
      adamw-style weight decay.

  Returns:
    The corresponding `GradientTransformation`.
  """

  if use_adan_wd:
    return optax.chain(
        scale_by_adan(
            b1=b1, b2=b2, b3=b3, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        add_adan_decayed_weights(learning_rate, weight_decay,
                                 weight_decay_mask),
    )
  else:
    return optax.chain(
        scale_by_adan(
            b1=b1, b2=b2, b3=b3, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        optax.add_decayed_weights(weight_decay, weight_decay_mask),
        _scale_by_learning_rate(learning_rate),
    )
