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

"""Transforms."""
from typing import Any
from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax

from optax._src import utils  # pylint:disable=protected-access

# pylint:disable=invalid-name


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_multimap(lambda g, t: (1 - decay) * (g**order) + decay * t,
                           updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def nesterov(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:

  return optax.trace(
      decay=decay, nesterov=True, accumulator_dtype=accumulator_dtype)


def polyak_hb(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:

  return optax.trace(
      decay=decay, nesterov=False, accumulator_dtype=accumulator_dtype)


def polyak_ema(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:

  return optax.ema(
      decay=decay, debias=False, accumulator_dtype=accumulator_dtype)


class PreconditionByAdamState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array
  nu: optax.Updates


def precondition_by_adam(b2: float = 0.999,
                         eps: float = 1e-8,
                         eps_root: float = 0.0,
                         debias: bool = True) -> optax.GradientTransformation:
  """Adam preconditioner."""

  def init_fn(params):
    return PreconditionByAdamState(
        count=jnp.zeros([], jnp.int32), nu=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_multimap(lambda u, v: u / (jnp.sqrt(v + eps_root) + eps),
                                updates, nu_hat)
    return updates, PreconditionByAdamState(count=count, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAMSGradState(NamedTuple):
  """State for the AMSGrad algorithm."""
  mu: optax.Updates
  nu: optax.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  References:
    [Reddi et al, 2018](https://arxiv.org/abs/1904.09237v1)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAMSGradState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    nu_hat = jax.tree_multimap(jnp.maximum, nu, state.nu)
    updates = jax.tree_multimap(lambda m, v: m / (jnp.sqrt(v + eps_root) + eps),
                                mu, nu_hat)
    return updates, ScaleByAMSGradState(mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class BiasCorrectionState(NamedTuple):
  """Holds an exponential moving average of past updates."""
  count: chex.Array  # shape=(), dtype=jnp.int32.


def bias_correction(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None) -> optax.GradientTransformation:
  """Compute the Adam style bias correction.

  Args:
    decay: the decay rate for the exponential moving average.
    accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    del params
    return BiasCorrectionState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    count_inc = utils.safe_int32_increment(state.count)
    new_vals = _bias_correction(updates, decay, count_inc)
    return new_vals, BiasCorrectionState(count=count_inc)

  return optax.GradientTransformation(init_fn, update_fn)
