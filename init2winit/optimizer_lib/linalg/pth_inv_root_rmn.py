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

"""Type-(2, 2) rational minimax iteration for the inverse matrix pth root of a symmetric matrix."""

import functools
import operator
from typing import Tuple

from absl import logging
import chex
from init2winit.optimizer_lib.linalg import paterson_stockmeyer
from init2winit.optimizer_lib.linalg import pth_inv_root_rmn_coefficients
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def _xxt_params(m: int, n: int) -> dict[str, int] | None:
  """Returns the parameters for _xxt if applicable."""
  bm, bk, bn = 256, 1024, 256  # Block sizes, hard coded for now.
  if (
      m == n
      and m % bm == 0
      and n % bn == 0
      and n % bk == 0
      and bm <= m
      and bn <= n
      and bk <= n
  ):
    return dict(bm=bm, bk=bk, bn=bn)
  return None


def pthroot_xxt(x: chex.Array, scale: chex.Array | None = None) -> chex.Array:
  """(x * scale) @ x.T ."""
  return (x if scale is None else x * scale) @ x.T


def _scalar_power(x: chex.Array, n: int) -> chex.Array:
  """Evaluation of x**n for n > 0 by repeated squaring."""
  if n == 1:
    return x
  elif n % 2 == 0:
    return _scalar_power(x * x, n // 2)
  else:
    return x * _scalar_power(x * x, (n - 1) // 2)


def _scalar_inverse_root(x: chex.Array, n: int) -> chex.Array:
  """Specialization of x**(-1/n) to work around low accuracy on TPU."""
  if n == 1:
    return 1 / x
  elif n == 2:
    return lax.rsqrt(x)
  elif n == 3:
    return 1 / lax.cbrt(x)
  elif n == 4:
    return lax.rsqrt(lax.sqrt(x))
  elif n == 6:
    return lax.rsqrt(lax.cbrt(x))
  elif n == 8:
    return lax.rsqrt(lax.sqrt(lax.sqrt(x)))
  else:
    r = x**(1 / n)
    # One step of Newton's method to polish the root
    r = ((n - 1) / n) * r + (x / n) / _scalar_power(r, n - 1)
    return 1 / r


@functools.cache
def _binomial_coefficients(n: int) -> np.ndarray:
  """The integer binomial coefficients 1, n, n(n-1)/2, ..., n, 1."""
  c = np.ones((n + 1,), np.int32)
  for j in range(1, n + 1):
    c[j] = (c[j - 1] * (n - j + 1)) // j
  return c


def _get_precision_string(precision: lax.Precision | str) -> str:
  if isinstance(precision, str):
    return precision
  elif precision == lax.Precision.HIGHEST:
    return "float32"
  elif precision == lax.Precision.HIGH:
    return "tensorfloat32"
  elif precision == lax.Precision.DEFAULT:
    return "bfloat16"
  else:
    raise NotImplementedError(f"Unsupported precision: {precision}")


def pth_inv_root_rmn(
    x: chex.Array,
    p: int,
    fast_root: bool = False,
    precision: lax.Precision | str = "float32",
    stable_iter: bool = False,
    unroll: bool = False,
    verbose: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Returns X⁻¹ᐟᵖ.

    Runs the following fixed point iteration:
      X_{k+1} = X_{k} f_k(Y_k X_k, alpha_k)^{p-1}, X_{0} = X
      Y_{k+1} = f_k(Y_k X_k, alpha_k) Y_{k}, Y_{0} = I
      alpha_{k+1} = alpha_k f_k(Y_k X_k, alpha_k), alpha_0 = sigma_{min}^{1/p}

      X_k converged to X^{1/p}, Y_k converges to X^{-1/p}

  Args:
    x: Input matrix must be SPD with eigenvalues >= float32 epsilon.
    p: Exponent.
    fast_root: If True, use a lower mixed degree approximation of x^{1/p} (in
      the slower version, we use a mixed degree approximation of 2 and 3.
      In the fast version, degree 2 and 3 is used).
    precision: Matrix multiplication precision to use (on TPU). See
      `jax.default_matmul_precision`.
    stable_iter: Whether to use the stable iteration for the inner loop.
    unroll: Whether to unroll the loop over iterations.
    verbose: Whether to log some information about the iteration, including the
      coefficients `a` for each iteration.

  Returns:
    An approximation of |X|⁻¹ᐟᵖ.
  """
  assert x.ndim == 2, x.shape
  assert x.shape[0] == x.shape[1], x.shape
  x = x.astype(jnp.float32)
  n = x.shape[-1]

  alpha = jax.lax.sqrt(jnp.linalg.norm(x, ord=1))
  alpha *= jax.lax.sqrt(jnp.linalg.norm(x, ord=jnp.inf))
  alpha = lax.select(alpha == 0, jnp.ones_like(alpha), alpha)
  beta = _scalar_inverse_root(alpha, p)

  x = x / alpha
  eye = jnp.identity(n, jnp.float32)

  ais, bis, cis = lax.cond(
      fast_root,
      lambda: pth_inv_root_rmn_coefficients.r12_schedule(p),
      lambda: pth_inv_root_rmn_coefficients.r23_schedule(p)
  )
  max_k = len(cis)

  @jax.default_matmul_precision(_get_precision_string(precision))
  def chol_inv(m):
    l_m = jnp.linalg.cholesky(m)
    e = jnp.finfo(jnp.float32).eps
    def _avoid_nan_body(val):
      (_, e, m) = val
      l = jnp.linalg.cholesky(m + jnp.diag(e * jnp.diag(m)))
      e = 2*e
      return (l, e, m)

    l_m, _, _ = lax.while_loop(
        lambda val: jnp.any(jnp.isnan(val[0])),
        _avoid_nan_body,
        (l_m, e, m),
    )
    d_m = lax.linalg.triangular_solve(
        l_m, eye, left_side=True, lower=True, transpose_a=False
    )
    return pthroot_xxt(d_m.T)
    # return d_m.T @ d_m

  @jax.default_matmul_precision(_get_precision_string(precision))
  def general_iteration(k, val, p, symm_inv=chol_inv):
    x, y = val
    a = lax.dynamic_index_in_dim(ais, k, keepdims=False)
    b = lax.dynamic_index_in_dim(bis, k, keepdims=False)
    c = lax.dynamic_index_in_dim(cis, k, keepdims=False)
    cpm1 = _scalar_power(c, p - 1)
    bc = _binomial_coefficients(p - 1).astype(jnp.float32)

    offset_k = lax.select(k == 0, 1, 0)
    offset_fr = lax.select(fast_root, 1, 0)
    max_r = len(a) - offset_k - offset_fr

    if stable_iter:
      w_n = jnp.zeros((n, n), jnp.float32)
      y_inv = lax.cond(k == 0, lambda m: m, symm_inv, y)
      def inside_iter(s, w_sum):
        a2 = lax.dynamic_index_in_dim(a, s, keepdims=False)
        b2 = lax.dynamic_index_in_dim(b, s, keepdims=False)
        d = symm_inv(x + b2 * y_inv)
        return w_sum + a2 * d

      w_n = lax.fori_loop(0, max_r, inside_iter, w_n)
      w = w_n @ y_inv
      y = c * (y + w_n)
    else:
      w = jnp.zeros((n, n), jnp.float32)
      def inside_iter2(s, w_sum):
        a2 = lax.dynamic_index_in_dim(a, s, keepdims=False)
        b2 = lax.dynamic_index_in_dim(b, s, keepdims=False)
        d = symm_inv(y @ x + b2 * eye)
        return w_sum + a2 * d

      w = lax.fori_loop(0, max_r, inside_iter2, w)
      y = c * (y + y @ w)

    if p > 1:
      x += x @ paterson_stockmeyer.polynomial_no_constant(
          bc[1:], w, operator.matmul
      )
    x = cpm1 * x
    return x, y

  if p == 1:
    iteration = functools.partial(general_iteration, p=2 * p)
  else:
    iteration = functools.partial(general_iteration, p=p)

  if verbose:
    logging.info(
        "[pth_inv_root_rmn] x-shape = %s x %s, exponent = %s, total outer steps"
        " = %s",
        x.shape[0],
        x.shape[1],
        p,
        max_k,
    )
  val = (x, eye)

  if unroll:
    for k in range(max_k):
      val = iteration(k, val)
  else:
    val = lax.fori_loop(0, max_k, iteration, val)

  if p == 1:
    # inverse is computed as square of the inverse square root
    with jax.default_matmul_precision(_get_precision_string(precision)):
      return alpha * x, beta * val[1] @ val[1]
  return 1/beta * val[0], beta * val[1]
