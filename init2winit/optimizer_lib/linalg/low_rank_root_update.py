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

"""A Sherman-Morrison-Woodbury-type update for the inverse sqrt."""

from typing import Tuple

from absl import logging
import chex
from init2winit.optimizer_lib.linalg import pth_inv_root_rmn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def norm_upper_bound(x):
  alpha = jax.lax.sqrt(jnp.linalg.norm(x, ord=1))
  alpha *= jax.lax.sqrt(jnp.linalg.norm(x, ord=jnp.inf))
  return lax.select(alpha == 0, jnp.ones_like(alpha), alpha)


def safe_chol(m):
  l_m = jnp.linalg.cholesky(m)
  l_m = jnp.where(jnp.any(jnp.isnan(l_m)), jnp.zeros_like(l_m), l_m)
  return l_m


def chol_inv(m):
  """Computes the inverse of a spd matrix using Cholesky decomposition.

  Args:
    m: The matrix to invert.

  Returns:
    The inverse of the matrix.
  """
  eye = jnp.identity(m.shape[-1], dtype=jnp.float32)
  l_m = jnp.linalg.cholesky(m)
  def _get_inv():
    d_m = lax.linalg.triangular_solve(
        l_m, eye, left_side=True, lower=True, transpose_a=False
    )
    return pth_inv_root_rmn.pthroot_xxt(d_m.T)

  def _get_zeros():
    return jnp.zeros_like(m)

  return lax.cond(jnp.any(jnp.isnan(l_m)), _get_zeros, _get_inv)


def lyapunov_solver(
    a: chex.Array,
    c: chex.Array,
    eps: float,
    num_terms: int = 5):
  """Solves the Sylvester equation ax + xa = c using R. A. Smith's method.

  Args:
    a: spd matrix
    c: the right hand side of above equation
    eps: small constant to avoid numerical issues with Lyapunov solver.
    num_terms: Number of terms in the expansion of the solution.

  Returns:
    solution to the Lyapunov equation

  """
  a_norm = norm_upper_bound(a)
  a_norm = jnp.where(a_norm == 0, 1, a_norm)
  c_norm = norm_upper_bound(c)
  c_norm = jnp.where(c_norm == 0, 1, c_norm)
  na = a / a_norm
  nc = c / c_norm
  alph = c_norm / a_norm
  gamma = jnp.median(jnp.diag(na))
  gamma = jnp.where(gamma < eps, eps, gamma)

  a2 = na + gamma * jnp.identity(na.shape[-1], dtype=na.dtype)
  b2 = na - gamma * jnp.identity(na.shape[-1], dtype=na.dtype)
  ai2 = chol_inv(a2)
  uu = ai2 @ b2
  ww = 2 * gamma * (ai2 @ (nc @ ai2.T))
  xx = ww + uu @ (ww @ uu.T)
  vv = uu

  def _loop_body(k, val):
    del k
    vv, xx = val
    vv = vv @ vv
    xx = xx + vv @ (xx @ vv.T)
    return (vv, xx)

  (vv, xx) = lax.fori_loop(0, num_terms, _loop_body, (vv, xx))
  del vv
  return alph * xx


def matrix_sqrt_update(
    sqrt_x: chex.Array,
    stat_update: chex.Array,
    rank_array: np.ndarray,
    eps: float,
    block_krylov_dim_multiplier: int,
    rng: jax.Array | None = None,
) -> chex.Array:
  """Given A^{1/2} and an update of the form ZZ^T, returns U such that (A + ZZ^T)^{1/2} = A^{1/2} + UU^T.

  Args:
    sqrt_x: Original matrix sqrt.
    stat_update: Update to the matrix of the form stat_update @ stat_update.T
    rank_array: Array containing rank estimate of the update.
    eps: small constant to avoid numerical issues with Lyapunov solver.
    block_krylov_dim_multiplier: Multiplier for the block krylov dimension over
      the rank estimate.
    rng: Jax key for random number generation.

  Returns:
    Gram factor U of the update to the matrix sqrt.

  """
  # given A^{1/2} and an update of the form ZZ^T, returns U such that
  # (A + ZZ^T)^{1/2} = A^{1/2} + UU^T and
  # actual update is stat_update @ stat_update.T
  def block_krylov_basis(a, q, k):
    pp = (q,)
    for _ in range(k-1):
      pp = pp + (a @ pp[-1],)
    ks = jnp.hstack(pp)
    return jnp.linalg.qr(ks)[0]

  if rng is None:
    s_rng = jax.random.key(0)
  else:
    s_rng, _ = jax.random.split(rng, 2)

  def low_rank_approx(m):
    z = jax.random.normal(s_rng, (m.shape[-1], rank_array.shape[0]))
    return m @ z

  b = pth_inv_root_rmn.pthroot_xxt(stat_update)
  y = low_rank_approx(b)
  bk_basis = block_krylov_basis(sqrt_x, y, block_krylov_dim_multiplier)
  na = bk_basis.T @ (sqrt_x @ bk_basis)
  nc = bk_basis.T @ stat_update
  nq = pth_inv_root_rmn.pthroot_xxt(nc)
  na_perturbed = na + eps * jnp.identity(na.shape[-1], dtype=na.dtype)
  ny = lyapunov_solver(na_perturbed, nq, eps)
  ly = safe_chol(ny)
  u_delta = bk_basis @ ly

  u_delta = jnp.where(
      jnp.any(jnp.isnan(u_delta)), jnp.zeros_like(u_delta), u_delta
  )
  return u_delta


def low_rank_root_update(
    sqrt_x: chex.Array,
    isqrt_x: chex.Array,
    update: chex.Array,
    rank_array: np.ndarray,
    eps: float,
    block_krylov_dim_multiplier: int,
    verbose: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Returns X^{-1/2} given the old X^{1/2}, X^{-1/2} and an update to X.

    X_{new} = X_{old} + update @ update.T

  Args:
    sqrt_x: Old sqrt of the matrix X.
    isqrt_x: Old inverse sqrt of the matrix X.
    update: Update to the matrix of the form update @ update.T
    rank_array: Array containing rank estimate of the update.
    eps: small constant to avoid numerical issues with Lyapunov solver.
    block_krylov_dim_multiplier: Multiplier for the block krylov dimension over
      the rank estimate.
    verbose: Whether to log some information about the iteration.

  Returns:
    An updated approximation of |X|{1/2} and |X|{-1/2}.
  """
  sqrt_x = sqrt_x.astype(jnp.float32)
  isqrt_x = isqrt_x.astype(jnp.float32)
  update = update.astype(jnp.float32)

  if verbose:
    logging.info(
        "[low_rank_root_update] rank_est = %s, block_krylov_dim_multiplier"
        " = %s",
        rank_array.shape[0],
        block_krylov_dim_multiplier,
    )

  def _update_inv_sqrt(sqrt_a, isqrt_a, u):
    # given A^{-1/2} and an update to the sqrt from the function above, it
    # returns B = (A + ZZ^T)^{-1/2}
    k = u.shape[-1]
    ik = jnp.identity(k, jnp.float32)
    m = u.T @ (isqrt_a @ u) + ik
    ui = (isqrt_a @ u) @ pth_inv_root_rmn.pth_inv_root_rmn(m, p=2)[1]
    return sqrt_a + pth_inv_root_rmn.pthroot_xxt(
        u
    ), isqrt_a - pth_inv_root_rmn.pthroot_xxt(ui)

  rng = jax.random.key(0)
  with jax.default_matmul_precision("float32"):
    u = matrix_sqrt_update(
        sqrt_x,
        update,
        rank_array,
        eps,
        block_krylov_dim_multiplier,
        rng=rng,
    )
    new_sqrt_x, new_isqrt_x = _update_inv_sqrt(
        sqrt_x, isqrt_x, u
    )
  return new_sqrt_x, new_isqrt_x
