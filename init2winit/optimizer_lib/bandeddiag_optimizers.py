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

"""Banded Matrix Optimizer.

This is based on https://arxiv.org/abs/2311.10085 and
https://github.com/devvrit/Autoencoder-on-MNIST-in-Pytorch.
"""

from typing import NamedTuple, Union
import chex
from init2winit.optimizer_lib import tridiag_optimizers
import jax
import jax.numpy as jnp
import optax
from optax._src import base

ScalarOrSchedule = Union[float, base.Schedule]


# Banded Matrix Operation


def solve_quad_eq(a, b, c, sig21):
  # input a, b, c are scalars, and sig21 a vector
  # eq to solve is ax^2 + bx + c
  # we'll take the solution depending on if sig21[-1]>=-b or not
  d = jnp.sqrt(b**2 - (4 * a * c))
  d = jnp.where(sig21[-1] >= -b, d, -d)
  return (-b + d) / (2 * a)


def new_genp(sig21, sig22, d):
  """Gaussian elimination with reweighing of edges if condcov<=0.

  Args:
   sig21: is batches x b vector, will play the role of "b_" in Ax=b_
   sig22: is n x (bxb) matrices, will play the rolw of "A" in Ax=b_
   d: diagonal statistics

  Returns:
   (condcov, sig21): final conditional covariance and sig21
  """
  diagsig22 = jnp.diagonal(sig22, axis1=1, axis2=2)

  def m_bmm(x):
    # return x
    return (
        jnp.broadcast_to(jnp.expand_dims(1 / diagsig22, axis=-1), x.shape) * x
    )

  a = m_bmm(sig22)
  b = m_bmm(jnp.expand_dims(sig21, axis=-1))
  z = jnp.zeros(b.shape, dtype=b.dtype)
  z_orig = z.at[:, -1, 0].set(1.0)
  z = m_bmm(z_orig)

  n = a.shape[1]
  orig_a = a
  mach_eps = jnp.finfo(orig_a.dtype).eps
  for pivot_row in range(n - 1):
    for row in range(pivot_row + 1, n):
      den = a[:, pivot_row, pivot_row]
      den = jnp.where(den == 0, mach_eps * orig_a[:, pivot_row, pivot_row], den)
      a = a.at[:, pivot_row, pivot_row].set(den)
      multiplier = a[:, row, pivot_row] / den
      multiplier = multiplier.reshape(-1, 1)
      a = a.at[:, row, pivot_row:].set(
          a[:, row, pivot_row:] - multiplier * a[:, pivot_row, pivot_row:]
      )
      b = b.at[:, row].set(b[:, row] - multiplier * b[:, pivot_row])
  batches = a.shape[0]

  def back_substitution(a, b):
    x = jnp.zeros((batches, n), dtype=a.dtype)
    k = n - 1
    den = a[:, k, k].reshape(-1, 1)
    temp = b[:, k] / den
    temp = temp.reshape(-1)
    x = x.at[:, k].set(temp)
    k = k - 1
    while k >= 0:
      first = a[:, k, k + 1 :].reshape((batches, 1, -1))
      second = x[:, k + 1 :].reshape((batches, -1, 1))
      second_term = jnp.matmul(first, second)
      temp = second_term.reshape(-1)
      den = a[:, k, k].reshape(-1)
      x = x.at[:, k].set((b[:, k].reshape(-1) - temp.reshape(-1)) / den)
      k = k - 1
    return x.reshape((batches, -1))

  k = n - 1
  a = a.at[:, k, k].set(
      jnp.where(a[:, k, k] == 0, mach_eps * orig_a[:, k, k], a[:, k, k])
  )
  x = back_substitution(a, b)
  y = back_substitution(a, z)

  psisig21 = (
      jnp.matmul(x.reshape((batches, 1, -1)), sig21.reshape((batches, -1, 1)))
      .squeeze(-1)
      .squeeze(-1)
  )
  condcov = d - psisig21
  temp = 2 * (
      jnp.matmul(
          y[:, :-1].reshape((batches, 1, -1)),
          sig21[:, :-1].reshape((batches, -1, 1)),
      ).reshape(-1)
      * sig21[:, -1]
  )
  temp += y[:, -1] * (sig21[:, -1] ** 2)
  temp = d - temp.reshape(-1) - condcov
  solve_quad_eq_vmap = jax.vmap(solve_quad_eq)
  temp = solve_quad_eq_vmap(
      y[:, -1],
      2
      * jnp.matmul(
          y[:, :-1].reshape((batches, 1, -1)),
          sig21[:, :-1].reshape((batches, -1, 1)),
      ).reshape(-1),
      temp - (1 - mach_eps) * d,
      sig21,
  )
  print("temp shape and val:", temp.shape, temp)
  tilde_sig21 = sig21.at[:, -1].set(temp.reshape(-1))
  print("sig21:", sig21)
  print("tilde_sig21:", tilde_sig21)
  b_hat = b.at[:, -1, 0].set(
      tilde_sig21[:, -1] / diagsig22[:, -1]
      - sig21[:, -1] / diagsig22[:, -1]
      + b[:, -1, 0]
  )
  x_hat = back_substitution(a, b_hat)
  # tilde_psisig21 = jnp.matmul(x_hat.reshape((batches, 1, -1)),
  #                             tilde_sig21.reshape((batches, -1, 1)))
  # tilde_psisig21 = tilde_psisig21.squeeze(-1).squeeze(-1)
  # tilde_condcov = d - tilde_psisig21
  tilde_condcov = mach_eps * d
  return jnp.where(
      condcov.reshape((-1, 1)) <= 0,
      jnp.concatenate((tilde_condcov.reshape((-1, 1)), x_hat), axis=1),
      jnp.concatenate((condcov.reshape((-1, 1)), x), axis=1),
  )


# pylint: disable=invalid-name
def get_band_pencil(M, n, b):
  """Get the band pencil to compute the inverse of banded matrices."""
  M_new = jnp.zeros((n, b + 1, b + 1))
  # print(M_new.shape)
  n = M_new.shape[0]

  for diag_num in range(b + 1):
    diag = M[:, diag_num]
    for offset in range(b + 1 - diag_num):
      diag_sliced = diag[offset : offset + n]
      # print(diag_sliced.shape)
      # print(M_new[offset,offset+diag_num].shape)

      M_new = M_new.at[:, offset, offset + diag_num].set(diag_sliced)
      M_new = M_new.at[:, offset + diag_num, offset].set(diag_sliced)
  return M_new


def bandedInv(sd, subdiags, eps):
  """Find the inverse of pd completion of this banded matrix.

  Given diagonal-Sd and subdiagonals-subDiags, find the inverse of pd completion
  of this banded matrix interms of Ldiag(D)L^T decomposition.

  Args:
    sd: diagonal of the banded matrix.
    subdiags: subdiagonal-band of the banded matrix.
    eps: eps to avoid divided-by-zero.

  Returns:
    Lsub and D, where Lsub-subdiagonals of L.
  """

  n = sd.shape[0]
  b = subdiags.shape[1]

  bandvecs = jnp.concatenate((sd.reshape(-1, 1), subdiags), axis=1)
  epsmat = jnp.zeros((b, b + 1), dtype=sd.dtype)
  epsmat = epsmat.at[:, 0].set(eps)
  bandwindows = jnp.concatenate((bandvecs, epsmat), axis=0)
  sig_full = get_band_pencil(bandwindows, n, b)
  sig22 = sig_full[:, 1:, 1:]
  sig21 = sig_full[:, 1:, 0]
  temp = new_genp(sig21, sig22, sd)
  condcov = temp[:, 0].reshape(-1)
  psi = temp[:, 1:]
  return psi.astype(sd.dtype), 1 / condcov.astype(sd.dtype)


def bandedMult(psi, D, vecv):
  """Multiply the banded matrix with a vector."""
  normalize = jnp.ones_like(D)
  b = psi.shape[1]
  normalize = jnp.sqrt(normalize)
  update = vecv * normalize
  for i in range(b):
    update = update.at[: -i - 1].set(
        update[: -i - 1] - vecv[i + 1 :] * psi[: -i - 1, i]
    )
  update = update * D
  vecv2 = update
  for i in range(b):
    update = update.at[i + 1 :].set(
        update[i + 1 :] - vecv2[: -i - 1] * psi[: -i - 1, i]
    )
  update = update * normalize
  return update


def _update_nu_banded(updates, nu_e, nu_d, beta2):
  """Update the statistics of the banded covariance matrix."""
  nu_d = jax.tree_map(
      lambda g, t: (1 - beta2) * (g**2) + beta2 * t, updates, nu_d
  )

  def update_band(g, band, b):
    for i in range(b):
      band = band.at[: -(i + 1), i].set(
          (1 - beta2) * (g[: -(i + 1)] * g[i + 1 :])
          + beta2 * band[: -(i + 1), i]
      )
    return band

  nu_e = jax.tree_map(
      lambda g, t: update_band(g, t, t.shape[-1]), updates, nu_e
  )
  return nu_e, nu_d


def bandedUpdates(Sd, subDiags, eps, mu):
  Sd = Sd + eps
  psi, D = bandedInv(Sd, subDiags, eps)
  return bandedMult(psi, D, mu)


# pylint: enable=invalid-name


def _banded_grafting(
    updates,
    updates_hat,
    nu_hat_d,
    mu_hat_flat,
    diag,
    beta2,
    graft_type,
    graft_eps,
):
  """Grafting when using banded approximation of inverse convariance."""
  if graft_type == 1:
    adam_updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + graft_eps), mu_hat_flat, nu_hat_d
    )
    updates = jax.tree_map(
        lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-12)) * u,
        updates,
        adam_updates,
    )
  elif graft_type == 2:
    updates_hat = jax.tree_map(
        lambda g: g / (jnp.linalg.norm(g) + 1e-16), updates_hat
    )
    diag = tridiag_optimizers._update_moment(updates_hat, diag, beta2, 2)  # pylint: disable=protected-access
    updates_hat = jax.tree_map(
        lambda g, d: g / (jnp.sqrt(d) + graft_eps), updates_hat, diag
    )
    updates = jax.tree_map(
        lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-12)) * u,
        updates,
        updates_hat,
    )
  return updates


# Preconditioners for Banded TriDiagonal


class PreconditionBandedDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""

  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  diag: optax.Updates


def precondition_by_bds(
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    graft_eps: float = 1e-8,
    graft_type: int = 0,
    transpose: bool = True,
    b: int = 3,
    debias: bool = True,
) -> optax.GradientTransformation:
  """Preconditioner for Banded TriDiagonal Approximation."""

  def init_fn(params):
    diag = None
    if graft_type == 2:
      diag = jax.tree_map(
          lambda g: jnp.zeros(len(g.reshape(-1)), dtype=g.dtype), params
      )
    return PreconditionBandedDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_map(
            lambda g: jnp.zeros((len(g.reshape(-1)), b), dtype=g.dtype), params
        ),
        nu_d=jax.tree_map(
            lambda g: jnp.zeros(len(g.reshape(-1)), dtype=g.dtype), params
        ),
        diag=diag,
    )

  def update_fn(updates, state, params):
    del params
    diag = state.diag
    updates_hat = jax.tree_map(
        lambda g: g.T.reshape(-1) if transpose else g.reshape(-1), updates
    )
    # pylint: disable=protected-access
    mu = tridiag_optimizers._update_moment(updates, state.mu, beta1, 1)
    nu_e, nu_d = _update_nu_banded(updates_hat, state.nu_e, state.nu_d, beta2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = (
        mu
        if not debias
        else tridiag_optimizers._bias_correction(mu, beta1, count)
    )
    nu_hat_e = (
        nu_e
        if not debias
        else tridiag_optimizers._bias_correction(nu_e, beta2, count)
    )
    nu_hat_d = (
        nu_d
        if not debias
        else tridiag_optimizers._bias_correction(nu_d, beta2, count)
    )
    # pylint: enable=protected-access

    mu_hat_flat = jax.tree_map(
        lambda m: m.T.reshape(-1) if transpose else m.reshape(-1), mu_hat
    )
    updates = jax.tree_map(
        lambda d, e, g: bandedUpdates(d, e, eps, g),
        nu_hat_d,
        nu_hat_e,
        mu_hat_flat,
    )

    # Grafting
    updates = _banded_grafting(
        updates,
        None,  # updates_step_hat
        nu_hat_d,
        mu_hat_flat,
        diag,
        beta2,
        graft_type,
        graft_eps,
    )

    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T
        if transpose
        else mf.reshape(m.shape),
        updates,
        mu_hat,
    )
    return updates, PreconditionBandedDiagonalState(
        count=count, mu=mu, nu_e=nu_e, nu_d=nu_d, diag=diag
    )

  return optax.GradientTransformation(init_fn, update_fn)


def bds(
    learning_rate: ScalarOrSchedule,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-8,
    graft_eps: float = 1e-8,
    graft_type: int = 0,
    weight_decay: float = 0.0,
    b: int = 3,
    transpose: bool = True,
) -> optax.GradientTransformation:
  """Banded TriDiagonal Optimizer."""
  return optax.chain(
      precondition_by_bds(
          beta1=beta1,
          beta2=beta2,
          eps=eps,
          graft_type=graft_type,
          graft_eps=graft_eps,
          b=b,
          transpose=transpose,
      ),
      optax.add_decayed_weights(weight_decay),
      tridiag_optimizers.scale_by_learning_rate(learning_rate),
  )


# Preconditioners for Banded Skip Combination


class PreconditionBandedDiagonalSkipCombinationState(NamedTuple):
  """State for the Adam preconditioner."""

  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_step_e: optax.Updates
  nu_step_d: optax.Updates
  nu_skip_e: optax.Updates
  nu_skip_d: optax.Updates
  diag: optax.Updates


def precondition_by_bskcs(
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    graft_eps: float = 1e-8,
    graft_type: int = 0,
    transpose: bool = True,
    b: int = 3,
    debias: bool = True,
) -> optax.GradientTransformation:
  """Preconditioner for Banded Skip Combination."""

  def init_fn(params):
    diag = None
    if graft_type == 2:
      diag = jax.tree_map(
          lambda g: jnp.zeros(len(g.reshape(-1)), dtype=g.dtype), params
      )
    return PreconditionBandedDiagonalSkipCombinationState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_step_e=jax.tree_map(
            lambda g: jnp.zeros((len(g.reshape(-1)), b), dtype=g.dtype), params
        ),
        nu_step_d=jax.tree_map(
            lambda g: jnp.zeros(len(g.reshape(-1)), dtype=g.dtype), params
        ),
        nu_skip_e=jax.tree_map(
            lambda g: jnp.zeros((len(g.reshape(-1)), b), dtype=g.dtype), params
        ),
        nu_skip_d=jax.tree_map(
            lambda g: jnp.zeros(len(g.reshape(-1)), dtype=g.dtype), params
        ),
        diag=diag,
    )

  def update_fn(updates, state, params):
    del params
    diag = state.diag
    # pylint: disable=protected-access
    mu = tridiag_optimizers._update_moment(updates, state.mu, beta1, 1)
    count = state.count + jnp.array(1, dtype=jnp.int32)

    updates_step_hat = jax.tree_map(
        lambda g: g.T.reshape(-1) if transpose else g.reshape(-1), updates
    )
    nu_step_e, nu_step_d = _update_nu_banded(
        updates_step_hat, state.nu_step_e, state.nu_step_d, beta2
    )
    updates_skip_hat = jax.tree_map(
        lambda g: tridiag_optimizers._sqrt_permutation(g.T).reshape(-1)
        if transpose
        else tridiag_optimizers._sqrt_permutation(g).reshape(-1),
        updates,
    )
    nu_skip_e, nu_skip_d = _update_nu_banded(
        updates_skip_hat, state.nu_skip_e, state.nu_skip_d, beta2
    )

    # Update the hat that will later store the real update (the inverse matrix)
    mu_hat = (
        mu
        if not debias
        else tridiag_optimizers._bias_correction(mu, beta1, count)
    )
    nu_step_hat_e = (
        nu_step_e
        if not debias
        else tridiag_optimizers._bias_correction(nu_step_e, beta2, count)
    )
    nu_step_hat_d = (
        nu_step_d
        if not debias
        else tridiag_optimizers._bias_correction(nu_step_d, beta2, count)
    )
    nu_skip_hat_e = (
        nu_skip_e
        if not debias
        else tridiag_optimizers._bias_correction(nu_skip_e, beta2, count)
    )
    nu_skip_hat_d = (
        nu_skip_d
        if not debias
        else tridiag_optimizers._bias_correction(nu_skip_d, beta2, count)
    )

    # Compute the inverse of Each bandeddiag approximation.
    mu_step_hat_flat = jax.tree_map(
        lambda m: m.T.reshape(-1) if transpose else m.reshape(-1), mu_hat
    )
    updates_step = jax.tree_map(
        lambda d, e, g: bandedUpdates(d, e, eps, g),
        nu_step_hat_d,
        nu_step_hat_e,
        mu_step_hat_flat,
    )
    mu_skip_hat_flat = jax.tree_map(
        lambda m: tridiag_optimizers._sqrt_permutation(m.T).reshape(-1)
        if transpose
        else tridiag_optimizers._sqrt_permutation(m).reshape(-1),
        mu_hat,
    )
    updates_skip = jax.tree_map(
        lambda d, e, g: bandedUpdates(d, e, eps, g),
        nu_skip_hat_d,
        nu_skip_hat_e,
        mu_skip_hat_flat,
    )

    # Taking average before grafting
    updates_skip = jax.tree_map(
        # CAREFUL that update_step still in transpose setting in this step even
        # if transpose==True
        # pylint: disable=g-long-ternary
        lambda mf, m: tridiag_optimizers._sqrt_reversed_permutation(
            mf.reshape(m.T.shape)
        ).reshape(-1)
        if transpose
        else tridiag_optimizers._sqrt_reversed_permutation(
            mf.reshape(m.shape)
        ).reshape(-1),
        updates_skip,
        mu_hat,
    )
    # pylint: enable=protected-access
    updates = jax.tree_map(
        lambda ust, usk: 0.5 * (ust + usk), updates_step, updates_skip
    )

    # Grafting
    updates = _banded_grafting(
        updates,
        None,  # updates_step_hat
        nu_step_hat_d,
        mu_step_hat_flat,
        diag,
        beta2,
        graft_type,
        graft_eps,
    )

    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T
        if transpose
        else mf.reshape(m.shape),
        updates,
        mu_hat,
    )
    return updates, PreconditionBandedDiagonalSkipCombinationState(
        count=count,
        mu=mu,
        nu_step_e=nu_step_e,
        nu_step_d=nu_step_d,
        nu_skip_e=nu_skip_e,
        nu_skip_d=nu_skip_d,
        diag=diag,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def bskcs(
    learning_rate: ScalarOrSchedule,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-8,
    graft_eps: float = 1e-8,
    graft_type: int = 0,
    weight_decay: float = 0.0,
    b: int = 3,
    transpose: bool = True,
) -> optax.GradientTransformation:
  """Banded Skip Combination Optimizer."""
  return optax.chain(
      precondition_by_bskcs(
          beta1=beta1,
          beta2=beta2,
          eps=eps,
          graft_type=graft_type,
          graft_eps=graft_eps,
          b=b,
          transpose=transpose,
      ),
      optax.add_decayed_weights(weight_decay),
      tridiag_optimizers.scale_by_learning_rate(learning_rate),
  )
