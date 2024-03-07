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

"""Tridiagonal Matrix Optimizer.

This is based on https://arxiv.org/abs/2311.10085 and
https://github.com/devvrit/Autoencoder-on-MNIST-in-Pytorch.
"""

from typing import NamedTuple, Union
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

ScalarOrSchedule = Union[float, optax.Schedule]


def getl1norm_tds(sd, se):
  """Get l1norm of tridiagonal matrix given diagonal and off-diag statistics."""
  temp = sd
  temp = temp.at[:-1].set(sd[:-1] + jnp.abs(se))
  temp = temp.at[1:].set(sd[1:] + jnp.abs(se))
  return jnp.max(temp)


def ldl2tridiag(lsub, d):
  """Use L and D to compute tridiag=LDL^T.

  Args:
   lsub: L is bidiagonal, where diag(L)=1. So providing subdiagonal of L.
   d: inverse conditional covariance. Used as "D" in LDL^T.

  Returns:
   xd, xe -- diagonal and subdiagonal preconditioner.
  """
  xd = jnp.zeros_like(d)
  xd = xd.at[1:].set(d[1:] + lsub * lsub * d[:-1])
  xd = xd.at[0].set(d[0])
  xe = lsub * d[:-1]
  return xd, xe


def tridiag_kfac(sd, se, eps, min_eps=1e-24, relative_epsilon=True):
  """Tridiagonal approximation.

  Given diagonal (sd) and subdiagonal(se), find the inverse
  of PD completion of this tridiag matrix using LDL^T
  Ref: https://arxiv.org/pdf/1503.05671.pdf, Sec 4.3.

  Args:
   sd: Diagonal statistics.
   se: Subdiagonal statistcs.
   eps: eps added to the diagonal statistic.
   min_eps: minimum eps to be added to the diag statistics.
   relative_epsilon: if True, use l1norm normalization else use eps directly.

  Returns:
   xd, xe -- diagonal and subdiagonal preconditioner
  """
  if relative_epsilon:
    l1norm = getl1norm_tds(sd, se)
    sd = sd + jnp.maximum(eps * l1norm, min_eps)
  else:
    sd = sd + eps
  psi = se / sd[1:]
  cond_cov = jnp.zeros_like(sd)
  cond_cov = cond_cov.at[:-1].set(sd[:-1] - se * (se / sd[1:]))
  cond_cov = cond_cov.at[-1].set(sd[-1])
  d = 1 / cond_cov
  mask1 = cond_cov[:-1] <= 0.0
  mask2 = cond_cov <= 0.0
  psi = jnp.where(mask1, 0, psi)
  d = jnp.where(mask2, 1 / sd, d)
  lsub = -psi
  return ldl2tridiag(lsub, d)


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
  )


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def _update_nu(updates, nu_e, nu_d, beta2):
  """Compute the exponential moving average of the tridiagonal structure of the moment."""
  nu_d = jax.tree_map(
      lambda g, t: (1 - beta2) * (g**2) + beta2 * t, updates, nu_d
  )
  nu_e = jax.tree_map(
      lambda g, t: (1 - beta2) * (g[:-1] * g[1:]) + beta2 * t, updates, nu_e
  )
  return nu_e, nu_d


def scale_by_learning_rate(
    learning_rate: ScalarOrSchedule, flip_sign: bool = True
) -> optax.GradientTransformation:
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


class PreconditionTriDiagonalState(NamedTuple):
  """State for the Adam preconditioner."""

  count: chex.Array  # shape=(), dtype=jnp.int32
  mu: optax.Updates
  nu_e: optax.Updates
  nu_d: optax.Updates
  diag: optax.Updates


# Pre conditioning by tri diagonal structure
def precondition_by_tds(
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    graft_type: int = 0,
    graft_eps: float = 1e-8,
    transpose: bool = True,
    relative_epsilon: bool = True,
    debias: bool = True,
) -> optax.GradientTransformation:
  """Preconditioning  by tri-diagonal structure."""

  def init_fn(params):
    diag = None
    if graft_type == 4:  # Normalized rmsprop grafting
      diag = jax.tree_map(
          lambda g: jnp.zeros(
              len(g.reshape(-1)), dtype=g.dtype  # pylint: disable=g-long-lambda
          ),
          params,
      )
    return PreconditionTriDiagonalState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree_map(jnp.zeros_like, params),
        nu_e=jax.tree_map(
            lambda g: jnp.zeros(
                len(g.reshape(-1)) - 1,  # pylint: disable=g-long-lambda
                dtype=g.dtype,
            ),
            params,
        ),
        nu_d=jax.tree_map(
            lambda g: jnp.zeros(
                len(g.reshape(-1)),  # pylint: disable=g-long-lambda
                dtype=g.dtype,
            ),
            params,
        ),
        diag=diag,
    )

  def update_fn(updates, state, params=None):
    del params
    diag = state.diag
    updates_hat = jax.tree_map(
        lambda g: g.T.reshape(-1) if transpose else g.reshape(-1), updates
    )
    mu = _update_moment(updates, state.mu, beta1, 1)
    nu_e, nu_d = _update_nu(updates_hat, state.nu_e, state.nu_d, beta2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _bias_correction(mu, beta1, count) if debias else mu
    nu_hat_e = _bias_correction(nu_e, beta2, count) if debias else nu_e
    nu_hat_d = _bias_correction(nu_d, beta2, count) if debias else nu_d

    temp = jax.tree_map(
        lambda d, e: tridiag_kfac(d, e, eps, relative_epsilon=relative_epsilon),  # pylint: disable=g-long-lambda
        nu_hat_d,
        nu_hat_e,
    )
    pre_d = jax.tree_map(lambda h, g: g[0], nu_hat_d, temp)
    pre_e = jax.tree_map(lambda h, g: g[1], nu_hat_e, temp)

    mu_hat_flat = jax.tree_map(
        lambda m: m.T.reshape(-1) if transpose else m.reshape(-1), mu_hat
    )
    # Multiply gradient with diagonal
    updates = jax.tree_map(lambda m, a: m * a, mu_hat_flat, pre_d)
    # updates[i] = updates[i] + gradient[i-1]*pre_e[i], for i>0
    updates = jax.tree_map(
        lambda u, m, a: u.at[1:].set(u[1:] + m[:-1] * a),
        updates,
        mu_hat_flat,
        pre_e,
    )
    # updates[i] = updates[i] + gradient[i+1]*pre_e[i], for i<n-1
    updates = jax.tree_map(
        lambda u, m, a: u.at[:-1].set(u[:-1] + m[1:] * a),
        updates,
        mu_hat_flat,
        pre_e,
    )

    # Get adam updates for biases
    adam_updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + graft_eps), mu_hat_flat, nu_hat_d
    )
    if graft_type == 1:
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-12))
          * u,
          updates,
          adam_updates,
      )
    elif graft_type == 2:
      # perform sgd grafting
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-12))
          * u,
          updates,
          updates_hat,
      )
    elif graft_type == 3:
      # perform momentum grafting
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-12))
          * u,
          updates,
          mu_hat_flat,
      )
    elif graft_type == 4:
      # perform normalized rmsprop grafting
      updates_hat = jax.tree_map(
          lambda g: g / (jnp.linalg.norm(g) + 1e-16), updates_hat
      )
      diag = _update_moment(updates_hat, diag, beta2, 2)
      updates_hat = jax.tree_map(
          lambda g, d: g / (jnp.sqrt(d) + graft_eps), updates_hat, diag
      )
      updates = jax.tree_map(
          lambda u, au: (jnp.linalg.norm(au) / (jnp.linalg.norm(u) + 1e-16))
          * u,
          updates,
          updates_hat,
      )

    # reshape them to the original param shapes
    updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T  # pylint: disable=g-long-lambda
        if transpose
        else mf.reshape(m.shape),
        updates,
        mu_hat,
    )
    adam_updates = jax.tree_map(
        lambda mf, m: mf.reshape(m.T.shape).T  # pylint: disable=g-long-lambda
        if transpose
        else mf.reshape(m.shape),
        adam_updates,
        mu_hat,
    )
    updates = jax.tree_map(
        lambda u, au: au if len(u.shape) <= 1 else u, updates, adam_updates
    )
    return updates, PreconditionTriDiagonalState(
        count=count, mu=mu, nu_e=nu_e, nu_d=nu_d, diag=diag
    )

  return optax.GradientTransformation(init_fn, update_fn)


def tds(
    learning_rate: ScalarOrSchedule,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-8,
    graft_type: int = 0,
    graft_eps: float = 1e-8,
    weight_decay: float = 0.0,
    transpose: bool = True,
    relative_epsilon: bool = True,
) -> optax.GradientTransformation:
  return optax.chain(
      precondition_by_tds(
          beta1=beta1,
          beta2=beta2,
          eps=eps,
          graft_type=graft_type,
          graft_eps=graft_eps,
          transpose=transpose,
          relative_epsilon=relative_epsilon,
      ),
      optax.add_decayed_weights(weight_decay),
      scale_by_learning_rate(learning_rate),
  )


# pylint: disable=invalid-name
def _sqrt_perm_indices(N):
  d = int(np.sqrt(N))
  m = N // d
  # pylint: disable=g-complex-comprehension
  return [i + j * d for i in range(d) for j in range(m + 1) if i + j * d < N]


def _sqrt_reversed_permu_indices(N):
  perm_indices = _sqrt_perm_indices(N)
  reversed_permu_indices = [0] * N
  for i, idx in enumerate(perm_indices):
    reversed_permu_indices[idx] = i
  return reversed_permu_indices


def _sqrt_reversed_permutation(g):
  v = g.T
  N = v.shape[0]
  return v[np.array(_sqrt_reversed_permu_indices(N))].T


# pylint: enable=invalid-name
