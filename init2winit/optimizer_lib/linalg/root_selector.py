# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Depending on a variable, selects whether to use exact or approximate root function."""

import functools
from typing import Tuple

import chex
from init2winit.optimizer_lib.linalg import low_rank_root_update
from init2winit.optimizer_lib.linalg import pth_inv_root_rmn
from jax import lax
import numpy as np


def root_selector(
    x: chex.Array,
    sx: chex.Array,
    isx: chex.Array,
    up: chex.Array,
    p: int,
    eps: float,
    exact_root: bool,
    rank_estimate: int,
    block_krylov_dim_multiplier: int = 2,
    stable_iter: bool = False,
    unroll: bool = False,
    verbose: bool = True,
) -> Tuple[chex.Array, chex.Array]:
  """Returns |X|^{1/p} and |X|⁻¹ᐟᵖ.

  Args:
    x: Input matrix must be SPD with eigenvalues >= float32 epsilon.
    sx: Old sqrt of the matrix X.
    isx: Old inverse sqrt of the matrix X.
    up: Update to the matrix of the form update @ update.T
    p: Exponent.
    eps: small constant to avoid numerical issues with Lyapunov solver.
    exact_root: If True, solve the root exactly, otherwise.
    rank_estimate: Rank estimate of the update.
    block_krylov_dim_multiplier: Multiplier for the block krylov dimension over
      the rank estimate.
    stable_iter: Whether to use the stable iteration for the inner loop.
    unroll: Whether to unroll the loop over iterations.
    verbose: Whether to log some information about the iteration, including the
      coefficients `a` for each iteration.

  Returns:
    An approximation of |X|⁻¹ᐟᵖ.
  """
  f_er = functools.partial(
      pth_inv_root_rmn.pth_inv_root_rmn,
      fast_root=True,
      precision="float32",
      stable_iter=stable_iter,
      unroll=unroll,
      verbose=verbose,
  )

  def _exact_root():
    return f_er(x, p)

  rank_array = np.zeros(np.where(
      x.shape[-1] < 64,
      x.shape[-1],
      np.minimum(x.shape[-1] // 8, rank_estimate),
  ))
  f_ar = functools.partial(
      low_rank_root_update.low_rank_root_update,
      rank_array=rank_array,
      eps=eps,
      block_krylov_dim_multiplier=block_krylov_dim_multiplier,
      verbose=verbose,
  )
  def _approx_root():
    return f_ar(sx, isx, up)

  return lax.cond(exact_root, _exact_root, _approx_root)
