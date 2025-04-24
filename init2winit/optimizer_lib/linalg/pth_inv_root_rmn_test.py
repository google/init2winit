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

"""Tests for computing the pth root and inverse pth root of a matrix."""

import functools
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.optimizer_lib.linalg import pth_inv_root_rmn
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


def _random_singular_values(n: int, gamma: float,
                            rng: np.random.RandomState) -> np.ndarray:
  """Returns n random singular values in [γ, 1]."""
  s = gamma**rng.random((n,))  # log of singular values uniformly distributed
  if n > 0:
    s[0] = gamma
  if n > 1:
    s[1] = 1
  return s


def _random_svd(
    n: int, gamma: float,
    rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns a random SVD decomposition with singular values in [γ, 1]."""
  # sample a uniformly random orthogonal matrix.
  u = scipy.stats.ortho_group.rvs(n, random_state=rng)
  v = scipy.stats.ortho_group.rvs(n, random_state=rng)
  s = _random_singular_values(n, gamma, rng)
  return u, s, v


@functools.partial(jax.jit, static_argnames=['p'])
def _root(x, p):
  return pth_inv_root_rmn.pth_inv_root_rmn(x, p)[1]


class PthInvRootTest(parameterized.TestCase):

  @parameterized.named_parameters({  # pylint:disable=g-complex-comprehension
      'testcase_name': f'n={n}_p={p}',
      'n': n,  # pylint: disable=undefined-variable
      'p': p,  # pylint: disable=undefined-variable
  } for n in [2, 31] for p in [1, 2, 3, 4, 6, 8])
  def test_zero_matrix(self, n, p):
    a = jnp.zeros((n, n), dtype=jnp.float32)
    x = _root(a, p)
    self.assertFalse(jnp.any(jnp.logical_or(jnp.isnan(x), jnp.isinf(x))))

  @parameterized.named_parameters(
      {  # pylint:disable=g-complex-comprehension
          'testcase_name': f'n={n}_p={p}_c={c}',
          'n': n,  # pylint: disable=undefined-variable
          'p': p,  # pylint: disable=undefined-variable
          'c': c,  # pylint: disable=undefined-variable
      }
      for n in [2, 31]
      for p in [1, 2, 3, 4, 6, 8]
      for c in [-1, 0, 1]
  )
  def test_random_matrix(self, n, p, c):
    rng = np.random.RandomState(seed=42)

    for k in range(6):
      sigma = 10**(-k - 1)  # smallest singular value of test matrix
      _, s, v = _random_svd(n, sigma, rng)
      s = s.astype(np.float64) * (1e6 ** c)  # c tests different matrix scalings
      v = v.astype(np.float64)
      a = jnp.array((v * s) @ v.T, jnp.float32)
      exact = (v * s**(-1 / p)) @ v.T
      x = _root(a, p)
      x = np.array(x).astype(np.float64)
      error = np.linalg.norm(x - exact, 2) / np.linalg.norm(exact, 2)

      kappa = 1 / p / sigma  # relative condition number
      expected_error = 3 * kappa * np.finfo(np.float32).eps
      self.assertLessEqual(error, expected_error)

  @parameterized.named_parameters(
      {  # pylint:disable=g-complex-comprehension
          'testcase_name': f'n={n}_p={p}',
          'n': n,  # pylint: disable=undefined-variable
          'p': p,  # pylint: disable=undefined-variable
      }
      for n in [2, 31]
      for p in [1, 2, 3, 4, 6, 8]
  )
  def test_singular_matrix(self, n, p):
    rng = np.random.RandomState(seed=42)

    for k in range(6):
      sigma = 10**(-k - 1)  # smallest singular value of test matrix
      _, s, v = _random_svd(n, sigma, rng)
      s = s.astype(np.float64)
      v = v.astype(np.float64)
      a = jnp.array((v * s) @ v.T, jnp.float32)
      exact = (v * s**(-1 / p)) @ v.T
      x = _root(a, p)
      x = np.array(x).astype(np.float64)
      error = np.linalg.norm(x - exact, 2) / np.linalg.norm(exact, 2)

      kappa = 1 / p / sigma  # relative condition number
      expected_error = 6 * kappa * np.finfo(np.float32).eps
      self.assertLessEqual(error, expected_error)

  @parameterized.named_parameters({  # pylint:disable=g-complex-comprehension
      'testcase_name': '_p={}'.format(p),
      'p': p  # pylint: disable=undefined-variable
  } for p in [1, 2, 3, 4, 6, 8])
  def test_random_diagonal_matrix(self, p):
    n = 16
    eps = np.finfo(np.float32).eps
    rng = np.random.RandomState(seed=37)
    s = _random_singular_values(n, eps, rng)
    exact = s.astype(np.float64)**(-1 / p)
    x = _root(np.diag(s).astype(np.float32), p).astype(np.float64)
    # since the matrix is diagonal, the error should be small despite the
    # large condition number
    error = np.abs(np.diagonal(x) - exact) / exact
    expected_error = 300 * eps
    self.assertLessEqual(np.max(error), expected_error)
    # off diagonal entries should be exactly zero
    x = x - np.diag(np.diagonal(x))
    self.assertEqual(np.max(np.abs(x.flatten())), 0)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', False)
  absltest.main()
