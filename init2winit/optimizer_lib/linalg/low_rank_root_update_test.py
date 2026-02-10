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

"""Test for computing the sqrt and inverse sqrt of a matrix."""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.optimizer_lib.linalg import low_rank_root_update
import jax
import numpy as np
import scipy.stats


def _small_perturbation(n: int, gamma: float,
                        rng: np.random.RandomState) -> np.ndarray:
  """Returns a vector of absolute values ofnormally distributed values with standard deviation gamma."""
  s = gamma*np.abs(rng.normal(size=n))
  return s


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
    rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
  """Returns a random SVD decomposition with singular values in [γ, 1]."""
  # sample a uniformly random orthogonal matrix.
  v = scipy.stats.ortho_group.rvs(n, random_state=rng)
  s = _random_singular_values(n, gamma, rng)
  return s, v


@jax.jit
def _update_sqrt(x, ix, g):
  ra_size = np.where(
      x.shape[-1] < 64, x.shape[-1], np.minimum(x.shape[-1] // 12, 64)
  )
  rank_array = np.zeros(ra_size)
  return low_rank_root_update.low_rank_root_update(
      x, ix, g, rank_array, 1e-6, 2
  )


class InvSquareRootTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {  # pylint:disable=g-complex-comprehension
          'testcase_name': f'n={n}',
          'n': n,  # pylint: disable=undefined-variable
          'p': p,  # pylint: disable=undefined-variable
      }
      for n in [2, 31]
      for p in [2]
  )
  def test_random_matrix(self, n, p):
    rng = np.random.RandomState(seed=42)

    sigma = 1e-2  # smallest singular value of test matrix
    s, v = _random_svd(n, sigma, rng)
    s = s.astype(np.float64)
    v = v.astype(np.float64)
    q = _small_perturbation(n, 1e-4, rng)
    q = np.diag(q.astype(np.float64))
    a_sqrt = (v * s**(1 / p)) @ v.T
    a_isqrt = (v * s**(-1 / p)) @ v.T
    exact = (v * (s + q**2)**(-1 / p)) @ v.T
    ans = _update_sqrt(a_sqrt, a_isqrt, q)[1]
    ans = np.array(ans).astype(np.float64)
    error = np.linalg.norm(ans - exact, 2) / np.linalg.norm(exact, 2)
    kappa = 1 / p / sigma
    expected_error = 3 * kappa * np.finfo(np.float32).eps
    self.assertLessEqual(error, expected_error)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', False)
  absltest.main()
