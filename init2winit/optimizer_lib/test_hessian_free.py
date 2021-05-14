# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

"""Tests for losses.py.

"""

from absl.testing import absltest
from init2winit.optimizer_lib import hessian_free
from init2winit.optimizer_lib.hessian_free import relative_per_iteration_progress_test
from init2winit.optimizer_lib.hessian_free import residual_norm_test
import numpy as np


def get_pd_mat(mat):
  """Returns a positive-definite matrix."""
  n = mat.shape[0]
  return mat @ np.transpose(mat) / n**2 + np.eye(n)


class HessianFreeTest(absltest.TestCase):
  """Tests for hessian_free.py."""

  def test_residual_norm_test(self):
    """Tests residual norm test."""
    rs_norm = 1e-6
    self.assertEqual(residual_norm_test(0, rs_norm, 0., [], 1e-2), 1)
    self.assertEqual(residual_norm_test(0, rs_norm, 0., [], 1e-4), 0)

  def test_relative_per_iteration_progress_test(self):
    """Tests relative_per_iteration_progress_test."""
    obj_value = -10
    obj_values = -15 * np.ones(10)
    tol = 1e-3
    step = 15

    convergd = relative_per_iteration_progress_test(step, 0, obj_value,
                                                    obj_values, tol)
    self.assertEqual(convergd, 1.0)

  def test_conjgrad(self):
    """Tests conjugate gradient method."""
    n = 5
    mat = get_pd_mat(
        np.array(
            [[2., 4., 5., 2., 8.],
             [0., 4., 3., 5., 3.],
             [-2., -2., 9., -2., -6.],
             [4., 1., -11., 1., 4.],
             [-5., 4., -9., 3., -2.]]))
    b = np.array([-3, 2, 0, 3, -4])
    x0 = np.ones(n)

    test_matmul_fn = lambda x: mat @ x
    x = hessian_free.mf_conjgrad_solver(test_matmul_fn, b, x0, n, 1e-6, 10,
                                        None, 'residual_norm_test')
    self.assertAlmostEqual(np.linalg.norm(test_matmul_fn(x) - b), 0, places=3)

  def test_conjgrad_preconditioning(self):
    """Tests conjugate gradient method with preconditioning."""
    n = 5
    mat = get_pd_mat(
        np.array(
            [[2., 4., 5., 2., 8.],
             [0., 4., 3., 5., 3.],
             [-2., -2., 9., -2., -6.],
             [4., 1., -11., 1., 4.],
             [-5., 4., -9., 3., -2.]]))
    precond_mat = get_pd_mat(
        np.array(
            [[4., 2., 0., 2., 4.],
             [-2., 4., 4., 2., 6.],
             [4., 4., -8., -2., -4.],
             [-2., 2., 4., 0., -2.],
             [2., 2., -6., 4., 0.]]))
    b = np.array([-3, 2, 0, 3, -4])
    x0 = np.ones(n)

    test_matmul_fn = lambda x: mat @ x
    test_precond_fn = lambda x: precond_mat @ x
    x = hessian_free.mf_conjgrad_solver(test_matmul_fn, b, x0, n, 1e-6, 10,
                                        test_precond_fn, 'residual_norm_test')
    self.assertAlmostEqual(np.linalg.norm(test_matmul_fn(x) - b), 0, places=3)

  def test_conjgrad_martens_termination_criterion(self):
    """Tests conjugate gradient method with martens termination criterion."""
    n = 500
    mat = get_pd_mat(
        np.array([[((i + j) % n) for j in range(n)] for i in range(n)]))
    b = np.linspace(1, n, n) / n
    x0 = np.zeros(n)

    test_mvm_fn = lambda x: mat @ x

    x = hessian_free.mf_conjgrad_solver(
        test_mvm_fn, b, x0, n, 1e-6, 500, None,
        'relative_per_iteration_progress_test')
    f_value = np.dot(x, test_mvm_fn(x) - 2 * b) / 2
    self.assertAlmostEqual(f_value, -0.223612323, places=8)


if __name__ == '__main__':
  absltest.main()
