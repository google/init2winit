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

"""Tests for losses.py.

"""

import copy
from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
from init2winit.model_lib import models
from init2winit.optimizer_lib.hessian_free import cg_backtracking
from init2winit.optimizer_lib.hessian_free import gvp
from init2winit.optimizer_lib.hessian_free import hessian_free
from init2winit.optimizer_lib.hessian_free import mf_conjgrad_solver
from init2winit.optimizer_lib.hessian_free import relative_per_iteration_progress_test
from init2winit.optimizer_lib.hessian_free import residual_norm_test
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
from optax import apply_updates


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
    x_arr, x_arr_idx = mf_conjgrad_solver(test_matmul_fn, b, x0, n, 1e-6, 10,
                                          None, 'residual_norm_test')
    x = x_arr[x_arr_idx]
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
    x_arr, x_arr_idx = mf_conjgrad_solver(test_matmul_fn, b, x0, n, 1e-6, 10,
                                          test_precond_fn, 'residual_norm_test')
    x = x_arr[x_arr_idx]
    self.assertAlmostEqual(np.linalg.norm(test_matmul_fn(x) - b), 0, places=3)

  def test_conjgrad_martens_termination_criterion(self):
    """Tests conjugate gradient method with martens termination criterion."""
    n = 500
    mat = get_pd_mat(
        np.array([[((i + j) % n) for j in range(n)] for i in range(n)]))
    b = np.linspace(1, n, n) / n
    x0 = np.zeros(n)

    test_mvm_fn = lambda x: mat @ x

    x_arr, x_arr_idx = mf_conjgrad_solver(
        test_mvm_fn, b, x0, n, 1e-6, 500, None,
        'relative_per_iteration_progress_test')
    x = x_arr[x_arr_idx]
    f_value = np.dot(x, test_mvm_fn(x) - 2 * b) / 2
    self.assertAlmostEqual(f_value, -0.223612576, places=5)

  def test_cg_backtracking(self):
    """Tests CG backtracking."""

    model_str = 'autoencoder'
    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)

    loss = 'sigmoid_binary_cross_entropy'
    metrics = 'binary_autoencoder_metrics'

    input_shape = (2, 2, 1)
    output_shape = (4,)

    hps = copy.copy(model_hps)
    hps.update({
        'hid_sizes': [2],
        'activation_function': 'id',
        'input_shape': input_shape,
        'output_shape': output_shape
    })

    model = model_cls(hps, {}, loss, metrics)

    inputs = jnp.array([[[1, 0], [1, 1]], [[1, 0], [0, 1]]])
    targets = inputs.reshape(tuple([inputs.shape[0]] + list(output_shape)))
    batch = {'inputs': inputs, 'targets': targets}

    def forward_fn(variables, inputs):
      logits = model.flax_module.apply(variables, inputs, train=False)
      return logits

    def opt_cost(params):
      return model.loss_fn(forward_fn(params, inputs), targets)

    params = {
        'Dense_0': {
            'kernel': jnp.array([[-1., 2.], [2., 0.], [-1., 3.], [-2., 2.]]),
            'bias': jnp.array([0., 0.])
        },
        'Dense_1': {
            'kernel': jnp.array([[4., 2., -2., 4.], [-3., 1., 2., -4.]]),
            'bias': jnp.array([0., 0., 0., 0.])
        }
    }
    unravel_fn = ravel_pytree(params)[1]

    p1 = np.array([
        0.5, 0.2, 0.1, -0.4, -0.6, 0.4, 0.6, -0.7, 0.0, 0.5, -0.7, 0.2, 0.1,
        -0.2, 0.4, -0.6, -0.8, 0.7, 0.2, 0.9, -0.1, 0.5
    ])
    p2 = np.array([
        0.3, -0.1, -0.5, 0.2, -0.4, 0.8, -0.2, 0.0, 0.2, -0.4, 0.6, -0.2, -0.4,
        0.2, 0.3, 0.2, -0.2, -0.4, -0.5, 0.2, 0.2, -0.4
    ])

    p_arr = jnp.array([p1, p2])
    p_arr_idx = 1

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

    flattened_p, obj_val = cg_backtracking(p_arr, p_arr_idx,
                                           partial_forward_fn,
                                           partial_loss_fn, {'params': params},
                                           unravel_fn)
    # Test the backtracking function.
    self.assertSameElements(flattened_p, p1)
    updated_params = apply_updates(params, unravel_fn(p1))
    self.assertEqual(opt_cost({'params': updated_params}), obj_val)

  def test_hessian_free_optimizer(self):
    """Tests the Hessian-free optimizer."""

    model_str = 'autoencoder'
    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)

    loss = 'sigmoid_binary_cross_entropy'
    metrics = 'binary_autoencoder_metrics'

    input_shape = (2, 2, 1)
    output_shape = (4,)

    hps = copy.copy(model_hps)
    hps.update({
        'hid_sizes': [2],
        'activation_function': 'id',
        'input_shape': input_shape,
        'output_shape': output_shape
    })

    model = model_cls(hps, {}, loss, metrics)

    inputs = jnp.array([[[1, 0], [1, 1]], [[1, 0], [0, 1]]])
    targets = inputs.reshape(tuple([inputs.shape[0]] + list(output_shape)))
    batch = {'inputs': inputs, 'targets': targets}

    def forward_fn(variables, inputs):
      logits = model.flax_module.apply(variables, inputs, train=True)
      return logits

    def opt_cost(variables):
      return model.loss_fn(forward_fn(variables, inputs), targets)

    optimizer = hessian_free(model.flax_module, model.loss_fn)

    params = {
        'Dense_0': {
            'kernel': jnp.array([[-1., 2.], [2., 0.], [-1., 3.], [-2., 2.]]),
            'bias': jnp.array([0., 0.])
        },
        'Dense_1': {
            'kernel': jnp.array([[4., 2., -2., 4.], [-3., 1., 2., -4.]]),
            'bias': jnp.array([0., 0., 0., 0.])
        }
    }
    variables = {'params': params}

    grad_fn = jax.grad(opt_cost)
    grads = grad_fn(variables)['params']

    outputs = forward_fn(variables, batch['inputs'])

    n = inputs.shape[0]
    m = outputs.shape[-1]
    d = ravel_pytree(params)[0].shape[0]

    v = np.ones(d)

    state = optimizer.init(params)

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

    matmul_fn = partial(gvp, variables, outputs, state.damping,
                        partial_forward_fn, partial_loss_fn)

    jacobian = jax.jacfwd(partial_forward_fn)(variables)['params']
    jacobian_tensor = np.concatenate((
        jacobian['Dense_0']['bias'].reshape(n, m, -1),
        jacobian['Dense_0']['kernel'].reshape(n, m, -1),
        jacobian['Dense_1']['bias'].reshape(n, m, -1),
        jacobian['Dense_1']['kernel'].reshape(n, m, -1)), axis=2)

    ggn_matrix = 0
    for i in range(n):
      jacobian_matrix = jacobian_tensor[i]
      hessian = jax.hessian(partial_loss_fn)(outputs[i, None])[0, :, 0, :]
      ggn_matrix += np.transpose(jacobian_matrix) @ hessian @ jacobian_matrix
    ggn_matrix /= n
    ggn_matrix += state.damping * np.identity(d)

    expected = ggn_matrix @ v

    # Test the gvp function
    self.assertAlmostEqual(
        jnp.linalg.norm(matmul_fn(v) - expected), 0, places=4)

    p, state = optimizer.update(grads, state, (variables, batch))

    # Test the damping parameter update
    self.assertEqual(state.damping, 3/2)

    # Test the search direction
    self.assertAlmostEqual(
        jnp.linalg.norm(
            ravel_pytree(p)[0] +
            jnp.linalg.inv(ggn_matrix) @ ravel_pytree(grads)[0]),
        0,
        places=4)


if __name__ == '__main__':
  absltest.main()
