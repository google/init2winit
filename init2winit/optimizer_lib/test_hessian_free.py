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

"""Tests for hessian_free.py.

"""

import copy
from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import models
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib.hessian_free import cg_backtracking
from init2winit.optimizer_lib.hessian_free import CGIterationTrackingMethod
from init2winit.optimizer_lib.hessian_free import get_obj_val
from init2winit.optimizer_lib.hessian_free import gvp
from init2winit.optimizer_lib.hessian_free import line_search
from init2winit.optimizer_lib.hessian_free import mf_conjgrad_solver
from init2winit.optimizer_lib.hessian_free import relative_per_iteration_progress_test
from init2winit.optimizer_lib.hessian_free import residual_norm_test
from init2winit.optimizer_lib.hessian_free import tree_slice
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
from optax import apply_updates
import tree_math as tm

_INPUT_SHAPE = (2, 2, 1)
_OUTPUT_SHAPE = (4,)
_INPUT_DATA = np.array([
    [[1, 0], [1, 1]],
    [[1, 0], [0, 1]]
])


def _get_pd_mat(mat):
  """Returns a positive-definite matrix."""
  n = mat.shape[0]
  return mat @ np.transpose(mat) / n**2 + np.eye(n)


def _load_conjgrad_inputs():
  """Loads inputs to the conjugate gradient solver."""
  params = {
      'Dense_0': {
          'kernel': np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]]),
          'bias': np.array([0., 0.])
      }
  }
  variables = {'params': params}
  n = 10
  mat = np.array([[2., -1., 2., 3., 4., -2., 3., 5., -10., 2.],
                  [-1., -1., 2., 3., 4., -2., 3., 5., -10., 4.],
                  [2., 2., 5., -2., 12., -6., 2., 6., -3., 0.],
                  [3., 3., -2., 6., 1., -2., 7., 1., 6., -2.],
                  [4., 4., 12., 1., 4., 5., -6., 2., 8., 1.],
                  [-2., -2., -6., -2., 4., -6., 3., 5., -10., 2.],
                  [3., 3., 2., 7., -4., 2., 3., 5., -10., 2.],
                  [5., 5., 6., 1., 2., 8., 3., 5., -10., 2.],
                  [-10., -10., -3., 6., 4., 1., 3., 5., -10., 2.],
                  [2., 4., 0., -2., 4., -2., 3., 5., -10., 2.]])
  def test_matmul_fn(v):
    flattened_v, unravel_fn = ravel_pytree(v)
    return unravel_fn(mat @ flattened_v)
  def obj_fn(v):
    flattened_v = ravel_pytree(v)[0]
    return flattened_v @ np.array([10., 6., -7., 5., 2., 8., 2, 2., 10, -20.])
  b = tm.Vector({
      'Dense_0': {
          'kernel': np.array([[-5., 10.], [-5., 20.], [-7., -5.], [8., 2.]]),
          'bias': np.array([4., -6.])
      }
  })
  x0 = tm.Vector({
      'Dense_0': {
          'kernel': np.array([[1., -4.], [6., -3.], [4., 5.], [1., -9.]]),
          'bias': np.array([4., 8.])
      }
  })
  return test_matmul_fn, b, x0, n, obj_fn, variables


def _load_autoencoder_model():
  """Loads a test autoencoder model."""
  model_str = 'autoencoder'
  model_cls = models.get_model(model_str)
  model_hps = models.get_model_hparams(model_str)

  loss = 'sigmoid_binary_cross_entropy'
  metrics = 'binary_autoencoder_metrics'

  hps = copy.copy(model_hps)
  hps.update({
      'optimizer': 'hessian_free',
      'opt_hparams': {
          'weight_decay': 0.0,
          'init_damping': 1.0,
          'damping_ub': 10**2,
          'damping_lb': 10**-6,
          'use_line_search': False,
          'cg_iter_tracking_method': 'back_tracking',
      },
      'hid_sizes': [2],
      'activation_function': ['id'],
      'input_shape': _INPUT_SHAPE,
      'output_shape': _OUTPUT_SHAPE,
  })

  model = model_cls(hps, {'apply_one_hot_in_loss': False}, loss, metrics)
  init_fn, update_fn = optimizers.get_optimizer(hps, model)
  params = {
      'Dense_0': {
          'kernel': np.array([[-1., 2.], [2., 0.], [-1., 3.], [-2., 2.]]),
          'bias': np.array([0., 0.])
      },
      'Dense_1': {
          'kernel': np.array([[4., 2., -2., 4.], [-3., 1., 2., -4.]]),
          'bias': np.array([0., 0., 0., 0.])
      }
  }
  state = init_fn(params)
  variables = {'params': params}

  return model, update_fn, state, variables


def _load_autoencoder_data():
  """Loads test autoencoder data."""
  targets = _INPUT_DATA.reshape(
      tuple([_INPUT_DATA.shape[0]] + list(_OUTPUT_SHAPE)))
  return {'inputs': _INPUT_DATA, 'targets': targets}


@partial(tm.unwrap, out_vectors=False)
def tm_norm(x):
  return np.linalg.norm(ravel_pytree(x)[0])


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
    mat = _get_pd_mat(
        np.array(
            [[2., 4., 5., 2., 8.],
             [0., 4., 3., 5., 3.],
             [-2., -2., 9., -2., -6.],
             [4., 1., -11., 1., 4.],
             [-5., 4., -9., 3., -2.]]))
    b = tm.Vector(np.array([-3, 2, 0, 3, -4]))
    x0 = tm.Vector(np.ones(n))

    test_matmul_fn = tm.unwrap(lambda x: mat @ x)
    x, *_ = mf_conjgrad_solver(
        test_matmul_fn, b, x0, n, 1e-6, 10, None, 'residual_norm_test')
    self.assertAlmostEqual(tm_norm(test_matmul_fn(x) - b), 0, places=3)

  def test_conjgrad_iteration_last_tracking(self):
    """Tests conjugate gradient method with iteration last-tracking."""
    test_matmul_fn, b, x0, n, obj_fn, variables = _load_conjgrad_inputs()
    expected, x, *_ = mf_conjgrad_solver(
        test_matmul_fn, b, x0, n, 1e-6, 10, None, 'residual_norm_test',
        CGIterationTrackingMethod.LAST_TRACKING,
        obj_fn=obj_fn, variables=variables)
    self.assertAlmostEqual(tm_norm(x - expected), 0, places=3)

  def test_conjgrad_iteration_best_tracking(self):
    """Tests conjugate gradient method with iteration best-tracking."""
    test_matmul_fn, b, x0, n, obj_fn, variables = _load_conjgrad_inputs()
    x = mf_conjgrad_solver(
        test_matmul_fn, b, x0, n, 1e-6, 10, None, 'residual_norm_test',
        CGIterationTrackingMethod.BEST_TRACKING,
        obj_fn=obj_fn, variables=variables)[1]
    expected = mf_conjgrad_solver(
        test_matmul_fn, b, x0, 5, 1e-6, 10, None, 'residual_norm_test',
        CGIterationTrackingMethod.BEST_TRACKING,
        obj_fn=obj_fn, variables=variables)[0]
    self.assertAlmostEqual(tm_norm(x - expected), 0, places=3)

  def test_conjgrad_iteration_back_tracking(self):
    """Tests conjugate gradient method with iteration back-tracking."""
    test_matmul_fn, b, x0, n, obj_fn, variables = _load_conjgrad_inputs()
    x = mf_conjgrad_solver(
        test_matmul_fn, b, x0, n, 1e-6, 10, None, 'residual_norm_test',
        CGIterationTrackingMethod.BACK_TRACKING,
        obj_fn=obj_fn, variables=variables)[1]
    expected = mf_conjgrad_solver(
        test_matmul_fn, b, x0, 9, 1e-6, 10, None, 'residual_norm_test',
        CGIterationTrackingMethod.BEST_TRACKING,
        obj_fn=obj_fn, variables=variables)[0]
    self.assertAlmostEqual(tm_norm(x - expected), 0, places=3)

  def test_conjgrad_preconditioning(self):
    """Tests conjugate gradient method with preconditioning."""
    n = 5
    mat = _get_pd_mat(
        np.array(
            [[2., 4., 5., 2., 8.],
             [0., 4., 3., 5., 3.],
             [-2., -2., 9., -2., -6.],
             [4., 1., -11., 1., 4.],
             [-5., 4., -9., 3., -2.]]))
    precond_mat = _get_pd_mat(
        np.array(
            [[4., 2., 0., 2., 4.],
             [-2., 4., 4., 2., 6.],
             [4., 4., -8., -2., -4.],
             [-2., 2., 4., 0., -2.],
             [2., 2., -6., 4., 0.]]))
    b = tm.Vector(np.array([-3, 2, 0, 3, -4]))
    x0 = tm.Vector(np.ones(n))

    test_matmul_fn = tm.unwrap(lambda x: mat @ x)
    test_precond_fn = tm.unwrap(lambda x: precond_mat @ x)
    x, *_ = mf_conjgrad_solver(
        test_matmul_fn, b, x0, n, 1e-6, 10, test_precond_fn,
        'residual_norm_test')
    self.assertAlmostEqual(tm_norm(test_matmul_fn(x) - b), 0, places=3)

  def test_conjgrad_martens_termination_criterion(self):
    """Tests conjugate gradient method with Martens termination criterion."""
    n = 500
    mat = _get_pd_mat(
        np.array([[((i + j) % n) for j in range(n)] for i in range(n)]))
    b = tm.Vector(np.linspace(1, n, n) / n)
    x0 = tm.Vector(np.zeros(n))

    test_mvm_fn = tm.unwrap(lambda x: mat @ x)

    x, *_ = mf_conjgrad_solver(
        test_mvm_fn, b, x0, n, 1e-6, 500, None,
        'relative_per_iteration_progress_test')
    f_value = x @ (test_mvm_fn(x) - 2 * b) / 2
    self.assertAlmostEqual(f_value, -0.223612576, places=5)

  def test_cg_backtracking(self):
    """Tests CG backtracking."""
    model, _, _, variables = _load_autoencoder_model()
    batch = _load_autoencoder_data()

    def forward_fn(variables, inputs):
      return model.flax_module.apply(variables, inputs, train=False)

    def opt_cost(params):
      return model.loss_fn(forward_fn(params, batch['inputs']),
                           batch['targets'])

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

    p_arr = tm.Vector({
        'Dense_0': {
            'kernel': jnp.array([
                [[0.1, -0.4], [-0.6, 0.4], [0.6, -0.7], [0.0, 0.5]],
                [[-0.5, 0.2], [-0.4, 0.8], [-0.2, 0.0], [0.2, -0.4]],
                [[-0.2, -0.2], [-0.2, 0.0], [0.4, 0.1], [0.2, 0.4]]]),
            'bias': jnp.array([[0.5, 0.2], [0.3, -0.1], [0.2, 0.4]])
        },
        'Dense_1': {
            'kernel': jnp.array([
                [[0.4, -0.6, -0.8, 0.7], [0.3, 0.2, -0.2, -0.4]],
                [[0.2, 0.9, -0.1, 0.5], [-0.5, 0.2, 0.2, -0.4]],
                [[0.2, -0.4, -0.4, 0.8], [-0.1, 0.3, 0.2, 0.2]]]),
            'bias': jnp.array([
                [-0.7, 0.2, 0.1, -0.2],
                [0.6, -0.2, -0.4, 0.2],
                [0.2, 0.3, -0.2, 0.4]])
        }
    })
    p_arr_idx = 2

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

    def obj_fn(variables):
      return partial_loss_fn(partial_forward_fn(variables))

    p, obj_val = cg_backtracking(p_arr, p_arr_idx, obj_fn, variables)
    expected = tree_slice(p_arr, 0).tree

    # Test the backtracking function.
    self.assertSameElements(p.tree, expected)
    updated_params = apply_updates(params, expected)
    self.assertAlmostEqual(opt_cost({'params': updated_params}),
                           obj_val, places=4)

  def test_line_search(self):
    """Tests the line search algorithm."""
    model, _, _, variables = _load_autoencoder_model()
    batch = _load_autoencoder_data()

    def forward_fn(variables, inputs):
      return model.flax_module.apply(variables, inputs, train=False)

    def opt_cost(params):
      return model.loss_fn(forward_fn(params, batch['inputs']),
                           batch['targets'])

    unravel_fn = ravel_pytree(variables['params'])[1]

    p = tm.Vector(
        unravel_fn(jnp.array([
            0.5, 0.2, 0.1, -0.4, -0.6, 0.4, 0.6, -0.7, 0.0, 0.5, -0.7, 0.2, 0.1,
            -0.2, 0.4, -0.6, -0.8, 0.7, 0.2, 0.9, -0.1, 0.5])))

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

    def obj_fn(variables):
      return partial_loss_fn(partial_forward_fn(variables))

    initial_lr = 1.0
    initial_obj_val = get_obj_val(obj_fn, variables, p)

    grad_fn = jax.grad(opt_cost)
    grads = tm.Vector(grad_fn(variables)['params'])

    final_lr = line_search(
        initial_lr, initial_obj_val, obj_fn, variables, grads, p)

    # Test the final learning rate value.
    self.assertEqual(final_lr, initial_lr)

  def test_gvp(self):
    """Tests the gvp function."""
    model, _, state, variables = _load_autoencoder_model()
    batch = _load_autoencoder_data()

    def forward_fn(variables, inputs):
      return model.flax_module.apply(variables, inputs, train=True)

    outputs = forward_fn(variables, batch['inputs'])

    n = batch['inputs'].shape[0]
    m = outputs.shape[-1]
    d = ravel_pytree(variables['params'])[0].shape[0]

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

    matmul_fn = tm.unwrap(
        partial(gvp, variables, outputs, state.inner_state.damping,
                partial_forward_fn, partial_loss_fn), out_vectors=False)

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
    ggn_matrix += state.inner_state.damping * np.identity(d)

    expected = ggn_matrix @ np.ones(d)

    ones = tm.Vector(
        jax.tree_map(lambda x: jnp.ones(x.shape), variables['params']))
    # Test the gvp function
    self.assertAlmostEqual(
        jnp.linalg.norm(
            ravel_pytree(matmul_fn(ones))[0] - expected), 0, places=4)

  def test_hessian_free_optimizer(self):
    """Tests the Hessian-free optimizer."""

    model, update_fn, state, variables = _load_autoencoder_model()
    batch = _load_autoencoder_data()

    def forward_fn(variables, inputs):
      logits = model.flax_module.apply(variables, inputs, train=True)
      return logits

    def opt_cost(variables):
      return model.loss_fn(forward_fn(variables, batch['inputs']),
                           batch['targets'])

    outputs = forward_fn(variables, batch['inputs'])

    n = batch['inputs'].shape[0]
    m = outputs.shape[-1]
    d = ravel_pytree(variables['params'])[0].shape[0]

    partial_forward_fn = partial(forward_fn, inputs=batch['inputs'])
    partial_loss_fn = partial(model.loss_fn, targets=batch['targets'])

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
    ggn_matrix += state.inner_state.damping * np.identity(d)

    grad_fn = jax.grad(opt_cost)
    grads = grad_fn(variables)['params']

    update_pmapped = jax.pmap(
        update_fn, axis_name='batch', in_axes=(None, None, None, 0, None))

    batch_shard = data_utils.shard(batch)
    state.hyperparams['learning_rate'] = 1.0
    p, state = update_pmapped(grads, state, variables['params'], batch_shard,
                              None)

    # Test the damping parameter update
    self.assertEqual(state.inner_state.damping, 1.5)

    # Test the search direction
    self.assertAlmostEqual(
        jnp.linalg.norm(
            ravel_pytree(p)[0] +
            jnp.linalg.inv(ggn_matrix) @ ravel_pytree(grads)[0]),
        0,
        places=4)


if __name__ == '__main__':
  absltest.main()
