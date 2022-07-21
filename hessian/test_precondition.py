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

"""Unit tests for precondition.py.

"""

from absl.testing import absltest
from init2winit.hessian.precondition import make_diag_preconditioner
from init2winit.optimizer_lib import optimizers
from init2winit.shared_test_utilities import pytree_allclose
import jax
import jax.numpy as jnp
from ml_collections import FrozenConfigDict
import optax


def _calculate_adam_preconditioner(gradients, beta2, epsilon,
                                   bias_correct):
  """Compute the Adam preconditioner after several steps of training.

  Args:
    gradients: (list of pytrees) The gradients at succesive steps.
    beta2: (float) The Adam beta2 parameter.
    epsilon: (float) The Adam epsilon parameter.
    bias_correct: (bool) True if bias correction is appplied to the second step.

  Returns:
    preconditioner: (pytree) The Adam preconditioner.
  """
  nu = jax.tree_map(lambda x: 0.0, gradients[0])
  for gradient in gradients:
    gradient_sq = jax.tree_map(jnp.square, gradient)
    nu = jax.tree_map(lambda nu, g: beta2*nu + (1 - beta2)*g, nu, gradient_sq)
  if bias_correct:
    nu = jax.tree_map(lambda nu: nu / (1 - beta2**(len(gradients))), nu)

  return jax.tree_map(lambda nu: jnp.sqrt(nu) + epsilon, nu)


class RunPreconditionTest(absltest.TestCase):
  """Tests precondition.py."""

  def test_adam(self):
    """Test Adam preconditioning."""

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7

    opt_hparams = FrozenConfigDict({
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon
    })
    hparams = FrozenConfigDict({
        'optimizer': 'adam',
        'opt_hparams': opt_hparams,
        'l2_decay_factor': 0.0,
        'batch_size': 50,
        'total_accumulated_batch_size': 50,
    })

    init_fn, update_fn = optimizers.get_optimizer(hparams)

    params = {'foo': 1.0, 'bar': {'baz': 3.0}}
    gradients = [{'foo': 0.5, 'bar': {'baz': 0.1}},
                 {'foo': 0.2, 'bar': {'baz': 0.6}}]

    optimizer_state = init_fn(params)
    optimizer_state.base_state.hyperparams['learning_rate'] = lr

    for gradient in gradients:
      updates, optimizer_state = update_fn(gradient, optimizer_state, params)
      params = optax.apply_updates(params, updates)

    # yes bias correction
    expected_preconditioner = _calculate_adam_preconditioner(
        gradients, beta2, epsilon, bias_correct=True)

    preconditioner = make_diag_preconditioner(
        'adam', opt_hparams, optimizer_state,
        FrozenConfigDict(dict(bias_correction=True)))

    self.assertTrue(pytree_allclose(expected_preconditioner, preconditioner))

    # no bias correction
    expected_preconditioner = _calculate_adam_preconditioner(
        gradients, beta2, epsilon, bias_correct=False)

    preconditioner = make_diag_preconditioner(
        'adam', opt_hparams, optimizer_state,
        FrozenConfigDict(dict(bias_correction=False)))

    self.assertTrue(pytree_allclose(expected_preconditioner, preconditioner))

  def test_adam_ks(self):
    """Test kitchen sink preconditioning with scale_by_adam."""
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7

    optimizer = 'kitchen_sink'
    opt_hparams = FrozenConfigDict({
        '0': {
            'element': 'scale_by_adam',
            'hps': {
                'b1': beta1,
                'b2': beta2,
                'eps': epsilon,
                'debias': True
            }
        }
    })

    hparams = FrozenConfigDict({
        'optimizer': optimizer,
        'opt_hparams': opt_hparams,
        'l2_decay_factor': 0.0,
        'batch_size': 50,
        'total_accumulated_batch_size': 50,
    })

    init_fn, update_fn = optimizers.get_optimizer(hparams)

    params = {'foo': 1.0, 'bar': {'baz': 3.0}}
    gradients = [{'foo': 0.5, 'bar': {'baz': 0.1}},
                 {'foo': 0.2, 'bar': {'baz': 0.6}}]

    optimizer_state = init_fn(params)
    optimizer_state.base_state.hyperparams['learning_rate'] = lr

    for gradient in gradients:
      updates, optimizer_state = update_fn(gradient,
                                           optimizer_state,
                                           params)
      params = optax.apply_updates(params, updates)

    expected_preconditioner = _calculate_adam_preconditioner(
        gradients, beta2, epsilon, bias_correct=True)

    preconditioner = make_diag_preconditioner(
        optimizer, opt_hparams, optimizer_state,
        FrozenConfigDict(dict()))

    self.assertTrue(pytree_allclose(
        expected_preconditioner, preconditioner))

if __name__ == '__main__':
  absltest.main()
