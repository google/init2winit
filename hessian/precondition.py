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

"""Compute diagonal preconditioners for different optimizers."""

from init2winit.optimizer_lib import gradient_accumulator
import jax
import jax.numpy as jnp


def _make_adam_preconditioner(nu, count, eps, beta2, bias_correction):
  """Construct the diagonal preconditioner for Adam.

  Args:
    nu: (pytree) The second moment EMA of the gradients.
    count: (int) The number of steps so far.
    eps: (float) The Adam epsilon parameter.
    beta2: (float) The Adam beta2 parameter.
    bias_correction: (bool): If true, incorporate bias correction into the
      preconditioner.

  Returns:
    (pytree) Adam's diagonal preconditioner, as a pytree.
  """
  if bias_correction:
    bias_correction_factor = 1 - beta2**count
    nu = jax.tree_map(lambda x: x / bias_correction_factor, nu)
  return jax.tree_map(lambda v: jnp.sqrt(v) + eps, nu)


def make_diag_preconditioner(optimizer, opt_hparams,
                             optimizer_state, precondition_config):
  """Construct a diagonal preconditioner.

  Given an optimizer and its state, return that optimizer's preconditioner.
  Note that in our nomenclature, the preconditioner is a diagonal approximation
  to the Hessian, not the inverse Hessian.

  The current config options are
    bias_correction: (bool) If set to true, and if the optimizer employs
      bias correction, we incorporate the bias correction into the
      preconditioner.


  Args:
    optimizer: (str) The optimizer name, from the init2winit config.
    opt_hparams: (dict) The opt_hparams dict from the init2winit config.
    optimizer_state: (pytree) The unreplicated optimizer state.
    precondition_config: (ConfigDict) Configs for the preconditioner.
  Returns:
    (pytree) diagonal preconditioner
  """
  if isinstance(optimizer_state, gradient_accumulator.GradientAccumulatorState):
    optimizer_state = optimizer_state.base_state

  if optimizer == 'adam':
    eps = opt_hparams.epsilon
    beta2 = opt_hparams.beta2
    nu = optimizer_state.inner_state[0].nu
    count = optimizer_state.inner_state[0].count
    bias_correction = precondition_config.get('bias_correction', default=True)
    return _make_adam_preconditioner(nu, count, eps, beta2, bias_correction)
