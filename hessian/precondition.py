# coding=utf-8
# Copyright 2023 The init2winit Authors.
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


def _maybe_bias_correct(ema, decay, count, debias):
  """Apply bias correction to an EMA.

  Args:
    ema: (pytree) The EMA before bias correction.
    decay: (float) The decay rate, e.g. Adam's beta2 parameter.
    count: (int) The number of EMA updates so far.
    debias: (bool) Whether to apply bias correction.

  Returns:
    bias-corrected EMA
  """
  if debias:
    bias_correction_factor = 1 - decay**count
    return jax.tree_map(lambda x: x / bias_correction_factor, ema)
  else:
    return ema


def _make_inv_power_preconditioner(diag, eps, eps_root=0.0, power=0.5):
  """Construct an inverse power preconditioner (e.g. Adam, Adagrad).

  Args:
    diag: (pytree) The values on the diagonal (e.g. Adam's nu).
    eps: (float) Added outside the inv power for numerical stability.
    eps_root: (float) Added inside the inv power for numerical stability.
    power: (float) The power to which diag is raised.

  Returns:
    (pytree) The preconditioner, as a pytree.
  """
  return jax.tree_map(lambda v: jnp.power(v + eps_root, power) + eps, diag)


# KS transforms for which preconditioning is implemented.
SUPPORTED_KS_TRANSFORMS = ['scale_by_adam',
                           'scale_by_nadam',
                           'scale_by_amsgrad',
                           'precondition_by_rms',
                           'precondition_by_rss']


def _make_ks_preconditioner(element, state, hps):
  """Construct the diagonal preconditioner for a KS transform.

  Args:
   element: (str) the KS element
   state: (pytree) the KS state for this transform
   hps: (pytree) the KS opt hparams for this transform

  Returns:
   (pytree) diagonal preconditioner
  """
  err_msg = 'all KS hps must be set in order to compute preconditioned Hessian'
  if element == 'scale_by_adam' or element == 'scale_by_nadam':
    if not ('b2' in hps and 'debias' in hps
            and 'eps' in hps and 'eps_root' in hps):
      raise ValueError(err_msg)
    nu = _maybe_bias_correct(state.nu, hps['b2'], state.count, hps['debias'])
    return _make_inv_power_preconditioner(nu, hps['eps'], hps['eps_root'],
                                          hps.get('power', 0.5))
  elif element == 'scale_by_amsgrad':
    if not ('eps' in hps and 'eps_root' in hps):
      raise ValueError(err_msg)
    return _make_inv_power_preconditioner(state.nu, hps['eps'], hps['eps_root'])
  elif element == 'precondition_by_rms':
    if not ('decay' in hps and 'debias' in hps
            and 'eps' in hps and 'eps_root' in hps):
      raise ValueError(err_msg)
    nu = _maybe_bias_correct(state.nu, hps['decay'], state.count, hps['debias'])
    return _make_inv_power_preconditioner(nu, hps['eps'], hps['eps_root'])
  elif element == 'precondition_by_rss':
    if 'eps' not in hps:
      raise ValueError(err_msg)
    return _make_inv_power_preconditioner(state.sum_of_squares,
                                          hps['eps'], 0.0)


def make_diag_preconditioner(optimizer, opt_hparams,
                             optimizer_state, precondition_config):
  """Construct a diagonal preconditioner.

  Given an optimizer and its state, return that optimizer's preconditioner.
  Note that in our nomenclature, the preconditioner is a diagonal approximation
  to the Hessian, not the inverse Hessian.  In other words, we return a pytree
  P, the same shape as the model parameters, with the property that the
  optimizer update (without momentum) is diag(P)^{-1} g.

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

  # unwrap the InjectHyperparamsState
  optimizer_state = optimizer_state.inner_state[0]

  if optimizer == 'adam':
    bias_correct = precondition_config.get('bias_correction', default=True)
    nu, count = optimizer_state.nu, optimizer_state.count
    nu = _maybe_bias_correct(nu, opt_hparams.beta2, count, bias_correct)
    return _make_inv_power_preconditioner(nu, opt_hparams.epsilon)

  # The following preconditioning logic covers the case where there is
  # a single KS transform chain, and a single preconditioner in that chain.
  if optimizer == 'kitchen_sink' and '0' in opt_hparams:
    precondition_steps = [
        step for step in opt_hparams.keys()
        if opt_hparams[step]['element'] in SUPPORTED_KS_TRANSFORMS
    ]
    if len(precondition_steps) != 1:
      raise ValueError("Don't know how to precondition this optimizer")

    step = precondition_steps[0]
    element = opt_hparams[step]['element']
    hps = opt_hparams[step]['hps']
    state = optimizer_state[int(step)]
    return _make_ks_preconditioner(element, state, hps)

  raise ValueError("Don't know how to precondition this optimizer")
