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

r"""Callback which runs the hessian eval."""

import os

from flax import jax_utils
from init2winit import base_callback
from init2winit import utils
from init2winit.hessian import hessian_eval
from init2winit.hessian import precondition
import jax
from ml_collections import FrozenConfigDict


def set_up_hessian_eval(model, flax_module, batch_stats, dataset,
                        checkpoint_dir, hessian_eval_config):
  """Builds the CurvatureEvaluator object."""

  # First copy then unreplicate batch_stats. Note batch_stats doesn't affect the
  # forward pass in the hessian eval because we always run the model in training
  # However, we need to provide batch_stats for the model.training_cost API.
  # The copy is needed b/c the trainer will modify underlying arrays.
  batch_stats = jax.tree_map(lambda x: x[:][0], batch_stats)
  def batch_loss(module, batch_rng):
    batch, rng = batch_rng
    return model.training_cost(
        module, batch, batch_stats=batch_stats, dropout_rng=rng)[0]

  pytree_path = os.path.join(checkpoint_dir, hessian_eval_config['name'])
  logger = utils.MetricLogger(pytree_path=pytree_path)
  hessian_evaluator = hessian_eval.CurvatureEvaluator(
      flax_module, hessian_eval_config, dataset=dataset, loss=batch_loss)
  return hessian_evaluator, logger


class HessianCallback(base_callback.BaseCallBack):
  """Used to run the hessian eval in the trainer binary."""

  def __init__(self, model, flax_module, batch_stats, optimizer_state,
               optimizer_update_fn, dataset, hps, callback_config, train_dir,
               rng):
    del rng
    del optimizer_state
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    # copy batch_stats as we close over it, and it gets modified.
    self.hessian_evaluator, self.logger = set_up_hessian_eval(
        model, flax_module, batch_stats, dataset, checkpoint_dir,
        callback_config)
    self.callback_config = FrozenConfigDict(callback_config)
    self.hps = hps
    self.name = callback_config['name']
    self.optimizer_update_fn = optimizer_update_fn

  def run_eval(self, flax_module, batch_stats, optimizer_state, global_step):
    """Computes the loss hessian and returns the max eigenvalue.

    Note, the full lanczos tridiagonal matrix is saved via the logger to
    train_dir/checkpoints/config['name'].

    Args:
      flax_module: Replicated flax module.
      batch_stats: Replicated batch_stats from the trainer.
      optimizer_state: Replicated optimizer state from the trainer.
      global_step: Current training step.

    Returns:
      Max eigenvalue of the loss (full tridiag is saved to disk).
    """
    del batch_stats
    if self.callback_config.get('precondition'):
      precondition_config = self.callback_config.get('precondition_config',
                                                     default=FrozenConfigDict())
      diag_preconditioner = precondition.make_diag_preconditioner(
          self.hps.optimizer, self.hps.opt_hparams,
          jax_utils.unreplicate(optimizer_state), precondition_config)
    else:
      diag_preconditioner = None
    hessian_metrics, hvex, _ = self.hessian_evaluator.evaluate_spectrum(
        flax_module, global_step, diag_preconditioner=diag_preconditioner)

    if self.callback_config.get('compute_stats'):
      grads, updates = self.hessian_evaluator.compute_dirs(
          flax_module, optimizer_state, self.optimizer_update_fn)
      stats_row = self.hessian_evaluator.evaluate_stats(flax_module, grads,
                                                        updates, hvex, [],
                                                        global_step)
      hessian_metrics.update(stats_row)
      interps_row = self.hessian_evaluator.compute_interpolations(
          flax_module, grads, updates, hvex, [], global_step)

      hessian_metrics.update(interps_row)
    if jax.host_id() == 0:
      self.logger.append_pytree(hessian_metrics)
    max_eig_key = self.name + '/max_eig'
    ratio_key = self.name + '/max_eig_ratio'
    pos_neg_key = self.name + '/pos_neg_ratio'
    return {
        max_eig_key: hessian_metrics['max_eig_hess'],
        ratio_key: hessian_metrics['max_eig_hess_ratio'],
        pos_neg_key: hessian_metrics['pos_neg_ratio'],
    }
