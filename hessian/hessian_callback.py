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

r"""Callback which runs the hessian eval."""

import os

from flax import jax_utils
from init2winit import base_callback
from init2winit import utils
from init2winit.hessian import hessian_eval
from init2winit.hessian import precondition
import jax
import jax.numpy as jnp
from ml_collections import FrozenConfigDict


def set_up_hessian_eval(model, params, batch_stats, dataset,
                        checkpoint_dir, hessian_eval_config):
  """Builds the CurvatureEvaluator object."""

  # First copy then unreplicate batch_stats. Note batch_stats doesn't affect the
  # forward pass in the hessian eval because we always run the model in training
  # However, we need to provide batch_stats for the model.training_cost API.
  # The copy is needed b/c the trainer will modify underlying arrays.
  batch_stats = jax.tree_map(lambda x: x[:][0], batch_stats)
  def batch_loss(params, batch_rng):
    batch, rng = batch_rng

    apply_kwargs = {'train': True}
    apply_kwargs['mutable'] = ['batch_stats']
    apply_kwargs['rngs'] = {'dropout': rng}

    logits, _ = model.apply_on_batch(
        params, batch_stats, batch, **apply_kwargs)
    weights = batch.get('weights')
    loss_numerator, loss_denominator = model.loss_fn(
        logits, batch['targets'], weights)

    return (loss_numerator / loss_denominator)

  def batch_output(module, batch_rng):
    batch, rng = batch_rng
    out = model.apply_on_batch(module, batch_stats, batch,
                               mutable=['batch_stats'], train=True,
                               rngs={'dropout': rng})[0]
    # If the rank is greater than 2, treat all but the last dimension
    # as distinct examples.
    return out.reshape(-1, out.shape[-1])

  def batch_weights(batch_rng):
    # TODO(b/280322542): this should be jax.random.bits(batch_rng)
    batch = jax.random.key_data(batch_rng)[0]
    if 'weights' not in batch:
      return jnp.ones(len(batch['inputs']))
    else:
      return batch['weights'].reshape(-1)

  pytree_path = os.path.join(checkpoint_dir, hessian_eval_config['name'])
  logger = utils.MetricLogger(pytree_path=pytree_path)
  hessian_evaluator = hessian_eval.CurvatureEvaluator(
      params, hessian_eval_config, dataset=dataset,
      loss=batch_loss, output_fn=batch_output, weights_fn=batch_weights)
  return hessian_evaluator, logger


class HessianCallback(base_callback.BaseCallBack):
  """Used to run the hessian eval in the trainer binary."""

  def __init__(self, model, params, batch_stats, optimizer_state,
               optimizer_update_fn, dataset, hps, callback_config, train_dir,
               rng):
    del rng
    del optimizer_state
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    # copy batch_stats as we close over it, and it gets modified.
    self.hessian_evaluator, self.logger = set_up_hessian_eval(
        model, params, batch_stats, dataset, checkpoint_dir,
        callback_config)
    self.callback_config = FrozenConfigDict(callback_config)
    self.hps = hps
    self.name = callback_config['name']
    self.optimizer_update_fn = optimizer_update_fn

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Computes the loss hessian and returns the max eigenvalue.

    Note, the full lanczos tridiagonal matrix is saved via the logger to
    train_dir/checkpoints/config['name'].

    Args:
      params: Replicated param pytree.
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
        params, global_step, diag_preconditioner=diag_preconditioner)

    if self.callback_config.get('compute_stats'):
      grads, updates = self.hessian_evaluator.compute_dirs(
          params, optimizer_state, self.optimizer_update_fn)
      stats_row = self.hessian_evaluator.evaluate_stats(params, grads,
                                                        updates, hvex, [],
                                                        global_step)
      hessian_metrics.update(stats_row)
      interps_row = self.hessian_evaluator.compute_interpolations(
          params, grads, updates, hvex, [], global_step)

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
