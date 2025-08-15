# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Callback which runs the model debugger."""

import functools
import os

import flax
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.hessian import model_debugger
import jax
import jax.numpy as jnp

DEFAULT_CONFIG = {
    'name': 'model_debugger',
}


def get_grad(params,
             batch,
             rng,
             batch_stats=None,
             module_flags=None,
             training_cost=None):
  """Single step of the training loop.

  Args:
    params: the Flax param pytree. batch norm statistics.
    batch: the per-device batch of data to process.
    rng: the RNG used for calling the model. Assumes the step and device index
      has already been folded in.
    batch_stats: Same as in trainer.py
    module_flags: Used in the skip analysis.
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates. Takes (`params`, `batch_stats`,
      `batch`, `rng`) as inputs.

  Returns:
    Gradient of the given loss.
  """

  if module_flags is not None:
    kwargs = {'module_flags': module_flags}
  else:
    kwargs = {}
  def opt_cost(params):
    return training_cost(
        params,
        batch,
        batch_stats=batch_stats,
        dropout_rng=rng,
        **kwargs)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  _, grad = grad_fn(params)

  grad = jax.lax.pmean(grad, axis_name='batch')
  return grad


class ModelDebugCallback:
  """Used to run the hessian eval in the trainer binary."""

  def __init__(
      self,
      model,
      params,
      batch_stats,
      optimizer_state,
      dataset,
      hps,
      callback_config,
      train_dir,
      rng,
      mesh,
      finalize_batch_fn,
  ):
    del hps
    del params
    del optimizer_state
    del mesh
    del finalize_batch_fn
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    # copy batch_stats as we close over it, and it gets modified.
    self.dataset = dataset
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    pytree_path = os.path.join(checkpoint_dir, 'debugger')
    logger = utils.MetricLogger(pytree_path=pytree_path)

    get_act_stats_fn = model_debugger.create_forward_pass_stats_fn(
        model.apply_on_batch,
        capture_activation_norms=True,
        sown_collection_names=callback_config.get('sown_collection_names'))
    batch_stats = jax.tree.map(lambda x: x[:][0], batch_stats)
    grad_fn = functools.partial(
        get_grad,
        batch_stats=batch_stats,
        training_cost=model.training_cost,
    )
    debugger = model_debugger.ModelDebugger(
        use_pmap=True,
        forward_pass=get_act_stats_fn,
        metrics_logger=logger,
        grad_fn=grad_fn,
        skip_flags=callback_config.get('skip_flags'),
        skip_groups=callback_config.get('skip_groups'))
    # pmap functions for the training loop
    # in_axes = (params = 0, batch_stats = 0, batch = 0, step = None,
    # lr = None, rng = None, local_device_index = 0, training_metrics_grabber=0,
    # training_metrics_grabber, training_cost )
    # Also, we can donate buffers for 'optimizer', 'batch_stats',
    # 'batch' and 'training_metrics_grabber' for update's pmapped computation.
    self.debugger = debugger
    self.logger = logger
    self.dataset = dataset

    batch = next(iter(dataset.train_iterator_fn()))
    self.batch = data_utils.shard(batch)

    self.batch_rng = flax.jax_utils.replicate(rng)

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Runs ModelDebugger.full_eval on the given params.

    Note, the full lanczos tridiagonal matrix is saved via the logger to
    train_dir/checkpoints/config['name'].

    Args:
      params: Replicated model parameter tree.
      batch_stats: Replicated batch_stats from the trainer.
      optimizer_state: Replicated optimizer state from the trainer.
      global_step: Current training step.

    Returns:
      Max eigenvalue of the loss (full tridiag is saved to disk).
    """
    del optimizer_state
    del batch_stats
    p_norms = jax.tree.map(lambda x: jnp.linalg.norm(x[0].reshape(-1))**2,
                           params)

    self.debugger.full_eval(
        step=global_step,
        params=params,
        param_norms_sql2=p_norms,
        batch=self.batch,
        rng=self.batch_rng)

    return {}
