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

"""Callback which runs the model debugger."""

import functools
import itertools
import os

from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.hessian import model_debugger
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_CONFIG = {
    'name': 'model_debugger',
}


def get_stats(
    params,
    batch_stats,
    batch,
    step,
    rng,
    local_device_index,
    training_cost):
  """Single step of the training loop.

  Args:
    params: the Flax param pytree.
    batch_stats: a flax.nn.Collection object tracking the model state, usually
      batch norm statistics.
    batch: the per-device batch of data to process.
    step: the current global step of this update. Used to fold in to `rng` to
      produce a unique per-device, per-step RNG.
    rng: the RNG used for calling the model. `step` and `local_device_index`
      will be folded into this to produce a unique per-device, per-step RNG.
    local_device_index: an integer that is unique to this device amongst all
      devices on this host, usually in the range [0, jax.local_device_count()].
      It is folded in to `rng` to produce a unique per-device, per-step RNG.
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates. Takes
      (`flax_module`, `batch_stats`, `batch`, `rng`) as inputs.

  Returns:
    A tuple of the new optimizer, the new batch stats, the scalar training cost,
    and the updated metrics_grabber.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  def opt_cost(flax_module):
    return training_cost(flax_module, batch, batch_stats, rng)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, _), grad = grad_fn(params)

  cost_value, grad = jax.lax.pmean((cost_value, grad), axis_name='batch')
  grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1))**2, grad)
  param_norms = jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1))**2,
                             params)
  return param_norms, grad_norms


class ModelDebugCallback:
  """Used to run the hessian eval in the trainer binary."""

  def __init__(self, model, optimizer, batch_stats, optimizer_state, dataset,
               hps, callback_config, train_dir, rng):
    del hps
    del rng
    del optimizer
    del batch_stats
    del optimizer_state
    del callback_config  # In future CL's we will use this.
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    # copy batch_stats as we close over it, and it gets modified.
    self.dataset = dataset
    checkpoint_dir = os.path.join(train_dir, 'checkpoints')
    pytree_path = os.path.join(checkpoint_dir, 'debugger')
    logger = utils.MetricLogger(pytree_path=pytree_path)
    debugger = model_debugger.ModelDebugger(
        use_pmap=True, metrics_logger=logger)
    # pmap functions for the training loop
    # in_axes = (optimizer = 0, batch_stats = 0, batch = 0, step = None,
    # lr = None, rng = None, local_device_index = 0, training_metrics_grabber=0,
    # training_metrics_grabber, training_cost )
    # Also, we can donate buffers for 'optimizer', 'batch_stats',
    # 'batch' and 'training_metrics_grabber' for update's pmapped computation.

    self.get_stats_pmapped = jax.pmap(
        functools.partial(
            get_stats,
            training_cost=model.training_cost,
            ),
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, 0))
    self.debugger = debugger
    self.logger = logger
    self.dataset = dataset
    self.train_iter = itertools.islice(dataset.train_iterator_fn(), 0, None)

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Computes the loss hessian and returns the max eigenvalue.

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
    batch = next(self.train_iter)
    batch = data_utils.shard(batch)
    rng = jax.random.PRNGKey(0)
    local_device_indices = np.arange(jax.local_device_count())
    p_norms, g_norms = self.get_stats_pmapped(
        params,
        batch_stats,
        batch,
        global_step,
        rng,
        local_device_indices)
    g_norms = jax.tree_map(lambda x: x[0], g_norms)
    p_norms = jax.tree_map(lambda x: x[0], p_norms)
    self.debugger.full_eval(
        step=global_step, grad_norms_sql2=g_norms, param_norms_sql2=p_norms)

    return {}
