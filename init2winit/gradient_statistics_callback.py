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

"""Callback for computing gradient statistics given set of params.
"""

import functools
import itertools
import os

import flax.linen as nn
from init2winit import base_callback
from init2winit import checkpoint
from init2winit.dataset_lib import data_utils
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


class GradientStatisticsCallback(base_callback.BaseCallBack):
  """Runs evals on MT models with datasets/params different than in training."""

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
    del optimizer_state
    del finalize_batch_fn

    self.dataset = dataset
    self.model = model
    self.hps = hps
    self.callback_config = callback_config
    self.rng = rng
    self.save_path = os.path.join(train_dir, 'gradient_statistics/')
    self.mesh = mesh

    self.num_batches_in_training_epoch = (
        self.hps.train_size // self.hps.batch_size
    )
    if callback_config is not None:
      if 'num_batches_in_training_epoch' in callback_config.keys():
        self.num_batches_in_training_epoch = callback_config[
            'num_batches_in_training_epoch'
        ]

    self.num_updates = 0
    self.orbax_checkpoint_manager = ocp.CheckpointManager(
        self.save_path,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True
        ),
    )

    def update(params, batch, batch_stats, dropout_rng):
      def opt_cost(params):
        return self.model.training_cost(
            params,
            batch=batch,
            batch_stats=batch_stats,
            dropout_rng=dropout_rng,
        )

      grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
      _, grad = grad_fn(params)

      return grad

    params_sharding = jax.tree_util.tree_map(
        lambda x: x.sharding, params
    )
    batch_stats_sharding = nn.get_sharding(batch_stats, self.mesh)

    self.jitted_update = jax.jit(
        update,
        in_shardings=(
            params_sharding,
            jax.sharding.NamedSharding(
                self.mesh, jax.sharding.PartitionSpec('devices')),
            batch_stats_sharding,
            None
        ),
        out_shardings=(params_sharding)
    )

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Computes gradient statistics from mini batches over full training data.
    """
    del optimizer_state
    train_iter = itertools.islice(
        self.dataset.train_iterator_fn(), self.num_batches_in_training_epoch
    )

    grad_sum = jax.tree.map(jnp.zeros_like, params)
    grad_squared_sum = jax.tree.map(jnp.zeros_like, params)
    self.num_updates = 0

    make_global_array_fn = functools.partial(
        data_utils.make_global_array, mesh=self.mesh
    )

    for batch in train_iter:
      sharded_batch = jax.tree_util.tree_map(make_global_array_fn, batch)
      grads = self.jitted_update(params, sharded_batch, batch_stats, self.rng)

      grad_sum = jax.tree_util.tree_map(
          lambda g_sum, g: g_sum + g, grad_sum, grads
      )

      grad_squared_sum = jax.tree_util.tree_map(
          lambda g_squared, g: g_squared + g**2, grad_squared_sum, grads
      )

      self.num_updates += 1

    grad_mean = jax.tree_util.tree_map(
        lambda g_sum: g_sum / self.num_updates, grad_sum
    )
    grad_std = jax.tree_util.tree_map(
        lambda g_squared, g_mean: jnp.sqrt(  # pylint: disable=g-long-lambda
            g_squared / self.num_updates - g_mean**2
        ),
        grad_squared_sum,
        grad_mean,
    )

    state = dict(
        grad_std=jax.device_get(grad_std),
        grad_mean=jax.device_get(grad_mean),
        step=global_step
    )

    checkpoint.save_checkpoint(
        step=global_step,
        state=state,
        orbax_checkpoint_manager=self.orbax_checkpoint_manager)

    return {}
