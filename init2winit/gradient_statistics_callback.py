# coding=utf-8
# Copyright 2024 The init2winit Authors.
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


import itertools
import os

import flax
from init2winit import base_callback
from init2winit import checkpoint
from init2winit.dataset_lib import data_utils
import jax
import jax.numpy as jnp


class GradientStatisticsCallback(base_callback.BaseCallBack):
  """Runs evals on MT models with datasets/params different than in training."""

  def __init__(self,
               model,
               params,
               batch_stats,
               optimizer_state,
               optimizer_update_fn,
               dataset,
               hps,
               callback_config,
               train_dir,
               rng):
    del optimizer_state
    del optimizer_update_fn
    del batch_stats

    self.dataset = dataset
    self.model = model
    self.hps = hps
    self.callback_config = callback_config
    self.rng = rng
    self.save_path = os.path.join(train_dir, 'gradient_statistics/')

    self.num_batches_in_training_epoch = (
        self.hps.train_size // self.hps.batch_size
    )
    if callback_config is not None:
      if 'num_batches_in_training_epoch' in callback_config.keys():
        self.num_batches_in_training_epoch = callback_config[
            'num_batches_in_training_epoch'
        ]

    self.num_updates = 0

    @jax.jit
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

      grad = jax.lax.pmean(grad, axis_name='batch')
      return grad

    self.pmapped_update = jax.pmap(
        update, axis_name='batch', in_axes=(0, 0, 0, None))

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Computes gradient statistics from mini batches over full training data.
    """
    del optimizer_state
    train_iter = itertools.islice(
        self.dataset.train_iterator_fn(), self.num_batches_in_training_epoch
    )
    unreplicated_params = flax.jax_utils.unreplicate(params)

    grad_sum = jax.tree_map(jnp.zeros_like, unreplicated_params)
    grad_squared_sum = jax.tree_map(jnp.zeros_like, unreplicated_params)
    self.num_updates = 0

    for batch in train_iter:
      sharded_batch = data_utils.shard(batch)
      grads = self.pmapped_update(params, sharded_batch, batch_stats, self.rng)
      grads = flax.jax_utils.unreplicate(grads)

      grad_sum = jax.tree_util.tree_map(
          lambda g_sum, g: g_sum + g, grad_sum, grads
      )

      grad_squared_sum = jax.tree_util.tree_map(
          lambda g_squared, g: g_squared + g**2, grad_squared_sum, grads
      )

      self.num_updates += 1

    self.grad_mean = jax.tree_util.tree_map(
        lambda g_sum: g_sum / self.num_updates, grad_sum
    )
    self.grad_std = jax.tree_util.tree_map(
        lambda g_squared, g_mean: jnp.sqrt(  # pylint: disable=g-long-lambda
            g_squared / self.num_updates - g_mean**2
        ),
        grad_squared_sum,
        self.grad_mean,
    )

    state = dict(
        grad_std=self.grad_std,
        grad_mean=self.grad_mean,
        step=global_step
    )

    checkpoint.save_checkpoint(
        self.save_path,
        step=global_step,
        state=state,
        prefix='measurement_',
        max_to_keep=None)

    return {}
