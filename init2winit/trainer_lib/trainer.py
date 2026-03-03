# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Standard trainer for the init2winit project."""

from init2winit.dataset_lib import data_utils
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
import jax

NamedSharding = jax.NamedSharding
P = jax.P


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def update(self, batch, rng, metrics_state, training_cost):
    """Single step of the training loop.

    Uses the training algorithm's update_params function to get the updated
    optimizer state, params, and batch stats.
    Note this method is also responsible for updating the private _global_step
    attribute of the Trainer.

    Args:
      batch: the per-device batch of data to process.
      rng: the RNG used for calling the model. `step` and `local_device_index`
        will be folded into this to produce a unique per-device, per-step RNG.
      metrics_state: the current metrics state.
      training_cost: the current training cost.

    Returns:
      A tuple containing:
        new_optimizer_state: Pytree of optimizer state.
        new_params: Pytree of model parameters.
        new_model_state: Pytree of model state.
        cost_value: The training cost used for the metrics state.
        grad: The gradient used for the metrics state.
    """

    # `jax.random.split` is very slow outside the train step, so instead we do a
    # `jax.random.fold_in` here.
    # The RNG is already sharded per-device, so each device has a scalar
    # PRNGKey.
    step = self._global_step
    rng = jax.random.fold_in(rng, step)

    new_optimizer_state, new_params, new_batch_stats, cost_value, grad = (
        self.training_algorithm.update_params(
            params=self._params,
            model_state=self._batch_stats,
            optimizer_state=self._optimizer_state,
            batch=batch,
            global_step=step,
            rng=rng,
        )
    )

    new_metrics_state = None
    if metrics_state is not None:
      new_metrics_state = self._metrics_update_fn(
          metrics_state,
          step,
          cost_value,
          grad,
          self._params,
          new_params,
          new_optimizer_state,
          new_batch_stats,
      )

    new_sum_train_cost = training_cost + cost_value
    self._global_step += 1
    return (
        new_optimizer_state,
        new_params,
        new_batch_stats,
        new_metrics_state,
        new_sum_train_cost,
    )

  def shard(
      self,
      unreplicated_params,
      unreplicated_optimizer_state,
      unreplicated_batch_stats,
      unreplicated_metrics_state,
  ):
    """Shards the training state pytrees and returns the sharded pytrees."""
    params_sharding = self._model.get_sharding(unreplicated_params, self._mesh)

    _, params = data_utils.shard_pytree(
        unreplicated_params, self._mesh, params_sharding
    )
    optimizer_state_sharding = jax.tree_util.tree_map(
        lambda x: x.sharding
        if isinstance(x.sharding, NamedSharding)
        else NamedSharding(self._mesh, P()),
        unreplicated_optimizer_state,
    )

    _, optimizer_state = data_utils.shard_pytree(
        unreplicated_optimizer_state, self._mesh, optimizer_state_sharding
    )

    batch_stats_sharding, batch_stats = data_utils.shard_pytree(
        unreplicated_batch_stats, self._mesh
    )
    metrics_state_sharding, metrics_state = data_utils.shard_pytree(
        unreplicated_metrics_state, self._mesh
    )

    return (
        params,
        params_sharding,
        optimizer_state,
        optimizer_state_sharding,
        batch_stats,
        batch_stats_sharding,
        metrics_state,
        metrics_state_sharding,
    )

  def finalize_batch_fn(self, batch):
    """Finalize the batch by making a global array out of the shards."""

    return trainer_utils.make_finalize_batch_fn(self._mesh)(batch)
