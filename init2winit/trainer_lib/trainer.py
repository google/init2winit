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

"""Standard trainer for the init2winit project."""

import functools

from init2winit import schedules
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
import jax
import jax.numpy as jnp
import optax


_GRAD_CLIP_EPS = 1e-6

NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def update(
    optimizer_state,
    params,
    batch_stats,
    metrics_state,
    batch,
    step,
    lr,
    rng,
    running_train_cost,
    training_cost,
    grad_clip,
    optimizer_update_fn,
    metrics_update_fn):
  """Single step of the training loop.

  This function will later be jitted so we keep it outside of the Trainer class
  to avoid the temptation to introduce side-effects.

  Args:
    optimizer_state: the optax optimizer state.
    params: a dict of trainable model parameters. Passed into training_cost(...)
      which then passes into flax_module.apply() as {'params': params} as part
      of the variables dict.
    batch_stats: a dict of non-trainable model state. Passed into
      training_cost(...) which then passes into flax_module.apply() as
      {'batch_stats': batch_stats} as part of the variables dict.
    metrics_state: a pytree of training metrics state.
    batch: the per-device batch of data to process.
    step: the current global step of this update. Used to fold in to `rng` to
      produce a unique per-device, per-step RNG.
    lr: the floating point learning rate for this step.
    rng: the RNG used for calling the model. `step` and `local_device_index`
      will be folded into this to produce a unique per-device, per-step RNG.
    running_train_cost: the cumulative train cost over some past number of train
      steps. Reset at evaluation time.
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates. Takes
      (`params`, `batch`, `batch_stats`, `dropout_rng`) as inputs.
    grad_clip: Clip the l2 norm of the gradient at the specified value. For
      minibatches with gradient norm ||g||_2 > grad_clip, we rescale g to the
      value g / ||g||_2 * grad_clip. If None, then no clipping will be applied.
    optimizer_update_fn: the optimizer update function.
    metrics_update_fn: the training metrics update function.

  Returns:
    A tuple of the new optimizer, the new batch stats, the scalar training cost,
    the new training metrics state, the gradient norm, and the update norm.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)

  optimizer_state = trainer_utils.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost(
        params, batch=batch, batch_stats=batch_stats, dropout_rng=rng)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  # TODO(znado): move to inside optax gradient clipping.
  if grad_clip:
    scaled_grad = jax.tree.map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
    grad = jax.lax.cond(grad_norm > grad_clip, lambda _: scaled_grad,
                        lambda _: grad, None)
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats,
      cost_fn=opt_cost,
      grad_fn=grad_fn,
      value=cost_value)

  update_norm = jnp.sqrt(model_utils.l2_regularization(model_updates, 0))
  new_params = optax.apply_updates(params, model_updates)

  new_metrics_state = None
  if metrics_state is not None:
    new_metrics_state = metrics_update_fn(metrics_state, step, cost_value, grad,
                                          params, new_params, optimizer_state,
                                          new_batch_stats)

  return (new_optimizer_state, new_params, new_batch_stats,
          new_metrics_state, running_train_cost + cost_value,
          grad_norm, update_norm)


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._optimizer_init_fn = None

  def init_optimizer_state(self, model, params, batch_stats, hps, rng):
    del batch_stats
    del rng

    stretch_factor = 1
    if hps.get('total_accumulated_batch_size') is not None:
      stretch_factor = (hps.total_accumulated_batch_size // hps.batch_size)

    self._lr_fn = schedules.get_schedule_fn(
        self._hps.lr_hparams,
        max_training_updates=self._num_train_steps // stretch_factor,
        stretch_factor=stretch_factor)

    self._optimizer_init_fn, self._optimizer_update_fn = (
        optimizers.get_optimizer(
            hps, model, batch_axis_name='batch'
        )
    )
    unreplicated_optimizer_state = self._optimizer_init_fn(params)
    return unreplicated_optimizer_state, self._optimizer_update_fn

  def update_params(self,
                    model,
                    hps,
                    optimizer_state,
                    params,
                    batch_stats,
                    metrics_state,
                    batch,
                    global_step,
                    rng,
                    sum_train_cost):
    """Sets up the train state."""

    optimizer_state, _ = optimizer_state

    update_fn = functools.partial(
        update,
        training_cost=model.training_cost,
        grad_clip=hps.get('grad_clip'),
        optimizer_update_fn=self._optimizer_update_fn,
        metrics_update_fn=self._metrics_update_fn,
    )

    # We donate optimizer_state, params and batch_stats in jitted computation.
    # This helps reduce memory usage as outputs corresponding to these inputs
    # arguments can re-use the memory.
    update_jitted = jax.jit(
        update_fn,
        donate_argnums=(0, 1, 2),
        in_shardings=(
            self._optimizer_state_sharding,
            self._params_sharding,
            self._batch_stats_sharding,
            self._metrics_state_sharding,
            NamedSharding(self._mesh, jax.sharding.PartitionSpec('devices')),
            None, None, None, None
        ),
        out_shardings=(
            self._optimizer_state_sharding,
            self._params_sharding,
            self._batch_stats_sharding,
            self._metrics_state_sharding,
            None,
            None,
            None,
        ),
    )

    lr = self._lr_fn(global_step)
    # It looks like we are reusing an rng key, but we aren't.
    (
        optimizer_state,
        params,
        batch_stats,
        metrics_state,
        sum_train_cost,
        grad_norm,
        update_norm,
    ) = update_jitted(
        optimizer_state,
        params,
        batch_stats,
        self._metrics_state,
        batch,
        global_step,
        lr,
        rng,
        sum_train_cost,
    )

    return (
        optimizer_state,
        params,
        batch_stats,
        metrics_state,
        sum_train_cost,
        grad_norm,
        update_norm,
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

    # Because we always store checkpoint without sharding annotations, we
    # restore it on host and then shard it. In order to propagate
    # annotations from params to the optimizer state, we need to initialize the
    # optimizer state with the sharded params and then device_put
    # the unreplicated_optimizer_state using the resulting annotations.
    optimizer_state_sharding = jax.tree_util.tree_map(
        lambda x: x.sharding
        if isinstance(x.sharding, NamedSharding)
        else NamedSharding(self._mesh, P()),
        self._optimizer_init_fn(params),
    )

    _, optimizer_state = data_utils.shard_pytree(
        unreplicated_optimizer_state, self._mesh, optimizer_state_sharding
    )

    batch_stats_sharding, batch_stats = data_utils.shard_pytree(
        unreplicated_batch_stats, self._mesh)
    metrics_state_sharding, metrics_state = data_utils.shard_pytree(
        unreplicated_metrics_state, self._mesh)

    return (params, params_sharding,
            optimizer_state, optimizer_state_sharding,
            batch_stats, batch_stats_sharding,
            metrics_state, metrics_state_sharding)


