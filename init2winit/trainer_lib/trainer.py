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
import itertools
import time

from absl import logging
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
import jax
import jax.numpy as jnp
import optax

_GRAD_CLIP_EPS = 1e-6


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
          running_train_cost + cost_value, new_metrics_state,
          grad_norm, update_norm)


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def train(self):
    """All training logic.

    The only side-effects are:
      - Initiailizing self._time_at_prev_eval_end to the current time
      - Initiailizing self._prev_eval_step to the current step

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
    # NOTE: the initialization RNG should *not* be per-host, as this will create
    # different sets of weights per host. However, all other RNGs should be
    # per-host.
    # TODO(znado,gilmer,gdahl): implement replicating the same initialization
    # across hosts.
    rng, init_rng = jax.random.split(self._rng)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, data_rng = jax.random.split(rng)
    rng, callback_rng = jax.random.split(rng)

    if jax.process_index() == 0:
      logging.info('Let the training begin!')
      logging.info('Dataset input shape: %r', self._hps.input_shape)
      logging.info('Hyperparameters: %s', self._hps)

    self._setup_and_maybe_restore(init_rng, data_rng, callback_rng, update)

    if jax.process_index() == 0:
      trainer_utils.log_message(
          'Starting training!', self._logging_pool, self._xm_work_unit)

    train_iter = itertools.islice(
        self._dataset.train_iterator_fn(),
        self._num_train_steps,
    )


    if self._data_selector:
      train_iter = self._data_selector(
          train_iter,
          optimizer_state=self._optimizer_state,
          params=self._params,
          batch_stats=self._batch_stats,
          hps=self._hps,
          global_step=self._global_step,
          constant_base_rng=rng)

    start_time = time.time()
    start_step = self._global_step

    # NOTE(dsuo): record timestamps for run_time since we don't have a duration
    # that we can increment as in the case of train_time.
    self._time_at_prev_eval_end = start_time
    self._prev_eval_step = self._global_step

    if self._global_step in self._checkpoint_steps:
      self._save(self._checkpoint_dir, max_to_keep=None)

    make_global_array_fn = functools.partial(
        data_utils.make_global_array, mesh=self._mesh
    )

    for _ in range(start_step, self._num_train_steps):
      with jax.profiler.StepTraceAnnotation(
          'train', step_num=self._global_step
      ):
        # NOTE(dsuo): to properly profile each step, we must include batch
        # creation in the StepTraceContext (as opposed to putting `train_iter`
        # directly in the top-level for loop).
        batch = next(train_iter)
        batch = jax.tree_util.tree_map(make_global_array_fn, batch)

        lr = self._lr_fn(self._global_step)
        # It looks like we are reusing an rng key, but we aren't.
        (
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._sum_train_cost,
            self._metrics_state,
            self._grad_norm,
            self._update_norm,
        ) = self._update_jitted(
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._metrics_state,
            batch,
            self._global_step,
            lr,
            rng,
            self._sum_train_cost,
        )
        self._global_step += 1
        if self._global_step in self._checkpoint_steps:
          self._save(self._checkpoint_dir, max_to_keep=None)

        lr = trainer_utils.fetch_learning_rate(self._optimizer_state)
        # TODO(gdahl, gilmer): consider moving this test up.
        # NB: Since this test is after we increment self._global_step, having 0
        # in eval_steps does nothing.
        if trainer_utils.should_eval(
            self._global_step, self._eval_frequency, self._eval_steps):
          try:
            report = self._eval(lr, start_step, start_time)
          except utils.TrainingDivergedError as e:
            self.wait_until_orbax_checkpointer_finished()
            raise utils.TrainingDivergedError(
                f'divergence at step {self._global_step}'
            ) from e
          yield report
          if self._check_early_stopping(report):
            return

    # Always log and checkpoint on host 0 at the end of training.
    # If we moved where in the loop body evals happen then we would not need
    # this test.
    if self._prev_eval_step != self._num_train_steps:
      report = self._eval(lr, start_step, start_time)
      yield report
    # To make sure the last checkpoint was correctly saved.
    self.wait_until_orbax_checkpointer_finished()

