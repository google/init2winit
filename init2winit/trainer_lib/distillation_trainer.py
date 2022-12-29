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

"""Distillation trainer for the init2winit project."""
import functools
import itertools
import time

from absl import logging
from init2winit import checkpoint
from init2winit import schedules
from init2winit import utils
from init2winit.init_lib import init_utils
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

_GRAD_CLIP_EPS = 1e-6


def update_teacher(
    optimizer_state,
    params,
    batch_stats,
    metrics_state,
    batch,
    step,
    lr,
    rng,
    local_device_index,
    running_train_cost,
    training_cost,
    grad_clip,
    optimizer_update_fn,
    metrics_update_fn):
  """Single step of the training loop for the teacher.

  This function will later be pmapped so we keep it outside of the Trainer class
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
    local_device_index: an integer that is unique to this device amongst all
      devices on this host, usually in the range [0, jax.local_device_count()].
      It is folded in to `rng` to produce a unique per-device, per-step RNG.
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
    the new training metrics state, and the gradient norm.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  trainer_utils.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost(
        params, batch=batch, batch_stats=batch_stats, dropout_rng=rng)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  # If we are accumulating gradients, we handle gradient synchronization inside
  # the optimizer so that we can only sync when actually updating the model.
  if isinstance(optimizer_state, gradient_accumulator.GradientAccumulatorState):
    cost_value = lax.pmean(cost_value, axis_name='batch')
  else:
    cost_value, grad = lax.pmean((cost_value, grad), axis_name='batch')

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  # TODO(znado): move to inside optax gradient clipping.
  if grad_clip:
    scaled_grad = jax.tree_map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
    grad = jax.lax.cond(grad_norm > grad_clip, lambda _: scaled_grad,
                        lambda _: grad, None)
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats)
  new_params = optax.apply_updates(params, model_updates)

  new_metrics_state = None
  if metrics_state is not None:
    new_metrics_state = metrics_update_fn(metrics_state, step, cost_value, grad,
                                          params, new_params, optimizer_state)

  return (new_optimizer_state, new_params, new_batch_stats,
          running_train_cost + cost_value, new_metrics_state, grad_norm)


def update(optimizer_state, params, batch_stats, metrics_state, batch, step, lr,
           rng, local_device_index, running_train_cost, teacher_params,
           teacher_batch_stats, teacher_model, training_cost, grad_clip,
           optimizer_update_fn, metrics_update_fn):
  """Single step of the training loop for the student.

  This function will later be pmapped so we keep it outside of the Trainer class
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
    local_device_index: an integer that is unique to this device amongst all
      devices on this host, usually in the range [0, jax.local_device_count()].
      It is folded in to `rng` to produce a unique per-device, per-step RNG.
    running_train_cost: the cumulative train cost over some past number of train
      steps. Reset at evaluation time.
    teacher_params: teacher model parameters.
    teacher_batch_stats: teacher model batch stats.
    teacher_model: teacher model in distillation.
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
    the new training metrics state, and the gradient norm.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  trainer_utils.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost(
        params,
        batch=batch,
        batch_stats=batch_stats,
        dropout_rng=rng,
        teacher_params=teacher_params,
        teacher_batch_stats=teacher_batch_stats,
        teacher_model=teacher_model)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  # If we are accumulating gradients, we handle gradient synchronization inside
  # the optimizer so that we can only sync when actually updating the model.
  if isinstance(optimizer_state, gradient_accumulator.GradientAccumulatorState):
    cost_value = lax.pmean(cost_value, axis_name='batch')
  else:
    cost_value, grad = lax.pmean((cost_value, grad), axis_name='batch')

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  # TODO(znado): move to inside optax gradient clipping.
  if grad_clip:
    scaled_grad = jax.tree_map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
    grad = jax.lax.cond(grad_norm > grad_clip, lambda _: scaled_grad,
                        lambda _: grad, None)
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats)
  new_params = optax.apply_updates(params, model_updates)

  new_metrics_state = None
  if metrics_state is not None:
    new_metrics_state = metrics_update_fn(metrics_state, step, cost_value, grad,
                                          params, new_params, optimizer_state)

  return (new_optimizer_state, new_params, new_batch_stats,
          running_train_cost + cost_value, new_metrics_state, grad_norm)


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def __init__(self,
               *args,
               dataset_meta_data=None,
               loss_name=None,
               metrics_name=None,
               **kwargs):
    super(Trainer, self).__init__(*args, **kwargs)
    self.dataset_meta_data = dataset_meta_data
    self.loss_name = loss_name
    self.metrics_name = metrics_name

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

    # Init teacher model.
    teacher_model_cls = models.get_model(self._hps.teacher_model_name)
    teacher_model = teacher_model_cls(self._hps, self.dataset_meta_data,
                                      self.loss_name, self.metrics_name)

    # Init teacher's learning function.
    stretch_factor = 1
    if self._hps.get('total_accumulated_batch_size') is not None:
      stretch_factor = (
          self._hps.total_accumulated_batch_size // self._hps.batch_size)

    self._teacher_lr_fn = schedules.get_schedule_fn(
        self._hps.teacher_lr_hparams,
        max_training_updates=self._num_train_steps // stretch_factor,
        stretch_factor=stretch_factor)

    # Init teacher params and batch stats.
    teacher_unreplicated_params, teacher_unreplicated_batch_stats = init_utils.initialize(
        teacher_model, self._initializer, self._hps, init_rng,
        self._init_logger)
    optimizer_init_fn, _ = optimizers.get_optimizer(
        self._hps, self._model, batch_axis_name='batch')
    teacher_unreplicated_optimizer_state = optimizer_init_fn(
        teacher_unreplicated_params)
    # Replicate teacher's params and batch stats.
    (_, teacher_params, teacher_batch_stats, _, _, _, _,
     _) = checkpoint.replicate_and_maybe_restore_checkpoint(
         teacher_unreplicated_optimizer_state,
         teacher_unreplicated_params,
         teacher_unreplicated_batch_stats,
         None,  # unreplicated_metrics_state
         train_dir='none',  # a fake directory that does not exist
         external_checkpoint_path=None)

    # pmap teacher's update function.
    update_teacher_fn = functools.partial(
        update_teacher,
        training_cost=self._model.training_cost,
        grad_clip=self._hps.get('grad_clip'),
        optimizer_update_fn=self._optimizer_update_fn,
        metrics_update_fn=self._metrics_update_fn)

    self._update_teacher_pmapped = utils.timed(
        jax.pmap(
            update_teacher_fn,
            axis_name='batch',
            in_axes=(0, 0, 0, 0, 0, None, None, None, 0, 0),
            donate_argnums=(0, 1, 2, 8)))

    # Overwrite self._update_pmap (student's update) with teacher model.
    update_fn = functools.partial(
        update,
        teacher_model=teacher_model,
        training_cost=self._model.training_cost,
        grad_clip=self._hps.get('grad_clip'),
        optimizer_update_fn=self._optimizer_update_fn,
        metrics_update_fn=self._metrics_update_fn)
    self._update_pmapped = utils.timed(
        jax.pmap(
            update_fn,
            axis_name='batch',
            in_axes=(0, 0, 0, 0, 0, None, None, None, 0, 0, 0, 0),
            donate_argnums=(0, 1, 2, 8)))

    if jax.process_index() == 0:
      trainer_utils.log_message(
          'Starting training!', self._logging_pool, self._xm_work_unit)

    # Start at the resumed step and continue until we have finished the number
    # of training steps. If building a dataset iterator using a tf.data.Dataset,
    # in the case of a batch size that does not evenly divide the training
    # dataset size, if using `ds.batch(..., drop_remainer=True)` on the training
    # dataset then the final batch in this iterator will be a partial batch.
    # However, if `drop_remainer=False`, then this iterator will always return
    # batches of the same size, and the final batch will have elements from the
    # start of the (num_epochs + 1)-th epoch.
    train_iter = itertools.islice(
        self._dataset.train_iterator_fn(),
        self._global_step,
        self._num_train_steps)


    train_iter = trainer_utils.prefetch_input_pipeline(
        train_iter, self._hps.num_device_prefetches)

    start_time = time.time()
    start_step = self._global_step

    # NOTE(dsuo): record timestamps for run_time since we don't have a duration
    # that we can increment as in the case of train_time.
    self._time_at_prev_eval_end = start_time
    self._prev_eval_step = self._global_step

    for _ in range(start_step, self._num_train_steps):
      with jax.profiler.StepTraceAnnotation('train',
                                            step_num=self._global_step):
        # NOTE(dsuo): to properly profile each step, we must include batch
        # creation in the StepTraceContext (as opposed to putting `train_iter`
        # directly in the top-level for loop).
        batch = next(train_iter)

        if (self._global_step in self._checkpoint_steps
            and jax.process_index() == 0):
          self._save(self._checkpoint_dir, max_to_keep=None)
        # Get both student and teacher's learning rate.
        lr = self._lr_fn(self._global_step)
        teacher_lr = self._teacher_lr_fn(self._global_step)
        # Update the teacher.
        (self._optimizer_state, self._params, self._batch_stats,
         self._sum_train_cost, self._metrics_state,
         self._grad_norm), _ = self._update_teacher_pmapped(
             self._optimizer_state, self._params, self._batch_stats,
             self._metrics_state, batch, self._global_step, teacher_lr, rng,
             self._local_device_indices, self._sum_train_cost)
        # Update the student.
        (self._optimizer_state, self._params, self._batch_stats,
         self._sum_train_cost,
         self._metrics_state, self._grad_norm), _ = self._update_pmapped(
             self._optimizer_state, self._params, self._batch_stats,
             self._metrics_state, batch, self._global_step, lr, rng,
             self._local_device_indices, self._sum_train_cost, teacher_params,
             teacher_batch_stats)
        self._global_step += 1
        # TODO(gdahl, gilmer): consider moving this test up.
        # NB: Since this test is after we increment self._global_step, having 0
        # in eval_steps does nothing.
        if trainer_utils.should_eval(
            self._global_step, self._eval_frequency, self._eval_steps):
          report = self._eval(lr, start_step, start_time)
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
    checkpoint.wait_for_checkpoint_save()
