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

"""Meta-optimization trainer for the init2winit project.

For reference of the method see https://arxiv.org/abs/2301.07902
"""

import collections
import functools
import itertools
import time

from absl import logging
import flax
from flax import jax_utils
from init2winit import utils
from init2winit.model_lib import model_utils
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import meta_opt_utils
from init2winit.trainer_lib import trainer_utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax


_GRAD_CLIP_EPS = 1e-6

TrainState = collections.namedtuple(
    'TrainState',
    [
        'optimizer_state',
        'params',
        'batch_stats',
        'metrics_state',
        'global_step',
        'lr',
        'rng',
        'local_device_indices',
        'sum_train_cost',
    ],
)


def _training_cost_fn(
    params,
    batch_stats,
    batch,
    step,
    rng,
    local_device_index,
    training_cost,
):
  """Training cost function.

  Args:
    params: a dict of trainable model parameters.
    batch_stats: a dict of non-trainable model state.
    batch: per-device batch of data.
    step: current global step of the update.
    rng: the RNG used for calling the model.
    local_device_index: an integer that is unique to this device amongst all
      devices on this host, usually in the range [0, jax.local_device_count()].
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates.
  Returns:
    cost: the training cost.
  """

  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  cost = training_cost(
      params, batch=batch, batch_stats=batch_stats, dropout_rng=rng
  )

  cost, _ = lax.pmean(cost, axis_name='batch')

  return cost


def _compute_controls(etas, disturbances):
  """Compute the control signal.

  The control signal is a linear transformation of the disturbances defined by
  the coefficients etas.

  Args:
    etas: scalar coefficients used to compute the control signal.
    disturbances: a dict of non-trainable parameters, has the same shape as
      gradients.

  Returns:
    the control signal.

  """
  return jax.tree_map(
      lambda a: jnp.sum(a * jnp.expand_dims(etas, range(1, a.ndim)), axis=0),
      disturbances,
  )


# index 0 is the least recent
def _hallucinated_step(
    update_fn, etas, disturbances, num_disturbances, train_state_h, batch
):  # pylint:disable=invalid-name
  """Hallucinate a single step.
  
  Update the model params by the sum of the gradient update and the control 
  signals.

  Args:
    update_fn: a function used to compute the gradient update without the 
      controls.
    etas: scalar coefficients used to compute the control signal.
    disturbances: a dict of non-trainable parameters, has the same shape as
      gradients.
    num_disturbances: number of disturbances used in the control signal.
    train_state_h: the train state at step h (how many steps we have rolled
      forward).
    batch: current batch used in update_fn
  Returns:
    the train state at step h + 1, and the new train loss
  """

  train_state, h = train_state_h
  # update train state with same update function as the actual training step
  (
      new_optimizer_state,
      new_params,
      new_batch_stats,
      sum_train_cost,
      new_metrics_state,
      _,
      _,
      _,
      cost,
  ) = update_fn(
      train_state.optimizer_state,
      train_state.params,
      train_state.batch_stats,
      train_state.metrics_state,
      batch,
      train_state.global_step,
      train_state.lr,
      train_state.rng,
      train_state.local_device_indices,
      train_state.sum_train_cost,
  )

  # compute controls with disturbances[h:h+num_disturbances-1]
  disturbance_slice = meta_opt_utils.get_pytree_history_window(
      disturbances, h, num_disturbances
  )

  controls = _compute_controls(etas, disturbance_slice)

  # update params with controls
  new_params = jax.tree_map(
      lambda param, control: param - control, new_params, controls
  )

  new_train_state = TrainState(
      optimizer_state=new_optimizer_state,
      params=new_params,
      batch_stats=new_batch_stats,
      metrics_state=new_metrics_state,
      global_step=train_state.global_step + 1,
      lr=train_state.lr,
      rng=train_state.rng,
      local_device_indices=train_state.local_device_indices,
      sum_train_cost=sum_train_cost,
  )

  return (new_train_state, h + 1), cost


def _hallucinate_fn(
    hallucinated_step_fn,
    update_fn,
    training_cost,
    num_disturbances,
    hallucinate_steps,
    etas,
    disturbances,
    train_state,
    batches,
):  # pylint:disable=invalid-name
  """Hallucinate for hallucinate_steps and returns the terminal cost.

  Args:
    hallucinated_step_fn: a function that takes in a train state and a batch,
      and hallucinate for one step.
    update_fn: a function used to compute the gradient update without the 
      controls.
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates.
    num_disturbances: number of disturbances to use in the control signal.
    hallucinate_steps: number of steps to hallucinate.
    etas: scalar coefficients used to compute the control signal.
    disturbances: a dict of non-trainable parameters, has the same shape as
      gradients.
    train_state: the initial train state to hallucinate from.
    batches: a list of batches to use in hallucination.
  Returns:
    The terminal cost.
  """
  step_fn = functools.partial(
      hallucinated_step_fn, update_fn, etas, disturbances, num_disturbances)

  for h in range(hallucinate_steps):
    batch = meta_opt_utils.get_pytree_history_index(batches, h)
    (train_state, _), _ = step_fn((train_state, h), batch)

  batch = meta_opt_utils.get_pytree_history_index(batches, hallucinate_steps)
  return (
      training_cost(
          train_state.params,
          train_state.batch_stats,
          batch,
          train_state.global_step,
          train_state.rng,
          train_state.local_device_indices,
      ).at[0].get()
  )


def update(
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
    metrics_update_fn,
    axis_name='batch'
):
  """Single step of the training loop.

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
      be differentiated to generate updates. Takes (`params`, `batch`,
      `batch_stats`, `dropout_rng`) as inputs.
    grad_clip: Clip the l2 norm of the gradient at the specified value. For
      minibatches with gradient norm ||g||_2 > grad_clip, we rescale g to the
      value g / ||g||_2 * grad_clip. If None, then no clipping will be applied.
    optimizer_update_fn: the optimizer update function.
    metrics_update_fn: the training metrics update function.
    axis_name: axis name used by pmap.

  Returns:
    A tuple of the new optimizer, the new batch stats, the scalar training cost,
    the new training metrics state, the gradient norm, the update to the model
    params, the gradient, and the training cost at the current step.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  trainer_utils.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost(
        params, batch=batch, batch_stats=batch_stats, dropout_rng=rng
    )

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', flax.core.FrozenDict())

  if axis_name is not None:
    # aggregate grads
    grad = lax.pmean((grad), axis_name=axis_name)

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  # TODO(znado): move to inside optax gradient clipping.
  if grad_clip:
    scaled_grad = jax.tree_map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad
    )
    grad = jax.lax.cond(
        grad_norm > grad_clip, lambda _: scaled_grad, lambda _: grad, None
    )
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats,
      cost_fn=opt_cost,
      grad_fn=grad_fn,
  )
  new_params = optax.apply_updates(params, model_updates)

  new_metrics_state = None
  if metrics_state is not None:
    new_metrics_state = metrics_update_fn(
        metrics_state,
        step,
        cost_value,
        grad,
        params,
        new_params,
        optimizer_state,
        new_batch_stats,
    )

  updates = jax.tree_map(lambda x: -x / lr, model_updates)

  return (
      new_optimizer_state,
      new_params,
      new_batch_stats,
      running_train_cost + cost_value,
      new_metrics_state,
      grad_norm,
      updates,
      grad,
      cost_value,
  )


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._etas = None

  def _eval(self, lr, start_step, start_time, save=False):
    time_since_last_eval = time.time() - self._time_at_prev_eval_end
    self._batch_stats = trainer_utils.maybe_sync_batchnorm_stats(
        self._batch_stats
    )

    if self._eval_use_ema:
      if isinstance(self._optimizer_state, optax.InjectHyperparamsState):
        eval_params = self._optimizer_state.inner_state[0][0].ema
      elif isinstance(
          self._optimizer_state, gradient_accumulator.GradientAccumulatorState
      ):
        eval_params = self._optimizer_state.base_state.inner_state[0][0].ema
      else:
        raise ValueError(
            'EMA computation should be the very first transformation in defined'
            ' kitchensink optimizer.'
        )
    else:
      eval_params = self._params

    report, eval_time = trainer_utils.eval_metrics(
        eval_params,
        self._batch_stats,
        self._dataset,
        self._eval_num_batches,
        self._test_num_batches,
        self._eval_train_num_batches,
        self._evaluate_batch_pmapped)
    self._run_eval_callbacks(report)
    if save:
      self._save(self._train_dir)
    steps_since_last_eval = self._global_step - self._prev_eval_step
    steps_per_sec_no_eval = steps_since_last_eval / time_since_last_eval
    run_time = time.time() - self._time_at_prev_eval_end
    steps_per_sec = steps_since_last_eval / run_time

    mean_train_cost = jax.lax.pmean(self._sum_train_cost, axis_name=[])[
        0
    ].item() / max(1, self._global_step - self._prev_eval_step)
    self._sum_train_cost = jax_utils.replicate(0.0)
    epoch = self._global_step * self._hps.batch_size // self._hps.train_size
    overall_steps_per_sec = self._get_step_frequency(
        self._global_step, start_step, start_time
    )
    report.update(
        learning_rate=float(lr),
        global_step=self._global_step,
        epoch=epoch,
        grad_norm=np.mean(self._grad_norm),
        preemption_count=self._preemption_count,
        train_cost=mean_train_cost,
        overall_steps_per_sec=overall_steps_per_sec,
        steps_per_sec_no_eval=steps_per_sec_no_eval,
        steps_per_sec=steps_per_sec,
        eval_time=eval_time,
        run_time_no_eval=time_since_last_eval,
        run_time=run_time,
    )

    ############# LOGGING ################
    for i in range(self._etas.shape[0]):
      report[f'eta {i}'] = self._etas.at[i].get()
    if jax.process_index() == 0:
      trainer_utils.log_eta(
          self._logging_pool,
          self._xm_work_unit,
          self._global_step,
          steps_per_sec_no_eval,
          self._num_train_steps,
          start_time,
          self._eval_frequency,
          self._eval_steps,
          eval_time,
      )
      trainer_utils.log_epoch_report(report, self._metrics_logger)
      trainer_utils.maybe_log_training_metrics(
          self._metrics_state, self._metrics_summary_fn, self._metrics_logger
      )

    self._time_at_prev_eval_end = time.time()
    self._prev_eval_step = self._global_step
    return report

  def train(self):
    """All training logic.

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
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
          'Starting training!', self._logging_pool, self._xm_work_unit
      )

    # Start at the resumed step and continue until we have finished the number
    # of training steps.
    if self._hps.get('use_grain'):
      train_iter = itertools.islice(
          self._dataset.train_iterator_fn(), self._num_train_steps
      )
    else:
      train_iter = itertools.islice(
          self._dataset.train_iterator_fn(),
          self._global_step,
          self._num_train_steps,
      )


    train_iter = trainer_utils.prefetch_input_pipeline(
        train_iter, self._hps.num_device_prefetches
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

    self._time_at_prev_eval_end = start_time
    self._prev_eval_step = self._global_step

    if self._global_step in self._checkpoint_steps:
      self._save(self._checkpoint_dir, max_to_keep=None)

    # initialize hps for meta optimization.
    lr = self._hps.lr_hparams.base_lr
    meta_lr = self._hps.opt_hparams.meta_lr
    num_disturbances = self._hps.opt_hparams.num_disturbances
    hallucinate_steps = self._hps.opt_hparams.hallucinate_steps
    self._etas = jnp.array([self._hps.opt_hparams.eta_init] * num_disturbances)
    disturbance_clip = self._hps.opt_hparams.disturbance_clip
    etas_clip = self._hps.opt_hparams.etas_clip

    # optimizer for updating the etas, clip element-wise
    etas_optimizer = optax.chain(optax.adam(learning_rate=meta_lr),
                                 optax.clip(etas_clip))
    etas_opt_state = etas_optimizer.init(self._etas)
    use_updates = self._hps.opt_hparams.use_updates

    # NOTE(dsuo): initialize histories for meta optimization.
    # Convention: 0 index is earliest update.
    # disturbances is a list of jax arrays.
    disturbances = meta_opt_utils.init_pytree_history(
        self._params, hallucinate_steps + num_disturbances)  # pylint: disable=protected-access
    init_train_state = TrainState(
        self._optimizer_state,
        self._params,
        self._batch_stats,
        self._metrics_state,
        self._global_step,
        lr,
        rng,
        self._local_device_indices,
        self._sum_train_cost,
    )

    train_states = meta_opt_utils.init_pytree_history(
        init_train_state, hallucinate_steps + 1)
    batches = None

    training_cost_fn = functools.partial(
        _training_cost_fn,
        training_cost=self._model.training_cost,
    )
    # in_axes = (
    #     params = 0,
    #     batch_stats = 0,
    #     batch = 0,
    #     step = None,
    #     rng = None,
    #     local_device_index = 0,
    #     training_cost)

    training_cost_fn_pmapped = jax.pmap(
        training_cost_fn,
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, 0),
    )

    hallucinate_fn = functools.partial(
        _hallucinate_fn,
        _hallucinated_step,
        self._update_pmapped,
        training_cost_fn_pmapped,
        num_disturbances,
        hallucinate_steps,
    )

    for step in range(start_step, self._num_train_steps):
      with jax.profiler.StepTraceAnnotation(
          'train', step_num=self._global_step
      ):
        # NOTE(dsuo): to properly profile each step, we must include batch
        # creation in the StepTraceContext (as opposed to putting `train_iter`
        # directly in the top-level for loop).
        batch = next(train_iter)
        if batches is None:
          batches = meta_opt_utils.init_pytree_history_zeros(
              batch, hallucinate_steps + 1
          )

        lr = self._lr_fn(self._global_step)

        (
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._sum_train_cost,
            self._metrics_state,
            self._grad_norm,
            updates,
            grad,
            _,
        ) = self._update_pmapped(
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._metrics_state,
            batch,
            self._global_step,
            lr,
            rng,
            self._local_device_indices,
            self._sum_train_cost,
        )
        self._global_step += 1

        # clip disturbance coordinate_wise
        clipped_grad = jax.tree_map(lambda g: jnp.clip(g, -disturbance_clip,
                                                       disturbance_clip), grad)

        # update disturbances
        disturbances = meta_opt_utils.roll_pytree_history(
            disturbances, shift=-1)
        if use_updates:
          disturbances = meta_opt_utils.update_pytree_history(
              disturbances, updates, index=-1)
        else:
          disturbances = meta_opt_utils.update_pytree_history(
              disturbances, clipped_grad, index=-1)

        # update batches
        batches = meta_opt_utils.roll_pytree_history(batches, shift=-1)
        batches = meta_opt_utils.update_pytree_history(
            batches, batch, index=-1)

        # compute control signal.
        controls = _compute_controls(
            self._etas, meta_opt_utils.get_pytree_history_window(
                disturbances, -num_disturbances, num_disturbances))

        self._params = jax.tree_map(
            lambda param, control: param - control, self._params, controls
        )

        # Compute grad w.r.t. etas.
        if step >= hallucinate_steps + num_disturbances:
          init_hallucinate_state = meta_opt_utils.get_pytree_history_index(
              train_states, 0
          )
          eta_grad = jax.grad(hallucinate_fn)(
              self._etas, disturbances, init_hallucinate_state, batches
          )

          # update eta
          etas_updates, etas_opt_state = etas_optimizer.update(eta_grad,
                                                               etas_opt_state,
                                                               self._etas)
          self._etas = optax.apply_updates(self._etas, etas_updates)

        # update train state history
        train_states = meta_opt_utils.roll_pytree_history(
            train_states, shift=-1)
        new_train_state = TrainState(
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._metrics_state,
            self._global_step,
            lr,
            rng,
            self._local_device_indices,
            self._sum_train_cost,
        )

        train_states = meta_opt_utils.update_pytree_history(
            train_states, new_train_state, index=-1
        )

        lr = self._optimizer_state.hyperparams['learning_rate'][0]
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
