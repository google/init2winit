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

"""Abstract parent class for all trainers."""
import abc
import functools
import multiprocessing
import os.path
import time

from absl import logging
from init2winit import callbacks
from init2winit import checkpoint
from init2winit import schedules
from init2winit import utils
from init2winit.init_lib import init_utils
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import trainer_utils
from init2winit.training_metrics_grabber import make_training_metrics
import jax
import jax.numpy as jnp
import numpy as np


class BaseTrainer(metaclass=abc.ABCMeta):
  """Abstract parent class for all trainers."""

  def __init__(
      self,
      train_dir,
      model,
      dataset_builder,
      initializer,
      num_train_steps,
      hps,
      rng,
      eval_batch_size,
      eval_num_batches,
      test_num_batches,
      eval_train_num_batches,
      eval_frequency,
      checkpoint_steps,
      early_stopping_target_name=None,
      early_stopping_target_value=None,
      early_stopping_mode=None,
      eval_steps=None,
      metrics_logger=None,
      init_logger=None,
      training_metrics_config=None,
      callback_configs=None,
      external_checkpoint_path=None,
      dataset_meta_data=None,
      loss_name=None,
      metrics_name=None):
    """Main training loop.

    Trains the given network on the specified dataset for the given number of
    epochs. Saves the training curve in train_dir/r=3/results.tsv.

    As a general design principle, we avoid side-effects in functions as much as
    possible.

    Args:
      train_dir: (str) Path of the training directory.
      model: (BaseModel) Model object to be trained.
      dataset_builder: dataset builder returned by datasets.get_dataset.
      initializer: Must have API as defined in initializers.py
      num_train_steps: (int) Number of steps to train on.
      hps: (tf.HParams) Model, initialization and training hparams.
      rng: (jax.random.PRNGKey) Rng seed used in model initialization and data
        shuffling.
      eval_batch_size: the evaluation batch size. If None, use hps.batch_size.
      eval_num_batches: (int) The number of batches used for evaluating on
        validation sets. Set to None to evaluate on the whole eval set.
      test_num_batches: (int) The number of batches used for evaluating on
        test sets. Set to None to evaluate on the whole test set.
      eval_train_num_batches: (int) The number of batches for evaluating on
        train. Set to None to evaluate on the whole training set.
      eval_frequency: (int) Evaluate every k steps.
      checkpoint_steps: List of integers indicating special steps to save
        checkpoints at. These checkpoints do not get used for preemption
        recovery.
      early_stopping_target_name: A string naming the metric to use to perform
        early stopping. If this metric reaches the value
        `early_stopping_target_value`, training will stop. Must include the
        dataset split (ex: validation/error_rate).
      early_stopping_target_value: A float indicating the value at which to stop
        training.
      early_stopping_mode: One of "above" or "below", indicates if we should
        stop when the metric is above or below the threshold value. Example: if
        "above", then training will stop when
        `report[early_stopping_target_name] >= early_stopping_target_value`.
      eval_steps: List of integers indicating which steps to perform evals. If
        provided, eval_frequency will be ignored. Performing an eval implies
        saving a checkpoint that will be used to resume training in the case of
        preemption.
      metrics_logger: Used to log all eval metrics during training. See
        utils.MetricLogger for API definition.
      init_logger: Used for black box initializers that have learning curves.
      training_metrics_config: Dict specifying the configuration of the
        training_metrics_grabber. Set to None to skip logging of advanced
        training metrics.
      callback_configs: List of configs specifying general callbacks to run
        during the eval phase. Empty list means no callbacks are run. See
        callbacks.py for details on what is expected in a config.
      external_checkpoint_path: (str) If this argument is set, we will load the
        optimizer_state, params, batch_stats, and training_metrics from the
        checkpoint at this location.
      dataset_meta_data: meta_data about the dataset. It is not directly used in
        the base trainer. Users are expected to overwrite the initialization
        method in a customimzed trainer to access it.
      loss_name: name of the loss function. Not directly used in base trainer.
        Users are expected to overwrite the initialization method in a
        customimzed trainer to access it.
      metrics_name: Not directly used in the base trainer. Users are expected
        to overwrite the initialization method in a customimzed trainer to
        access it.
    """
    del dataset_meta_data
    del loss_name
    del metrics_name
    self._train_dir = train_dir
    self._model = model
    self._dataset_builder = dataset_builder
    self._initializer = initializer
    self._num_train_steps = num_train_steps
    self._hps = hps
    self._rng = rng
    eval_batch_size = (
        self._hps.batch_size if eval_batch_size is None else eval_batch_size)
    self._eval_batch_size = eval_batch_size
    self._eval_num_batches = eval_num_batches
    self._test_num_batches = test_num_batches
    self._eval_train_num_batches = eval_train_num_batches
    self._eval_frequency = eval_frequency
    self._checkpoint_steps = checkpoint_steps
    self._early_stopping_target_name = early_stopping_target_name
    self._early_stopping_target_value = early_stopping_target_value
    self._early_stopping_mode = early_stopping_mode
    self._eval_steps = eval_steps
    self._metrics_logger = metrics_logger
    self._init_logger = init_logger
    self._training_metrics_config = training_metrics_config
    self._xprof_steps = None
    self._xm_work_unit = None
    if callback_configs is None:
      self._callback_configs = []
    elif isinstance(callback_configs, dict):
      self._callback_configs = [callback_configs]
    else:
      self._callback_configs = callback_configs
    self._external_checkpoint_path = external_checkpoint_path

    # For logging / processing off the main thread
    self._logging_pool = multiprocessing.pool.ThreadPool()

    # Initialized in train() when training starts.
    self._time_at_prev_eval_end = None
    self._prev_eval_step = None

    assert hps.batch_size % (jax.device_count()) == 0
    assert eval_batch_size % (jax.device_count()) == 0

    # Only used if checkpoints_steps is non-empty. Standard checkpoints are
    # saved in train_dir.
    self._checkpoint_dir = os.path.join(self._train_dir, 'checkpoints')

    # During eval, we can donate the 'batch' buffer. We don't donate the
    # 'params' and 'batch_stats' buffers as we don't re-assign those values in
    # eval, we do that only in train.
    self._evaluate_batch_pmapped = jax.pmap(
        self._model.evaluate_batch, axis_name='batch', donate_argnums=(2,))

    # Numpy array of range(0, local_device_count) to send to each device to be
    # folded into the RNG inside each train step to get a unique per-device RNG.
    self._local_device_indices = np.arange(jax.local_device_count())

  def setup_and_maybe_restore(self, init_rng, data_rng, trainer_update_fn):
    """Set up member variables for training and maybe Restore training state.

    This function is useful when setting up all the objects necessary for
    training in init2winit (such as in a colab). _setup_and_maybe_restore is
    likely more useful for subclasses to call, because it takes all the returned
    values from this and sets them as the proper member variables.

    Args:
      init_rng: the jax PRNGKey used for initialization. Should be *the same*
        across hosts!
      data_rng: the jax PRNGKey used for dataset randomness. Should be
        *different* across hosts!
      trainer_update_fn: the function for updating the model.

    Returns:
      A long tuple of the following:
        lr_fn: the learning rate schedule fn.
        optimizer_update_fn: the optax update fn.
        metrics_update_fn: the optional metrics update fn.
        metrics_summary_fn: the optional metrics summary fn.
        optimizer_state: the replicated optimizer state.
        params: the replicated model parameters.
        batch_stats: the replicated (optional) model batch statistics.
        metrics_state: the replicated metric states.
        global_step: the global step to start training at.
        sum_train_cost: the sum of the train costs.
        preemption_count: the number of times training has been preempted.
    """
    # Note that self._global_step refers to the number of gradients
    # calculations, not the number of model updates. This means when using
    # gradient accumulation, one must supply configs where the number of steps
    # are in units of gradient calculations, not model updates, and in post
    # processing one must divide self._global_step by grad_accum_step_multiplier
    # to get the number of updates.
    #
    # The learning rate schedules are all defined in number of model updates. In
    # order to make it clearer in the logging that the LR does not change across
    # gradient calculations (only when updates are applied), we stretch the
    # learning rate schedule by the number of gradient calculations per weight
    # update.
    stretch_factor = 1
    if self._hps.get('total_accumulated_batch_size') is not None:
      stretch_factor = (
          self._hps.total_accumulated_batch_size // self._hps.batch_size)
    lr_fn = schedules.get_schedule_fn(
        self._hps.lr_hparams,
        max_training_updates=self._num_train_steps // stretch_factor,
        stretch_factor=stretch_factor)

    unreplicated_params, unreplicated_batch_stats = init_utils.initialize(
        self._model.flax_module,
        self._initializer,
        self._model.loss_fn,
        self._hps.input_shape,
        self._hps.output_shape,
        self._hps,
        init_rng,
        self._init_logger,
        self._model.get_fake_batch(self._hps))

    if jax.process_index() == 0:
      utils.log_pytree_shape_and_statistics(unreplicated_params)
      logging.info('train_size: %d,', self._hps.train_size)
      utils.tabulate_model(self._model, self._hps)

    optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(
        self._hps, self._model, batch_axis_name='batch')
    unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)

    # Move to host to avoid OOM as restoring from a checkpoint will
    # keep around these buffers on the Device(s). We use numpy instead of
    # jax.device_get for 3 benefits: 1) avoid possible OOM / compilation errors,
    # 2) easier to use if one adds model partitioning, 3) faster eager execution
    # on CPU.
    logging.info('Moving initial data to Host RAM')
    unreplicated_params = jax.tree_util.tree_map(np.array, unreplicated_params)
    unreplicated_batch_stats = jax.tree_util.tree_map(np.array,
                                                      unreplicated_batch_stats)
    unreplicated_optimizer_state = jax.tree_util.tree_map(
        np.array, unreplicated_optimizer_state)

    unreplicated_metrics_state = None
    metrics_update_fn = None
    metrics_summary_fn = None
    if self._training_metrics_config is not None:
      (metrics_init_fn, metrics_update_fn,
       metrics_summary_fn) = make_training_metrics(
           self._num_train_steps, self._hps, **self._training_metrics_config)
      unreplicated_metrics_state = metrics_init_fn(unreplicated_params)

    (optimizer_state,
     params,
     batch_stats,
     metrics_state,
     global_step,
     sum_train_cost,
     preemption_count,
     is_restored) = checkpoint.replicate_and_maybe_restore_checkpoint(
         unreplicated_optimizer_state,
         unreplicated_params,
         unreplicated_batch_stats,
         unreplicated_metrics_state,
         train_dir=self._train_dir,
         external_checkpoint_path=self._external_checkpoint_path)
    if is_restored:
      preemption_count += 1

    # TODO(gdahl): Is there any harm in removing the test and always folding-in
    # global step, even when training fresh at step 0?
    if is_restored:
      # Fold the restored step into the dataset RNG so that we will get a
      # different shuffle each time we restore, so that we do not repeat a
      # previous dataset ordering again after restoring. This is not the only
      # difference in shuffling each pre-emption, because we often times
      # reshuffle the input files each time in a non-deterministic manner.
      #
      # Note that if we are pre-empted more than once per epoch then we will
      # retrain more on the beginning of the training split, because each time
      # we restore we refill the shuffle buffer with the first
      # `shuffle_buffer_size` elements from the training split to continue
      # training.
      #
      # Also note that for evaluating on the training split, because we are
      # reshuffling each time, we will get a new eval_train split each time we
      # are pre-empted.
      data_rng = jax.random.fold_in(data_rng, global_step)

    dataset = self._dataset_builder(
        data_rng,
        self._hps.batch_size,
        eval_batch_size=self._eval_batch_size,
        hps=self._hps)

    update_fn = functools.partial(
        trainer_update_fn,
        training_cost=self._model.training_cost,
        grad_clip=self._hps.get('grad_clip'),
        optimizer_update_fn=optimizer_update_fn,
        metrics_update_fn=metrics_update_fn)
    # in_axes = (
    #     optimizer_state = 0,
    #     params = 0,
    #     batch_stats = 0,
    #     metrics_state = 0,
    #     batch = 0,
    #     step = None,
    #     lr = None,
    #     rng = None,
    #     local_device_index = 0,
    #     running_train_cost = 0,
    #     training_cost,
    #     grad_clip,
    #     optimizer_update_fn,
    #     metrics_state_update_fn)
    # Also, we can donate buffers for 'optimizer', 'batch_stats',
    # 'batch' and 'training_metrics_state' for update's pmapped computation.
    update_pmapped = jax.pmap(
        update_fn,
        axis_name='batch',
        in_axes=(0, 0, 0, 0, 0, None, None, None, 0, 0),
        donate_argnums=(0, 1, 2, 8))
    return (
        lr_fn,
        optimizer_update_fn,
        metrics_update_fn,
        metrics_summary_fn,
        optimizer_state,
        params,
        batch_stats,
        metrics_state,
        global_step,
        sum_train_cost,
        preemption_count,
        dataset,
        update_pmapped)

  def _setup_and_maybe_restore(
      self, init_rng, data_rng, callback_rng, trainer_update_fn):
    """Calls setup_and_maybe_restore and sets return values as member vars.

    Has the side-effects of:
      - setting self._lr_fn
      - initializing and maybe restoring self._optimizer_update_fn.
      - initializing and maybe restoring self._metrics_update_fn.
      - initializing and maybe restoring self._metrics_summary_fn.
      - initializing and maybe restoring self._optimizer_state.
      - initializing and maybe restoring self._params.
      - initializing and maybe restoring self._batch_stats.
      - initializing and maybe restoring self._metrics_state.
      - initializing and maybe restoring self._global_step.
      - initializing and maybe restoring self._sum_train_cost.
      - initializing and maybe restoring self._preemption_count.
      - setting self._dataset
      - setting self._update_pmapped
      - setting self._eval_callbacks

    Args:
      init_rng: the jax PRNGKey used for initialization. Should be *the same*
        across hosts!
      data_rng: the jax PRNGKey used for dataset randomness. Should be
        *different* across hosts!
      callback_rng: the jax PRNGKey used for eval callbacks. Should be
        *different* across hosts!
      trainer_update_fn: the function for updating the model.
    """
    (self._lr_fn,
     self._optimizer_update_fn,
     self._metrics_update_fn,
     self._metrics_summary_fn,
     self._optimizer_state,
     self._params,
     self._batch_stats,
     self._metrics_state,
     self._global_step,
     self._sum_train_cost,
     self._preemption_count,
     self._dataset,
     self._update_pmapped) = self.setup_and_maybe_restore(
         init_rng, data_rng, trainer_update_fn)
    self._eval_callbacks = self._setup_eval_callbacks(callback_rng)

  def _save(self, checkpoint_dir, max_to_keep=1):
    checkpoint.save_unreplicated_checkpoint_background(
        checkpoint_dir,
        self._optimizer_state,
        self._params,
        self._batch_stats,
        self._metrics_state,
        self._global_step,
        self._preemption_count,
        self._sum_train_cost,
        max_to_keep=max_to_keep)

  def _get_step_frequency(self, cur_step, start_step, start_time):
    return float(cur_step - start_step) / (time.time() - start_time)

  def _setup_eval_callbacks(self, callback_rng):
    eval_callbacks = []
    callback_rngs = jax.random.split(callback_rng, len(self._callback_configs))
    for rng, config in zip(callback_rngs, self._callback_configs):
      eval_callback = callbacks.get_callback(config['callback_name'])(
          self._model, self._params, self._batch_stats, self._optimizer_state,
          self._optimizer_update_fn, self._dataset, self._hps, config,
          self._train_dir, rng)
      eval_callbacks.append(eval_callback)
    return eval_callbacks

  def _run_eval_callbacks(self, report):
    for eval_callback in self._eval_callbacks:
      callback_metrics = eval_callback.run_eval(self._params, self._batch_stats,
                                                self._optimizer_state,
                                                self._global_step)
      if set(callback_metrics.keys()).intersection(set(report.keys())):
        raise ValueError('There was a collision between the callback'
                         'metrics and the standard eval metrics keys')
      report.update(callback_metrics)

  def _check_early_stopping(self, report):
    """Check if training should stop early."""
    early_stopping_condition = trainer_utils.check_for_early_stopping(
        self._early_stopping_target_name,
        self._early_stopping_target_value,
        self._early_stopping_mode,
        report)
    if early_stopping_condition:
      if self._early_stopping_mode == 'above':
        comparison_string = '>='
      else:
        comparison_string = '<='
      logging.info(
          'Early stopping because metric %s=%f, reached the target value '
          'of %s %f.',
          self._early_stopping_target_name,
          report[self._early_stopping_target_name],
          comparison_string,
          self._early_stopping_target_value)
    return early_stopping_condition

  def _eval(
      self,
      lr,
      start_step,
      start_time):
    """Evaluate.

    Has the side-effects of:
      - synchronizing self._batch_stats across hosts
      - checkpointing via self._save(self._train_dir)
      - resetting self._sum_train_cost to jnp.zeros
      - resetting self._time_at_prev_eval_end to the current time
      - resetting self._prev_eval_step to self._global_step

    Args:
      lr: the current learning rate.
      start_step: the training start step.
      start_time: the training start time.

    Returns:
      A Dict[str, Any] eval report, originally created in
      trainer_utils.eval_metrics.
    """
    time_since_last_eval = time.time() - self._time_at_prev_eval_end
    self._batch_stats = trainer_utils.maybe_sync_batchnorm_stats(
        self._batch_stats)
    report, eval_time = trainer_utils.eval_metrics(
        self._params,
        self._batch_stats,
        self._dataset,
        self._eval_num_batches,
        self._test_num_batches,
        self._eval_train_num_batches,
        self._evaluate_batch_pmapped)
    self._run_eval_callbacks(report)
    if jax.process_index() == 0:
      self._save(self._train_dir)
    steps_since_last_eval = self._global_step - self._prev_eval_step
    steps_per_sec_no_eval = steps_since_last_eval / time_since_last_eval
    run_time = time.time() - self._time_at_prev_eval_end
    steps_per_sec = steps_since_last_eval / run_time

    mean_train_cost = self._sum_train_cost.mean().item() / max(
        1, self._global_step - self._prev_eval_step)
    self._sum_train_cost = jnp.zeros(jax.local_device_count())
    epoch = self._global_step * self._hps.batch_size // self._hps.train_size
    overall_steps_per_sec = self._get_step_frequency(
        self._global_step, start_step, start_time)
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
        run_time=run_time)
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
          eval_time)
      trainer_utils.log_epoch_report(report, self._metrics_logger)
      trainer_utils.maybe_log_training_metrics(self._metrics_state,
                                               self._metrics_summary_fn,
                                               self._metrics_logger)

    self._time_at_prev_eval_end = time.time()
    self._prev_eval_step = self._global_step
    return report

  @abc.abstractmethod
  def train(self):
    """All training logic.

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
