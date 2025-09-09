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

"""Abstract parent class for all trainers."""

import abc
import itertools
import multiprocessing
import os.path
import time

from absl import logging
from init2winit import callbacks
from init2winit import checkpoint
from init2winit import utils
from init2winit.model_lib import model_utils
from init2winit.trainer_lib import trainer_utils
from init2winit.trainer_lib import training_algorithm
from init2winit.training_metrics_grabber import make_training_metrics
import jax
import orbax.checkpoint as orbax_checkpoint


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
      eval_use_ema,
      eval_num_batches,
      test_num_batches,
      eval_train_num_batches,
      eval_frequency,
      checkpoint_steps,
      early_stopping_target_name=None,
      early_stopping_target_value=None,
      early_stopping_mode=None,
      early_stopping_min_steps=0,
      eval_steps=None,
      metrics_logger=None,
      init_logger=None,
      training_metrics_config=None,
      callback_configs=None,
      external_checkpoint_path=None,
      dataset_meta_data=None,
      loss_name=None,
      metrics_name=None,
      data_selector=None,
      training_algorithm_class=training_algorithm.OptaxTrainingAlgorithm,
  ):
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
      eval_use_ema: if True evals will use ema of params.
      eval_num_batches: (int) The number of batches used for evaluating on
        validation sets. Set to None to evaluate on the whole eval set.
      test_num_batches: (int) The number of batches used for evaluating on test
        sets. Set to None to evaluate on the whole test set.
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
      early_stopping_min_steps: Only allows early stopping after at least this
        many steps.
      eval_steps: List of integers indicating which steps to perform evals. If
        provided, eval_frequency will be ignored. Performing an eval implies
        saving a checkpoint that will be used to resume training in the case of
        preemption.
      metrics_logger: Used to log all eval metrics during training. See
        utils.MetricLogger for API definition.
      init_logger: Used for black box initializers that have learning curves.
      training_metrics_config: Dict specifying the configuration of the
        training_metrics_grabber. Set to None to skip logging of advanced
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
      metrics_name: Not directly used in the base trainer. Users are expected to
        overwrite the initialization method in a customimzed trainer to access
        it.
      data_selector: data selection function returned by
        datasets.get_data_selector.
      training_algorithm_class: Class of training algorithm to use.
    """
    del dataset_meta_data
    del loss_name
    del metrics_name
    self._train_dir = train_dir
    self._model = model
    self._dataset_builder = dataset_builder
    self._data_selector = data_selector
    self._initializer = initializer
    self._num_train_steps = num_train_steps
    self._hps = hps
    self._rng = rng
    eval_batch_size = (
        self._hps.batch_size if eval_batch_size is None else eval_batch_size
    )
    self._eval_batch_size = eval_batch_size
    self._eval_use_ema = eval_use_ema
    self._eval_num_batches = eval_num_batches
    self._test_num_batches = test_num_batches
    self._eval_train_num_batches = eval_train_num_batches
    self._eval_frequency = eval_frequency
    self._checkpoint_steps = checkpoint_steps
    self._orbax_checkpointer = orbax_checkpoint.AsyncCheckpointer(
        orbax_checkpoint.PyTreeCheckpointHandler(use_ocdbt=False),
        timeout_secs=600,
        file_options=orbax_checkpoint.checkpoint_manager.FileOptions(
            path_permission_mode=0o775),
    )
    self._early_stopping_target_name = early_stopping_target_name
    self._early_stopping_target_value = early_stopping_target_value
    self._early_stopping_mode = early_stopping_mode
    self._early_stopping_min_steps = early_stopping_min_steps
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
    self._evaluate_batch_jitted = jax.jit(
        self._model.evaluate_batch, donate_argnums=(2,))

    # Creates a 1-d mesh with all devices available globally.
    self._mesh = model_utils.get_default_mesh()

    # Set training algorithm class.
    self._training_algorithm_class = training_algorithm_class
    logging.info('Using training algorithm class: %s',
                 self._training_algorithm_class)

  def wait_until_orbax_checkpointer_finished(self):
    self._orbax_checkpointer.wait_until_finished()

  def log_model_info(self, unreplicated_params):
    if jax.process_index() == 0:
      utils.log_pytree_shape_and_statistics(unreplicated_params)
      logging.info('train_size: %d,', self._hps.train_size)
      utils.tabulate_model(self._model, self._hps)

  def maybe_restore_from_checkpoint(self,
                                    unreplicated_optimizer_state,
                                    unreplicated_params,
                                    unreplicated_batch_stats,
                                    unreplicated_metrics_state):
    """Restores the training state from a checkpoint if one exists.

    Args:
      unreplicated_optimizer_state: (optax.OptState) The optimizer state.
      unreplicated_params: (FrozenDict) The model parameters.
      unreplicated_batch_stats: (FrozenDict) The batch statistics.
      unreplicated_metrics_state: (FrozenDict) The metrics state.

    Returns:
      The restored optimizer state, parameters, batch statistics, and metrics
      state.
    """
    (
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
        self._global_step,
        self._sum_train_cost,
        self._preemption_count,
        self._is_restored,
    ) = checkpoint.maybe_restore_checkpoint(
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
        train_dir=self._train_dir,
        external_checkpoint_path=self._external_checkpoint_path,
        orbax_checkpointer=self._orbax_checkpointer,
    )

    if self._is_restored:
      self._preemption_count += 1

    return (
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
    )

  def setup_data_loader(self, data_rng, global_step):
    """Sets up the data loader.

    Args:
      data_rng: (jax.random.PRNGKey) Rng seed used in data shuffling.
      global_step: (int) The global step.

    Returns:
      The dataset.
    """
    # TODO(gdahl): Is there any harm in removing the test and always folding-in
    # global step, even when training fresh at step 0?
    if self._is_restored and not self._hps.get('use_grain'):
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
      #
      # We don't change the RNG when using Grain, because it has its own
      # preemption recovery logic.
      data_rng = jax.random.fold_in(data_rng, global_step)

    if self._hps.get('use_grain'):
      # Datasets that support Grain must allow the additional `global_step`
      # argument.
      dataset = self._dataset_builder(
          data_rng,
          self._hps.batch_size,
          eval_batch_size=self._eval_batch_size,
          hps=self._hps,
          global_step=global_step,
      )
    else:
      dataset = self._dataset_builder(
          data_rng,
          self._hps.batch_size,
          eval_batch_size=self._eval_batch_size,
          hps=self._hps,
      )

    return dataset

  def _save(self, checkpoint_dir, max_to_keep=1):
    if utils.use_mock_tpu_backend():
      logging.info('Skip saving checkpoint when running with mock backend.')
      return

    checkpoint.save_unreplicated_checkpoint(
        checkpoint_dir,
        self._optimizer_state,
        self._params,
        self._batch_stats,
        self._metrics_state,
        self._global_step,
        self._preemption_count,
        self._sum_train_cost,
        self._orbax_checkpointer,
        max_to_keep=max_to_keep,
    )

  def _get_step_frequency(self, cur_step, start_step, start_time):
    return float(cur_step - start_step) / (time.time() - start_time)

  def _setup_eval_callbacks(self, callback_rng):
    """Sets up the eval callbacks."""
    eval_callbacks = []
    callback_rngs = jax.random.split(callback_rng, len(self._callback_configs))
    for rng, config in zip(callback_rngs, self._callback_configs):
      logging.info('Setting up eval callback: %s', config['callback_name'])
      eval_callback = callbacks.get_callback(config['callback_name'])(
          self._model,
          self._params,
          self._batch_stats,
          self._optimizer_state,
          self._dataset,
          self._hps,
          config,
          self._train_dir,
          rng,
          self._mesh,
          self.finalize_batch_fn,
      )
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
        self._early_stopping_min_steps,
        report,
    )
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

  def _eval(self, start_step, start_time, save=True):
    """Evaluate.

    Has the side-effects of:
      - synchronizing self._batch_stats across hosts
      - checkpointing via self._save(self._train_dir)
      - resetting self._sum_train_cost to jnp.zeros
      - resetting self._time_at_prev_eval_end to the current time
      - resetting self._prev_eval_step to self._global_step

    Args:
      start_step: the training start step.
      start_time: the training start time.
      save: flag to save a checkpoint to disk. defaults to True.

    Returns:
      A Dict[str, Any] eval report, originally created in
      trainer_utils.eval_metrics.
    """
    time_since_last_eval = time.time() - self._time_at_prev_eval_end

    if self._eval_use_ema:
      eval_params = self.training_algorithm.get_ema_eval_params(
          self._optimizer_state
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
        self._evaluate_batch_jitted,
        self.finalize_batch_fn,
    )
    self._run_eval_callbacks(report)
    if save:
      self._save(self._train_dir)
    steps_since_last_eval = self._global_step - self._prev_eval_step
    steps_per_sec_no_eval = steps_since_last_eval / time_since_last_eval
    run_time = time.time() - self._time_at_prev_eval_end
    steps_per_sec = steps_since_last_eval / run_time

    mean_train_cost = self._sum_train_cost / max(
        1, self._global_step - self._prev_eval_step
    )
    self._sum_train_cost = 0.0
    epoch = self._global_step * self._hps.batch_size // self._hps.train_size
    overall_steps_per_sec = self._get_step_frequency(
        self._global_step, start_step, start_time)
    report.update(
        global_step=self._global_step,
        epoch=epoch,
        preemption_count=self._preemption_count,
        train_cost=mean_train_cost,
        overall_steps_per_sec=overall_steps_per_sec,
        steps_per_sec_no_eval=steps_per_sec_no_eval,
        steps_per_sec=steps_per_sec,
        eval_time=eval_time,
        run_time_no_eval=time_since_last_eval,
        run_time=run_time,
    )
    report.update(self.training_algorithm.eval_report_metrics)
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

    self._time_at_prev_eval_end = time.time()
    self._prev_eval_step = self._global_step
    return report

  def setup_and_maybe_restore(self, init_rng, data_rng, callback_rng):
    """Sets up the training state and restores from checkpoint if one exists.

    Args:
      init_rng: (jax.random.PRNGKey) Rng seed used in model initialization.
      data_rng: (jax.random.PRNGKey) Rng seed used in data shuffling.
      callback_rng: (jax.random.PRNGKey) Rng seed used in callback functions.
    """
    self.training_algorithm = self._training_algorithm_class(
        self._hps, self._model, self._num_train_steps
    )

    unreplicated_params, unreplicated_batch_stats = self._model.initialize(
        self._initializer,
        self._hps,
        init_rng,
        self._init_logger,
    )

    self.log_model_info(unreplicated_params)

    unreplicated_optimizer_state = self.training_algorithm.init_optimizer_state(
        self._model,
        unreplicated_params,
        unreplicated_batch_stats,
        self._hps,
        init_rng,
    )

    unreplicated_metrics_state = None
    # TODO(kasimbeg): move this to initialization.
    self._metrics_update_fn = None
    self._metrics_summary_fn = None

    if self._training_metrics_config is not None:
      (metrics_init_fn, self._metrics_update_fn,
       self._metrics_summary_fn) = make_training_metrics(
           self._num_train_steps, self._hps, **self._training_metrics_config)
      unreplicated_metrics_state = metrics_init_fn(
          unreplicated_params, unreplicated_batch_stats
      )

    (
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
    ) = self.maybe_restore_from_checkpoint(
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
    )

    (
        self._params,
        self._params_sharding,
        self._optimizer_state,
        self._optimizer_state_sharding,
        self._batch_stats,
        self._batch_stats_sharding,
        self._metrics_state,
        self._metrics_state_sharding,
    ) = self.shard(
        unreplicated_params,
        unreplicated_optimizer_state,
        unreplicated_batch_stats,
        unreplicated_metrics_state,
    )

    self._dataset = self.setup_data_loader(data_rng, self._global_step)
    self._eval_callbacks = self._setup_eval_callbacks(callback_rng)

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

    self.setup_and_maybe_restore(init_rng, data_rng, callback_rng)

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

    for _ in range(start_step, self._num_train_steps):
      with jax.profiler.StepTraceAnnotation(
          'train', step_num=self._global_step
      ):
        # NOTE(dsuo): to properly profile each step, we must include batch
        # creation in the StepTraceContext (as opposed to putting `train_iter`
        # directly in the top-level for loop).
        batch = next(train_iter)
        batch = self.finalize_batch_fn(batch)

        # It looks like we are reusing an rng key, but we aren't.
        (
            self._optimizer_state,
            self._params,
            self._batch_stats,
            self._metrics_state,
            self._sum_train_cost,
        ) = self.update(
            batch,
            rng,
            self._metrics_state,
            self._sum_train_cost,
        )
        self._global_step += 1
        if self._global_step in self._checkpoint_steps:
          self._save(self._checkpoint_dir, max_to_keep=None)

        # TODO(gdahl, gilmer): consider moving this test up.
        # NB: Since this test is after we increment self._global_step, having 0
        # in eval_steps does nothing.
        if trainer_utils.should_eval(
            self._global_step, self._eval_frequency, self._eval_steps):
          try:
            report = self._eval(start_step, start_time)
          except utils.TrainingDivergedError as e:
            self.wait_until_orbax_checkpointer_finished()
            raise utils.TrainingDivergedError(
                f'divergence at step {self._global_step}'
            ) from e
          yield report
          if self._check_early_stopping(report):
            self.wait_until_orbax_checkpointer_finished()
            return

    # Always log and checkpoint on host 0 at the end of training.
    # If we moved where in the loop body evals happen then we would not need
    # this test.
    if self._prev_eval_step != self._num_train_steps:
      report = self._eval(start_step, start_time)
      yield report
    # To make sure the last checkpoint was correctly saved.
    self.wait_until_orbax_checkpointer_finished()

  @abc.abstractmethod
  def update(self, batch, rng, metrics_update_fn, metrics_state, training_cost):
    """Single step of the training loop.
    
    Args:
      batch: the per-device batch of data to process.
      rng: the RNG used for calling the model. `step` and `local_device_index`
        will be folded into this to produce a unique per-device, per-step RNG.
      metrics_update_fn: a function that takes in the current metrics state, the
        current step, and the training cost, gradients, params, and optimizer
        state and returns the new metrics state.
      metrics_state: the current metrics state.
      training_cost: the current training cost.

    Returns:
      A tuple of the new optimizer, the new batch stats, the scalar training
      cost,
      the new training metrics state, the gradient norm, and the update norm.
    """

  @abc.abstractmethod
  def shard(self, unreplicated_params, unreplicated_optimizer_state,
            unreplicated_batch_stats, unreplicated_metrics_state):
    """Shard the training state.

    Args:
      unreplicated_params: (FrozenDict) The model parameters.
      unreplicated_optimizer_state: (optax.OptState) The optimizer state.
      unreplicated_batch_stats: (FrozenDict) The batch statistics.
      unreplicated_metrics_state: (FrozenDict) The metrics state.

    Yields:
      The sharded training state.
    """

  @abc.abstractmethod
  def finalize_batch_fn(self, batch):
    """Finalizes the batch before passing to the model.

    Example use cases: Handle sharding or reshaping.

    Args:
      batch: (dict) A batch of data.

    Returns:
      The finalized batch of data.
    """

  def fetch_learning_rate(self, optimizer_state):
    return trainer_utils.fetch_learning_rate(optimizer_state)
