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
      external_checkpoint_path=None):
    """Main training loop.

    Trains the given network on the specified dataset for the given number of
    epochs. Saves the training curve in train_dir/r=3/results.tsv.

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
    """
    self._train_dir = train_dir
    self._model = model
    self._dataset_builder = dataset_builder
    self._initializer = initializer
    self._num_train_steps = num_train_steps
    self._hps = hps
    self._rng = rng
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
    self._callback_configs = callback_configs
    self._external_checkpoint_path = external_checkpoint_path

  @abc.abstractmethod
  def train(self):
    """All training logic.

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
