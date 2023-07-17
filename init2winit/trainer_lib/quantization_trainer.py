# coding=utf-8
# Copyright 2023 The init2winit Authors.
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
from init2winit import checkpoint
from init2winit import schedules
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
from init2winit.model_lib.binarize_layers import DynamicContext
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer_utils
import jax
from jax import lax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
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
    local_device_index,
    running_train_cost,
    dynamic_context,
    teacher_params,
    teacher_batch_stats,
    teacher_model,
    training_cost,
    grad_clip,
    optimizer_update_fn,
    metrics_update_fn):
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
    dynamic_context: a dataclass with all quantization flags.
    teacher_params: pre-trained teacher model parameters.
    teacher_batch_stats: pre-trained teacher model batch stats.
    teacher_model: used for computing teacher logits in knowledge distillation.
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
        dynamic_context=dynamic_context,
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


def _merge_and_apply_prefix(d1, d2, prefix):
  d1 = d1.copy()
  for key in d2:
    d1[prefix+key] = d2[key]
  return d1


@utils.timed
def eval_metrics(params, batch_stats, dataset, eval_num_batches,
                 test_num_batches, eval_train_num_batches,
                 evaluate_batch_pmapped, dynamic_context):
  """Evaluates the given network on the train, validation, and test sets.

  WARNING: we assume that `batch_stats` has already been synchronized across
  devices before being passed to this function! See
  `trainer_utils.maybe_sync_batchnorm_stats`.

  The metric names will be of the form split/measurement for split in the set
  {train, valid, test} and measurement in the set {loss, error_rate}.

  Args:
    params: a dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: a dict of non-trainable model state. Passed as
      {'batch_stats': batch_stats} into flax_module.apply().
    dataset: Dataset returned from datasets.get_dataset. train, validation, and
      test sets.
    eval_num_batches: (int) The batch size used for evaluating on validation
      sets. Set to None to evaluate on the whole validation set.
    test_num_batches: (int) The batch size used for evaluating on test
      sets. Set to None to evaluate on the whole test set.
    eval_train_num_batches: (int) The batch size used for evaluating on train
      set. Set to None to evaluate on the whole training set.
    evaluate_batch_pmapped: Computes the metrics on a sharded batch.
    dynamic_context: a dataclass with quantization flags. Passed to
      evaluate_batch_pmapped to replace the dynamic_context in flax_module.

  Returns:
    A dictionary of all computed metrics.
  """
  train_iter = dataset.eval_train_epoch(eval_train_num_batches)
  valid_iter = dataset.valid_epoch(eval_num_batches)
  test_iter = dataset.test_epoch(test_num_batches)

  metrics = {}
  for split_iter, split_name in zip([train_iter, valid_iter, test_iter],
                                    ['train', 'valid', 'test']):
    split_metrics = evaluate(params, batch_stats, split_iter,
                             evaluate_batch_pmapped, dynamic_context)
    # Metrics are None if the dataset doesn't have that split
    if split_metrics is not None:
      metrics = _merge_and_apply_prefix(metrics, split_metrics,
                                        (split_name + '/'))
  return metrics


def evaluate(
    params,
    batch_stats,
    batch_iter,
    evaluate_batch_pmapped,
    dynamic_context):
  """Compute aggregated metrics on the given data iterator.

  WARNING: The caller is responsible for synchronizing the batch norm statistics
  before calling this function!

  Assumed API of evaluate_batch_pmapped:
  metrics = evaluate_batch_pmapped(params, batch_stats, batch, dynamic_context)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. The metrics are
  merged using the CLU metrics logic for that metric type. See
  classification_metrics.py for a definition of evaluate_batch.

  Args:
    params: a dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: a dict of non-trainable model state. Passed as
      {'batch_stats': batch_stats} into flax_module.apply().
    batch_iter: Generator which yields batches. Must support the API
      for b in batch_iter:
    evaluate_batch_pmapped: A function with API
       evaluate_batch_pmapped(params, batch_stats, batch, dynamic_context).
       Returns a dictionary mapping keys to the metric values
       across the sharded batch.
    dynamic_context: a dataclass with quantization flags. Passed to
      evaluate_batch_pmapped to replace the dynamic_context in flax_module.

  Returns:
    A dictionary of aggregated metrics. The keys will match the keys returned by
    evaluate_batch_pmapped.
  """
  metrics = None
  for batch in batch_iter:
    batch = data_utils.shard(batch)
    # Returns a clu.metrics.Collection object. We assume that
    # `evaluate_batch_pmpapped` calls CLU's `gather_from_model_outputs`,
    # which includes an `all_gather` to replicate the values on all devices.
    # We need to `unreplicate` before merging the results across batches to
    # accommodate CollectingMetric, which concatenates the values across the
    # leading dimension, so we need to remove the leading shard dimension first.
    computed_metrics = evaluate_batch_pmapped(
        params=params,
        batch_stats=batch_stats,
        batch=batch,
        dynamic_context=dynamic_context).unreplicate()
    if metrics is None:
      metrics = computed_metrics
    else:
      # `merge` aggregates the metrics across batches.
      metrics = metrics.merge(computed_metrics)

  # For data splits with no data (e.g. Imagenet no test set) no values
  # will appear for that split.
  if metrics is not None:
    # `compute` aggregates the metrics across batches into a single value.
    metrics = metrics.compute()
    for key, val in metrics.items():
      if np.isnan(val):
        raise utils.TrainingDivergedError('NaN detected in {}'.format(key))
  return metrics


class Trainer(base_trainer.BaseTrainer):
  """Default trainer."""

  def __init__(
      self,
      *args,
      dataset_meta_data=None,
      loss_name=None,
      metrics_name=None,
      **kwargs):
    super(Trainer, self).__init__(*args, **kwargs)
    # need extra arguments to build the teacher
    self.dataset_meta_data = dataset_meta_data
    self.loss_name = loss_name
    self.metrics_name = metrics_name
    self._additional_eval_steps = range(
        self._num_train_steps,
        self._num_train_steps + self._hps.num_additional_train_steps,
        self._hps.additional_eval_frequency)
    self._num_train_steps += self._hps.num_additional_train_steps

  def train(self):
    """All training logic.

    The only side-effects are:
      - Initiailizing self._time_at_prev_eval to the current time
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

    # for knowledge distillation: create, initialize, and load teacher model
    if self._hps.teacher_model_name is None:
      teacher_model_cls = None
      if self._hps.teacher_checkpoint_path is not None:
        raise ValueError(
            'Got teacher_model_name None but teacher_checkpoint_path %s.' %
            self._hps.teacher_checkpoint_path)
      teacher_model = None
      teacher_params, teacher_batch_stats = None, None
    else:
      teacher_model_cls = models.get_model(self._hps.teacher_model_name)
      # Teacher model and student model share the same config file.
      # Note that the teacher does not need quantization related hps.
      teacher_model = teacher_model_cls(
          self._hps, self.dataset_meta_data, self.loss_name, self.metrics_name)
      logging.info('Initialize the teacher model.')
      (teacher_unreplicated_params,
       teacher_unreplicated_batch_stats) = teacher_model.initialize(
           self._initializer,
           self._hps,
           init_rng,
           self._init_logger,
       )
      logging.info('Initialize teacher_model optimizer state.')
      optimizer_init_fn, _ = optimizers.get_optimizer(
          self._hps, self._model, batch_axis_name='batch')
      teacher_unreplicated_optimizer_state = optimizer_init_fn(
          teacher_unreplicated_params)
      logging.info('Restore teacher model ckpt.')
      (_, teacher_params, teacher_batch_stats, _, _, _, _, _
      ) = checkpoint.replicate_and_maybe_restore_checkpoint(
          teacher_unreplicated_optimizer_state,
          teacher_unreplicated_params,
          teacher_unreplicated_batch_stats,
          None,  # unreplicated_metrics_state
          train_dir='none',  # a fake directory that does not exist
          external_checkpoint_path=self._hps.teacher_checkpoint_path,
          checkpointer=self._orbax_checkpointer)

    # overwrite self._update_pmap with teacher_model as an argument
    update_fn = functools.partial(
        update,
        teacher_model=teacher_model,
        training_cost=self._model.training_cost,
        grad_clip=self._hps.get('grad_clip'),
        optimizer_update_fn=self._optimizer_update_fn,
        metrics_update_fn=self._metrics_update_fn)
    self._update_pmapped = utils.timed(jax.pmap(
        update_fn,
        axis_name='batch',
        in_axes=(0, 0, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0),
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
        train_iter, self._hps.num_device_prefetches)

    start_time = time.time()
    start_step = self._global_step

    # NOTE(dsuo): record timestamps for run_time since we don't have a duration
    # that we can increment as in the case of train_time.
    self._time_at_prev_eval = start_time
    train_time_since_prev_eval = 0
    self._prev_eval_step = self._global_step

    configured_train_steps = (
        self._num_train_steps - self._hps.num_additional_train_steps)

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

        # if global step exceeds actual number of configured train steps,
        # init and end step is initialized with 'last' stage of training.
        if self._global_step >= configured_train_steps:
          init_step = self._hps.lr_restart_steps[-2]
          end_step = self._hps.lr_restart_steps[-1]
        else:
          # overwrite self._lr_fn with multi-step lr fn
          # calculate num_train_steps depending on the lr_restart_steps
          init_step = max(
              [x for x in self._hps.lr_restart_steps if x <= self._global_step])
          end_step = min(
              [x for x in self._hps.lr_restart_steps if x > self._global_step])
        if self._hps.restart_base_lr is None:
          # use vanilla lr_hparams if restart_base_lr is not specified
          lr_hparams = self._hps.lr_hparams
        else:
          # if restart_base_lr is specified, update base_lr at every train step
          which_lr_cycle = self._hps.lr_restart_steps.index(init_step)
          lr_hparams = self._hps.lr_hparams.to_dict()
          lr_hparams['base_lr'] = self._hps.restart_base_lr[which_lr_cycle]
          lr_hparams = config_dict.ConfigDict(lr_hparams)
        stretch_factor = 1
        if self._hps.get('total_accumulated_batch_size') is not None:
          stretch_factor = (
              self._hps.total_accumulated_batch_size // self._hps.batch_size)
        self._lr_fn = schedules.get_schedule_fn(
            lr_hparams, (end_step - init_step) // stretch_factor,
            stretch_factor=stretch_factor)
        # global step exceeds configured_train_steps
        if self._global_step >= configured_train_steps:
          # use last learning rate value from the schedule.
          lr = self._lr_fn(configured_train_steps - 1 - init_step)
        else:
          lr = self._lr_fn(self._global_step - init_step)
        # It looks like we are reusing an rng key, but we aren't.
        # TODO(gdahl): Make it more obvious that passing rng is safe.
        # TODO(gdahl,gilmer,znado): investigate possibly merging the member
        # variable inputs/outputs of this function into a named tuple.
        dynamic_context = DynamicContext(  # re-create quant flags at each step
            quant_ff_weights=(self._global_step >=
                              self._hps.quant_steps.ff_weights),
            quant_att_weights=(self._global_step >=
                               self._hps.quant_steps.att_weights),
            quant_ff_acts=(self._global_step >= self._hps.quant_steps.ff_acts),
            quant_att_out_acts=(self._global_step >=
                                self._hps.quant_steps.att_out_acts),
            quant_att_kqv_acts=(self._global_step >=
                                self._hps.quant_steps.att_kqv_acts))
        (self._optimizer_state, self._params, self._batch_stats,
         self._sum_train_cost, self._metrics_state,
         self._grad_norm), train_time = self._update_pmapped(
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
             dynamic_context,
             teacher_params,
             teacher_batch_stats)
        train_time_since_prev_eval += train_time
        self._global_step += 1
        # TODO(gdahl, gilmer): consider moving this test up.
        # NB: Since this test is after we increment self._global_step, having 0
        # in eval_steps does nothing.
        if trainer_utils.should_eval(
            self._global_step, self._eval_frequency,
            self._eval_steps) or (self._global_step
                                  in self._additional_eval_steps):
          report = self._eval(lr, start_step, start_time,
                              train_time_since_prev_eval, dynamic_context)
          train_time_since_prev_eval = 0
          yield report
          if self._check_early_stopping(report):
            return

    # Always log and checkpoint on host 0 at the end of training.
    # If we moved where in the loop body evals happen then we would not need
    # this test.
    if self._prev_eval_step != self._num_train_steps:
      report = self._eval(lr, start_step, start_time,
                          train_time_since_prev_eval, dynamic_context)
      yield report
    # To make sure the last checkpoint was correctly saved.
    self.wait_until_orbax_checkpointer_finished()

  def _eval(
      self,
      lr,
      start_step,
      start_time,
      train_time_since_prev_eval,
      dynamic_context):
    """Evaluate.

    Has the side-effects of:
      - synchronizing self._batch_stats across hosts
      - checkpointing via self._save(self._train_dir)
      - resetting self._sum_train_cost to jnp.zeros
      - resetting self._time_at_prev_eval to the current time
      - resetting self._prev_eval_step to self._global_step

    Args:
      lr: the current learning rate.
      start_step: the training start step.
      start_time: the training start time.
      train_time_since_prev_eval: the time since the last eval (as measured by
        utils.timed).
      dynamic_context: model quantization flags.

    Returns:
      A Dict[str, Any] eval report, originally created in
      trainer_utils.eval_metrics.
    """
    self._batch_stats = trainer_utils.maybe_sync_batchnorm_stats(
        self._batch_stats)
    report, eval_time = eval_metrics(
        self._params,
        self._batch_stats,
        self._dataset,
        self._eval_num_batches,
        self._test_num_batches,
        self._eval_train_num_batches,
        self._evaluate_batch_pmapped,
        dynamic_context)
    self._run_eval_callbacks(report)
    if jax.process_index() == 0:
      self._save(self._train_dir)
    steps_since_last_eval = self._global_step - self._prev_eval_step
    steps_per_sec_train_only = (
        steps_since_last_eval / train_time_since_prev_eval)
    time_since_last_eval = time.time() - self._time_at_prev_eval
    steps_per_sec = steps_since_last_eval / time_since_last_eval

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
        steps_per_sec_train_only=steps_per_sec_train_only,
        steps_per_sec=steps_per_sec,
        eval_time=eval_time,
        train_time=train_time_since_prev_eval,
        run_time=time_since_last_eval)
    if jax.process_index() == 0:
      trainer_utils.log_eta(
          self._logging_pool,
          self._xm_work_unit,
          self._global_step,
          steps_per_sec_train_only,
          self._num_train_steps,
          start_time,
          self._eval_frequency,
          self._eval_steps,
          eval_time)
      trainer_utils.log_epoch_report(report, self._metrics_logger)
      trainer_utils.maybe_log_training_metrics(self._metrics_state,
                                               self._metrics_summary_fn,
                                               self._metrics_logger)

    self._time_at_prev_eval = time.time()
    self._prev_eval_step = self._global_step
    return report
