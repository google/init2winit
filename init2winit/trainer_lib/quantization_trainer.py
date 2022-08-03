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

"""Standard trainer for the init2winit project."""
import functools
import itertools
import multiprocessing
import os
import time

from absl import logging
from init2winit import callbacks
from init2winit import checkpoint
from init2winit import schedules
from init2winit import utils
from init2winit.init_lib import init_utils
from init2winit.model_lib import model_utils
from init2winit.model_lib.binarize_layers import DynamicContext
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import trainer_utils
from init2winit.training_metrics_grabber import make_training_metrics
import jax
from jax import lax
import jax.numpy as jnp
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


def train(
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
    teacher_model=None):
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
    eval_train_num_batches: (int) The number of batches for evaluating on train.
      Set to None to evaluate on the whole training set.
    eval_frequency: (int) Evaluate every k steps.
    checkpoint_steps: List of integers indicating special steps to save
      checkpoints at. These checkpoints do not get used for preemption recovery.
    early_stopping_target_name: A string naming the metric to use to perform
       early stopping. If this metric reaches the value
      `early_stopping_target_value`, training will stop. Must include the
      dataset split (ex: validation/error_rate).
    early_stopping_target_value: A float indicating the value at which to stop
      training.
    early_stopping_mode: One of "above" or "below", indicates if we should stop
      when the metric is above or below the threshold value. Example: if
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
      training_metrics_grabber. Set to None to skip logging of advanced training
      metrics.
    callback_configs: List of configs specifying general callbacks to run
      during the eval phase. Empty list means no callbacks are run. See
      callbacks.py for details on what is expected in a config.
    external_checkpoint_path: (str) If this argument is set, we will load the
      optimizer_state, params, batch_stats, and training_metrics from the
      checkpoint at this location.
    teacher_model: an instantiated model_cls for knowledge distillation.

  Yields:
    metrics: A dictionary of all eval metrics from the given epoch.
  """
  # NOTE: the initialization RNG should *not* be per-host, as this will create
  # different sets of weights per host. However, all other RNGs should be
  # per-host.
  # TODO(znado,gilmer,gdahl): implement replicating the same initialization
  # across hosts.
  rng, init_rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, data_rng = jax.random.split(rng)

  # only used if checkpoints_steps is non-empty.
  checkpoint_dir = os.path.join(train_dir, 'checkpoints')

  # For logging / processing off the main thread
  pool = multiprocessing.pool.ThreadPool()

  if jax.process_index() == 0:
    logging.info('Let the training begin!')
    logging.info('Dataset input shape: %r', hps.input_shape)
    logging.info('Hyperparameters: %s', hps)

  if eval_batch_size is None:
    eval_batch_size = hps.batch_size
  if callback_configs is None:
    callback_configs = []

  # Maybe run the initializer.
  unreplicated_params, unreplicated_batch_stats = init_utils.initialize(
      model.flax_module,
      initializer,
      model.loss_fn,
      hps.input_shape,
      hps.output_shape,
      hps,
      init_rng,
      init_logger,
      model.get_fake_batch(hps))

  if jax.process_index() == 0:
    utils.log_pytree_shape_and_statistics(unreplicated_params)
    logging.info('train_size: %d,', hps.train_size)

  # Note that global_step refers to the number of gradients calculations, not
  # the number of model updates. This means when using gradient accumulation,
  # one must supply configs where the number of steps are in units of gradient
  # calculations, not model updates, and in post processing one must divide
  # global_step by grad_accum_step_multiplier to get the number of updates.
  #
  # The learning rate schedules are all defined in number of model updates. In
  # order to make it clearer in the logging that the LR does not change across
  # gradient calculations (only when updates are applied), we stretch the
  # learning rate schedule by the number of gradient calculations per weight
  # update.
  stretch_factor = 1
  if hps.get('total_accumulated_batch_size') is not None:
    stretch_factor = hps.total_accumulated_batch_size // hps.batch_size

  optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(hps, model)
  unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)

  (unreplicated_metrics_state, metrics_update_fn, metrics_summary_fn
   ) = None, None, None
  if training_metrics_config is not None:
    (metrics_init_fn, metrics_update_fn, metrics_summary_fn
     ) = make_training_metrics(num_train_steps, **training_metrics_config)
    unreplicated_metrics_state = metrics_init_fn(unreplicated_params)

  (optimizer_state, params, batch_stats, metrics_state,
   global_step, sum_train_cost, preemption_count, is_restored
   ) = checkpoint.replicate_and_maybe_restore_checkpoint(
       unreplicated_optimizer_state,
       unreplicated_params,
       unreplicated_batch_stats,
       unreplicated_metrics_state,
       train_dir=train_dir,
       external_checkpoint_path=external_checkpoint_path)

  # initialize the teacher model
  if teacher_model is None:
    logging.info('Found teacher_model None. Proceed with regular training.')
    teacher_params, teacher_batch_stats = None, None
  else:
    logging.info('Initialize the teacher_model in trainer.')
    teacher_unreplicated_params, teacher_unreplicated_batch_stats = init_utils.initialize(
        teacher_model.flax_module,
        initializer,
        teacher_model.loss_fn,
        hps.input_shape,
        hps.output_shape,
        hps,
        init_rng,
        init_logger,
        teacher_model.get_fake_batch(hps))
    logging.info('Initialize teacher_model optimizer state.')
    teacher_unreplicated_optimizer_state = optimizer_init_fn(
        teacher_unreplicated_params)
    logging.info('Restore teacher_model ckpt.')
    (_, teacher_params, teacher_batch_stats, _, _, _, _, _
    ) = checkpoint.replicate_and_maybe_restore_checkpoint(
        teacher_unreplicated_optimizer_state,
        teacher_unreplicated_params,
        teacher_unreplicated_batch_stats,
        None,  # unreplicated_metrics_state
        train_dir='none',  # a fake directory that does not exist
        external_checkpoint_path=hps.teacher_checkpoint_path
    )

  if is_restored:
    preemption_count += 1
    # Fold the restored step into the dataset RNG so that we will get a
    # different shuffle each time we restore, so that we do not repeat a
    # previous dataset ordering again after restoring. This is not the only
    # difference in shuffling each pre-emption, because we often times reshuffle
    # the input files each time in a non-deterministic manner.
    #
    # Note that if we are pre-empted more than once per epoch then we will
    # retrain more on the beginning of the training split, because each time we
    # restore we refill the shuffle buffer with the first `shuffle_buffer_size`
    # elements from the training split to continue training.
    #
    # Also note that for evaluating on the training split, because we are
    # reshuffling each time, we will get a new eval_train split each time we are
    # pre-empted.
    data_rng = jax.random.fold_in(data_rng, global_step)

  assert hps.batch_size % (jax.device_count()) == 0
  assert eval_batch_size % (jax.device_count()) == 0
  dataset = dataset_builder(
      data_rng,
      hps.batch_size,
      eval_batch_size=eval_batch_size,
      hps=hps,
  )

  update_fn = functools.partial(
      update,
      teacher_model=teacher_model,
      training_cost=model.training_cost,
      grad_clip=hps.get('grad_clip'),
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
  #     dynamic_context=0,
  #     teacher_params=0,
  #     teacher_batch_stats=0,
  #     teacher_model,
  #     training_cost,
  #     grad_clip,
  #     optimizer_update_fn,
  #     metrics_state_update_fn)
  # Also, we can donate buffers for 'optimizer', 'batch_stats',
  # 'batch' and 'training_metrics_state' for update's pmapped computation.
  update_pmapped = utils.timed(jax.pmap(
      update_fn,
      axis_name='batch',
      in_axes=(0, 0, 0, 0, 0, None, None, None, 0, 0, 0, 0, 0),
      donate_argnums=(0, 1, 2, 8)))
  # During eval, we can donate the 'batch' buffer. We don't donate the
  # 'params' and 'batch_stats' buffers as we don't re-assign those values in
  # eval, we do that only in train.
  evaluate_batch_pmapped = jax.pmap(
      model.evaluate_batch, axis_name='batch', donate_argnums=(2,))

  if jax.process_index() == 0:
    trainer_utils.log_message('Starting training!', pool, xm_work_unit)

  # Numpy array of range(0, local_device_count) to send to each device to be
  # folded into the RNG inside each train step to get a unique per-device RNG.
  local_device_indices = np.arange(jax.local_device_count())

  # Start at the resumed step and continue until we have finished the number of
  # training steps. If building a dataset iterator using a tf.data.Dataset, in
  # the case of a batch size that does not evenly divide the training dataset
  # size, if using `ds.batch(..., drop_remainer=True)` on the training dataset
  # then the final batch in this iterator will be a partial batch. However, if
  # `drop_remainer=False`, then this iterator will always return batches of the
  # same size, and the final batch will have elements from the start of the
  # (num_epochs + 1)-th epoch.
  train_iter = itertools.islice(
      dataset.train_iterator_fn(), global_step, num_train_steps)

  eval_callbacks = []
  rng, callback_rng = jax.random.split(rng)
  callback_rngs = jax.random.split(callback_rng, len(callback_configs))
  for callback_rng, config in zip(callback_rngs, callback_configs):
    eval_callback = callbacks.get_callback(config['callback_name'])(
        model, params, batch_stats, optimizer_state, optimizer_update_fn,
        dataset, hps, config, train_dir, callback_rng)
    eval_callbacks.append(eval_callback)


  train_iter = trainer_utils.prefetch_input_pipeline(train_iter,
                                                     hps.num_device_prefetches)

  start_time = time.time()
  start_step = global_step
  def get_step_frequency(cur_step):
    return float(cur_step - start_step) / (time.time() - start_time)

  # NOTE(dsuo): record timestamps for run_time since we don't have a duration
  # that we can increment as in the case of train_time.
  time_at_prev_eval = start_time
  train_time_since_prev_eval = 0
  prev_eval_step = global_step

  for _ in range(start_step, num_train_steps):
    with jax.profiler.StepTraceAnnotation('train', step_num=global_step):
      # NOTE(dsuo): to properly profile each step, we must include batch
      # creation in the StepTraceContext (as opposed to putting `train_iter`
      # directly in the top-level for loop).
      batch = next(train_iter)

      if global_step in checkpoint_steps and jax.process_index() == 0:
        checkpoint.save_unreplicated_checkpoint_background(
            checkpoint_dir,
            optimizer_state,
            params,
            batch_stats,
            metrics_state,
            global_step,
            preemption_count,
            sum_train_cost,
            max_to_keep=None)
      # calculate num_train_steps depending on the lr_restart_steps
      init_step = max([x for x in hps.lr_restart_steps if x <= global_step])
      end_step = min([x for x in hps.lr_restart_steps if x > global_step])
      lr_fn = schedules.get_schedule_fn(
          hps.lr_hparams, (end_step - init_step) // stretch_factor,
          stretch_factor=stretch_factor)
      lr = lr_fn(global_step-init_step)
      dynamic_context = DynamicContext(  # re-create quant flags at each step
          quant_ff_weights=(global_step >= hps.quant_steps.ff_weights),
          quant_att_weights=(global_step >= hps.quant_steps.att_weights),
          quant_ff_acts=(global_step >= hps.quant_steps.ff_acts),
          quant_att_out_acts=(global_step >= hps.quant_steps.att_out_acts),
          quant_att_kqv_acts=(global_step >= hps.quant_steps.att_kqv_acts))
      (optimizer_state, params, batch_stats, sum_train_cost,
       metrics_state, grad_norm), train_time = update_pmapped(
           optimizer_state, params, batch_stats, metrics_state, batch,
           global_step, lr, rng, local_device_indices, sum_train_cost,
           dynamic_context, teacher_params, teacher_batch_stats)
      train_time_since_prev_eval += train_time
      global_step += 1
      # TODO(gdahl, gilmer): consider moving this test up.
      # NB: Since this test is after we increment global_step, having 0 in
      # eval_steps does nothing.
      if trainer_utils.should_eval(global_step, eval_frequency, eval_steps):
        batch_stats = trainer_utils.maybe_sync_batchnorm_stats(batch_stats)
        report, eval_time = trainer_utils.eval_metrics(
            params,
            batch_stats,
            dataset,
            eval_num_batches,
            test_num_batches,
            eval_train_num_batches,
            evaluate_batch_pmapped)

        for eval_callback in eval_callbacks:
          callback_metrics = eval_callback.run_eval(params, batch_stats,
                                                    optimizer_state,
                                                    global_step)
          if set(callback_metrics.keys()).intersection(set(report.keys())):
            raise ValueError('There was a collision between the callback'
                             'metrics and the standard eval metrics keys')
          report.update(callback_metrics)
        if jax.process_index() == 0:
          checkpoint.save_unreplicated_checkpoint_background(
              train_dir,
              optimizer_state,
              params,
              batch_stats,
              metrics_state,
              global_step,
              preemption_count,
              sum_train_cost)
        steps_since_last_eval = global_step - prev_eval_step
        steps_per_sec_train_only = steps_since_last_eval / train_time_since_prev_eval
        time_since_last_eval = time.time() - time_at_prev_eval
        steps_per_sec = steps_since_last_eval / time_since_last_eval

        mean_train_cost = sum_train_cost.mean().item() / max(
            1, global_step - prev_eval_step)
        report.update(learning_rate=float(lr),
                      global_step=global_step,
                      epoch=global_step * hps.batch_size // hps.train_size,
                      grad_norm=np.mean(grad_norm),
                      preemption_count=preemption_count,
                      train_cost=mean_train_cost,
                      overall_steps_per_sec=get_step_frequency(global_step),
                      steps_per_sec_train_only=steps_per_sec_train_only,
                      steps_per_sec=steps_per_sec,
                      eval_time=eval_time,
                      train_time=train_time_since_prev_eval,
                      run_time=time_since_last_eval
                      )
        yield report
        if jax.process_index() == 0:
          trainer_utils.log_eta(pool, xm_work_unit, global_step,
                                steps_per_sec_train_only, num_train_steps,
                                start_time, eval_frequency, eval_steps,
                                eval_time)
          trainer_utils.log_epoch_report(report, metrics_logger)
          trainer_utils.maybe_log_training_metrics(metrics_state,
                                                   metrics_summary_fn,
                                                   metrics_logger)

        sum_train_cost = jnp.zeros(jax.local_device_count())
        time_at_prev_eval = time.time()
        train_time_since_prev_eval = 0
        prev_eval_step = global_step

        early_stopping_condition = trainer_utils.check_for_early_stopping(
            early_stopping_target_name,
            early_stopping_target_value,
            early_stopping_mode,
            report)
        if early_stopping_condition:
          comparison_string = '>=' if early_stopping_mode == 'above' else '<='
          logging.info(
              'Early stopping because metric %s=%f, reached the target value '
              'of %s %f.',
              early_stopping_target_name,
              report[early_stopping_target_name],
              comparison_string,
              early_stopping_target_value)
          return

  # Always log and checkpoint on host 0 at the end of training.
  # If we moved where in the loop body evals happen then we would not need this
  # test.
  if prev_eval_step != num_train_steps:
    batch_stats = trainer_utils.maybe_sync_batchnorm_stats(batch_stats)
    report, eval_time = trainer_utils.eval_metrics(
        params,
        batch_stats,
        dataset,
        eval_num_batches,
        test_num_batches,
        eval_train_num_batches,
        evaluate_batch_pmapped)
    lr = lr_fn(global_step)
    # Correct the average for the final partial epoch.
    mean_train_cost = sum_train_cost.mean().item() / max(
        1, global_step - prev_eval_step)
    if jax.process_index() == 0:
      checkpoint.save_unreplicated_checkpoint_background(
          train_dir,
          optimizer_state,
          params,
          batch_stats,
          metrics_state,
          global_step,
          preemption_count,
          sum_train_cost)
    steps_since_last_eval = global_step - prev_eval_step
    steps_per_sec_train_only = steps_since_last_eval / train_time_since_prev_eval
    time_since_last_eval = time.time() - time_at_prev_eval
    steps_per_sec = steps_since_last_eval / time_since_last_eval

    report.update(learning_rate=float(lr),
                  global_step=global_step,
                  epoch=global_step * hps.batch_size // hps.train_size,
                  grad_norm=np.mean(grad_norm),
                  preemption_count=preemption_count,
                  train_cost=mean_train_cost,
                  overall_steps_per_sec=get_step_frequency(global_step),
                  steps_per_sec_train_only=steps_per_sec_train_only,
                  steps_per_sec=steps_per_sec,
                  eval_time=eval_time,
                  train_time=train_time_since_prev_eval,
                  run_time=time_since_last_eval)
    yield report
    if jax.process_index() == 0:
      trainer_utils.log_eta(pool, xm_work_unit, global_step,
                            steps_per_sec_train_only, num_train_steps,
                            start_time, eval_frequency, eval_steps, eval_time)
      trainer_utils.log_epoch_report(report, metrics_logger)
      trainer_utils.maybe_log_training_metrics(metrics_state,
                                               metrics_summary_fn,
                                               metrics_logger)
  # To make sure the last checkpoint was correctly saved.
  checkpoint.wait_for_checkpoint_save()
