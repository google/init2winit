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

"""Utility functions related to training."""
import time

from absl import logging

from flax import jax_utils
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax


def format_time(s):
  """Format time in hours/minutes/seconds."""
  if s < 60:
    return f'{s:.0f}s'
  m, s = divmod(s, 60)
  if m < 60:
    return f'{m:.0f}m{s:.0f}s'
  h, m = divmod(m, 60)
  return f'{h:.0f}h{m:.0f}m'  # Seconds intentionally omitted


def log_message(msg, pool=None, work_unit=None):
  if work_unit is not None and pool is not None:
    pool.apply_async(lambda msg=msg: work_unit.set_notes(msg))
  logging.info('%s', msg)


def log_eta(pool, work_unit, global_step, steps_per_sec_no_eval,
            num_train_steps, start_time, eval_frequency, eval_steps, eval_time):
  """Construct and ETA / total time entry."""
  msg = f'Steps: {global_step} / {num_train_steps} '
  msg += f'[{global_step / num_train_steps:.1%}] '

  # Time remaining from training
  train_eta = (num_train_steps - global_step) / steps_per_sec_no_eval

  # Time remaining from eval
  if eval_steps:
    num_evals = len(list(filter(lambda x: x > global_step, eval_steps)))
  else:
    num_evals = (num_train_steps - global_step) // eval_frequency
  eval_eta = eval_time * num_evals

  msg += f'ETA: {format_time(train_eta + eval_eta)} '
  total_time = time.time() - start_time
  msg += f'Total time: {format_time(total_time)}'

  log_message(msg, pool, work_unit)


def log_epoch_report(report, metrics_logger):
  logging.info('Step %d, steps/second: %f, report: %r', report['global_step'],
               report['steps_per_sec_no_eval'], report)
  if metrics_logger:
    metrics_logger.append_scalar_metrics(report)
  logging.info('Finished (estimated) epoch %d. Saving checkpoint.',
               report['epoch'])


def maybe_log_training_metrics(metrics_state,
                               metrics_summary_fn,
                               metrics_logger):
  """If appropriate, send a summary tree of training metrics to the logger."""
  if metrics_state:
    unreplicated_metrics_state = jax_utils.unreplicate(metrics_state)
    summary_tree = metrics_summary_fn(unreplicated_metrics_state)
    metrics_logger.append_pytree(summary_tree)
    metrics_logger.write_pytree(unreplicated_metrics_state,
                                prefix='metrics_state')


def maybe_sync_batchnorm_stats(batch_stats):
  """Sync batch_stats across devices."""
  # We first check that batch_stats is used (pmap will throw an error if
  # it's a non batch norm model). If batch norm is not used then
  # batch_stats = None. Note that, in the case of using our implementation of
  # virtual batch norm, this will also handle synchronizing the multiple moving
  # averages on each device before doing a cross-host sync.
  if batch_stats:
    batch_stats = jax.pmap(
        model_utils.sync_batchnorm_stats, axis_name='batch')(
            batch_stats)
  return batch_stats


def should_eval(global_step, eval_frequency, eval_steps):
  on_step = eval_steps and global_step in eval_steps
  on_freq = (global_step % eval_frequency == 0)

  return on_step or on_freq


def check_for_early_stopping(
    early_stopping_target_name,
    early_stopping_target_value,
    early_stopping_mode,
    early_stopping_min_steps,
    eval_report,
):
  """Check if we reached the metric value to stop training early."""
  if early_stopping_target_name is not None:
    if early_stopping_target_name not in eval_report:
      raise ValueError(
          'Provided early_stopping_target_name '
          f'{early_stopping_target_name} not in the computed metrics: '
          f'{eval_report.keys()}.')
    if early_stopping_mode is None:
      raise ValueError(
          'Need to provide a early_stopping_mode if using early stopping.')
    # Note that because eval metrics are synced across hosts, this should
    # stop training on every host at the same step.
    if eval_report['global_step'] < early_stopping_min_steps:
      return False
    if early_stopping_mode == 'above':
      return (
          eval_report[early_stopping_target_name] >= early_stopping_target_value
      )
    else:
      return (
          eval_report[early_stopping_target_name] <= early_stopping_target_value
      )


def prefetch_input_pipeline(ds, n_prefetch=0, devices=None):
  """Modify input pipeline to prefetch from host to device.

  Args:
    ds: tf.data pipeline
    n_prefetch: number of items to prefetch
    devices: devices to prefetch to

  Returns:
    prefetching ds

  """
  it = iter(ds)
  it = (data_utils.shard(x) for x in it)
  if n_prefetch > 0:
    it = jax_utils.prefetch_to_device(it, n_prefetch, devices=devices)
  return it


def evaluate(
    params,
    batch_stats,
    batch_iter,
    evaluate_batch_pmapped):
  """Compute aggregated metrics on the given data iterator.

  WARNING: The caller is responsible for synchronizing the batch norm statistics
  before calling this function!

  Assumed API of evaluate_batch_pmapped:
  metrics = evaluate_batch_pmapped(params, batch_stats, batch)
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
       evaluate_batch_pmapped(params, batch_stats, batch). Returns a dictionary
       mapping keys to the metric values across the sharded batch.

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
        params=params, batch_stats=batch_stats, batch=batch).unreplicate()
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


def _filtering(path, _) -> bool:
  """Filter to ensure that we inject/fetch lrs from 'InjectHyperparamsState'-like states."""
  if (
      (len(path) > 1)
      and isinstance(path[-2], jax.tree_util.GetAttrKey)
      and path[-2].name == 'hyperparams'
  ):
    return True
  else:
    return False


def inject_learning_rate(optimizer_state, lr):
  """Inject the given LR into any optimizer state that will accept it.

  We require that the optimizer state exposes an 'InjectHyperparamsState'-like
  interface, i.e., it should contain a `hyperparams` dictionary with a
  'learning_rate' key where the learning rate can be set. We need to do this
  to allow arbitrary (non-jittable) LR schedules.

  Args:
    optimizer_state: optimizer state returned by an optax optimizer
    lr: learning rate to inject

  Returns:
    new_optimizer_state
      optimizer state with the same structure as the input. The learning_rate
      entry in the state has been set to lr.
  """
  return optax.tree_utils.tree_set(
      optimizer_state, _filtering, learning_rate=lr
  )


def fetch_learning_rate(optimizer_state):
  """Fetch the LR from any optimizer state."""
  lrs_with_path = optax.tree_utils.tree_get_all_with_path(
      optimizer_state, 'learning_rate', _filtering
  )
  if not lrs_with_path:
    raise ValueError(f'No learning rate found in {optimizer_state}.')
  all_equal = all(
      jnp.array_equal(lr, lrs_with_path[0][1]) for _, lr in lrs_with_path
  )
  if all_equal:
    lr_array = lrs_with_path[0][1]
    return lr_array[0]
  else:
    raise ValueError(
        'All learning rates in the optimizer state must be the same.'
        f'Found {lrs_with_path} in {optimizer_state}.'
    )


def _merge_and_apply_prefix(d1, d2, prefix):
  d1 = d1.copy()
  for key in d2:
    d1[prefix+key] = d2[key]
  return d1


@utils.timed
def eval_metrics(params, batch_stats, dataset, eval_num_batches,
                 test_num_batches, eval_train_num_batches,
                 evaluate_batch_pmapped):
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
                             evaluate_batch_pmapped)
    # Metrics are None if the dataset doesn't have that split
    if split_metrics is not None:
      metrics = _merge_and_apply_prefix(metrics, split_metrics,
                                        (split_name + '/'))
  return metrics
