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

"""Utility functions related to training."""

import functools
import time

from absl import logging
from flax import jax_utils
from init2winit import utils
from init2winit.dataset_lib import data_utils
import jax
from jax.experimental.multihost_utils import process_allgather  # pylint: disable=g-importing-member
import jraph
import numpy as np


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


def make_finalize_batch_fn(mesh):
  """Returns a function that makes each element of a batch a global array."""

  def finalize_batch_fn(batch):
    make_global_array_fn = functools.partial(
        data_utils.make_global_array, mesh=mesh
    )
    return jax.tree.map(make_global_array_fn, batch)

  return finalize_batch_fn


def evaluate(
    params,
    batch_stats,
    batch_iter,
    evaluate_batch_jitted,
    finalize_batch_fn,
):
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
    batch_stats: a dict of non-trainable model state. Passed as {'batch_stats':
      batch_stats} into flax_module.apply().
    batch_iter: Generator which yields batches. Must support the API for b in
      batch_iter.
    evaluate_batch_jitted: A function with API evaluate_batch_jitted(params,
      batch_stats, batch). Returns a dictionary mapping keys to the metric
      values across the sharded batch.
    finalize_batch_fn: Function to finalize the batch before passing to
      evaluate_batch_jitted. For sharding or reshaping if necessary. Can be a
      no-op otherwise.

  Returns:
    A dictionary of aggregated metrics. The keys will match the keys returned by
    evaluate_batch_jitted.
  """
  metrics = None

  for batch in batch_iter:
    batch = finalize_batch_fn(batch)
    # Returns a clu.metrics.Collection object. We assume that
    # `evaluate_batch_jitted` calls CLU's `single_from_model_outputs`.
    computed_metrics = evaluate_batch_jitted(
        params=params, batch_stats=batch_stats, batch=batch
    )
    if metrics is None:
      metrics = computed_metrics
    else:
      # `merge` aggregates the metrics across batches.
      metrics = metrics.merge(computed_metrics)

  metrics = jax.device_get(process_allgather(metrics, tiled=True))
  metrics = jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 1 else x, metrics)
  # For data splits with no data (e.g. Imagenet no test set) no values
  # will appear for that split.
  if metrics is not None:
    # `compute` aggregates the metrics across batches into a single value.
    metrics = metrics.compute()
    for key, val in metrics.items():
      if np.isnan(val):
        raise utils.TrainingDivergedError('NaN detected in {}'.format(key))
  return metrics


def _merge_and_apply_prefix(d1, d2, prefix):
  d1 = d1.copy()
  for key in d2:
    d1[prefix+key] = d2[key]
  return d1


@utils.timed
def eval_metrics(
    params,
    batch_stats,
    dataset,
    eval_num_batches,
    test_num_batches,
    eval_train_num_batches,
    evaluate_batch_jitted,
    finalize_batch_fn=None,
):
  """Evaluates the given network on the train, validation, and test sets.

  WARNING: we assume that `batch_stats` has already been synchronized across
  devices before being passed to this function! See
  `trainer_utils.maybe_sync_batchnorm_stats`.

  The metric names will be of the form split/measurement for split in the set
  {train, valid, test} and measurement in the set {loss, error_rate}.

  Args:
    params: a dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: a dict of non-trainable model state. Passed as {'batch_stats':
      batch_stats} into flax_module.apply().
    dataset: Dataset returned from datasets.get_dataset. train, validation, and
      test sets.
    eval_num_batches: (int) The batch size used for evaluating on validation
      sets. Set to None to evaluate on the whole validation set.
    test_num_batches: (int) The batch size used for evaluating on test sets. Set
      to None to evaluate on the whole test set.
    eval_train_num_batches: (int) The batch size used for evaluating on train
      set. Set to None to evaluate on the whole training set.
    evaluate_batch_jitted: Computes the metrics on a sharded batch.
    finalize_batch_fn: Function to finalize the batch before passing to
      evaluate_batch_jitted. For sharding or reshaping.

  Returns:
    A dictionary of all computed metrics.
  """
  train_iter = dataset.eval_train_epoch(eval_train_num_batches)
  valid_iter = dataset.valid_epoch(eval_num_batches)
  test_iter = dataset.test_epoch(test_num_batches)

  metrics = {}
  for split_iter, split_name in zip([train_iter, valid_iter, test_iter],
                                    ['train', 'valid', 'test']):
    logging.info('Evaluating split: %s', split_name)
    split_metrics = evaluate(
        params,
        batch_stats,
        split_iter,
        evaluate_batch_jitted,
        finalize_batch_fn,
    )
    # Metrics are None if the dataset doesn't have that split
    if split_metrics is not None:
      metrics = _merge_and_apply_prefix(metrics, split_metrics,
                                        (split_name + '/'))
  return metrics


def get_batch_size(pytree):
  """Returns a pytree of batch sizes."""
  if isinstance(pytree['inputs'], jraph.GraphsTuple):
    pytree = pytree[
        'inputs'
    ].n_node  # Infer bsz from node field of GraphTuple.
  batch_sizes_pytree = jax.tree.map(lambda x: x.shape[0], pytree)
  return batch_sizes_pytree
