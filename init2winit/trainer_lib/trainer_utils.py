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

"""Utility functions related to training."""
from absl import logging

from flax import jax_utils
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
import jax


def log_epoch_report(report, metrics_logger):
  logging.info('Step %d, steps/second: %f, report: %r', report['global_step'],
               report['steps_per_sec'], report)
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
  if eval_steps:
    return global_step in eval_steps
  return global_step % eval_frequency == 0


def check_for_early_stopping(
    early_stopping_target_name,
    early_stopping_target_value,
    early_stopping_mode,
    eval_report):
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
    if early_stopping_mode == 'above':
      return (eval_report[early_stopping_target_name] >=
              early_stopping_target_value)
    else:
      return (eval_report[early_stopping_target_name] <=
              early_stopping_target_value)


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
