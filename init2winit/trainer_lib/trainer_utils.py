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
import functools
import time

from absl import logging

from flax import jax_utils
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import model_utils
import jax


class Timer:
  """TODO(dsuo): add documentation."""

  def __init__(self, name):
    self.name = name
    self.reset()

  def reset(self):
    self.splits = []
    self.duration = 0
    self.running = False

  def start(self):
    self._start_time = time.time()
    self.running = True

  def stop(self):
    if not self.running:
      raise ValueError(f'Timer {self.name} has not been started yet.')
    curr_duration = time.time() - self._start_time
    self.duration += curr_duration
    self.splits.append(curr_duration)
    self.running = False

  def set(self, duration):
    if self.running:
      raise ValueError(f'Cannot set timer {self.name} while it is running.')
    self.duration = duration

  def __enter__(self):
    self.start()

  def __exit__(self, exc_type, exc_value, exc_tb):
    self.stop()

  def __str__(self):
    return f'name: {self.name}, duration: {self.duration}, running: {self.running}'

  def __repr__(self):
    return f'Timer<{self.__str__()}>'


class TimerCollection:
  """TODO(dsuo): add documentation."""

  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(TimerCollection, cls).__new__(cls)
      cls._timers = {}
    return cls.instance

  def items(self):
    return self._timers.items()

  def timed(self, func, name=None):
    """Throwaway."""
    name = name or func.__name__
    if name in self._timers:
      name = f'{func.__module__}.{name}'
    timer = self.__call__(name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      timer.start()
      retval = func(*args, **kwargs)
      timer.stop()
      return retval

    return wrapper

  def __call__(self, name):
    if name not in self._timers:
      self._timers[name] = Timer(name)
    return self._timers[name]


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


def log_eta(pool, work_unit, global_step, steps_per_sec_train_only,
            num_train_steps, start_time, eval_frequency, eval_steps, eval_time):
  """Construct and ETA / total time entry."""
  msg = f'Steps: {global_step} / {num_train_steps} '
  msg += f'[{global_step / num_train_steps:.1%}] '

  # Time remaining from training
  train_eta = (num_train_steps - global_step) / steps_per_sec_train_only

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
               report['steps_per_sec_train_only'], report)
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
  on_step = eval_steps and global_step in eval_steps
  on_freq = (global_step % eval_frequency == 0)

  return on_step or on_freq


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
