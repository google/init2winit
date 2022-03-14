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

"""Utility functions for logging and recording training metrics."""

import concurrent.futures
import functools
import json
import logging
import operator
import os.path
import time

from absl import logging as absl_logging
from clu import metric_writers
from flax.training import checkpoints as flax_checkpoints
from init2winit import checkpoint
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
import pandas as pd
from tensorflow.io import gfile



class TrainingDivergedError(Exception):
  pass


def dtype_from_str(dtype_string):
  # We use strings to avoid having to import jnp into the config files.
  if dtype_string == 'float32':
    return jnp.float32
  elif dtype_string == 'bfloat16':
    return jnp.bfloat16
  else:
    raise ValueError('Invalid dtype: {}'.format(dtype_string))


def tree_norm_sql2(pytree):
  """Compute the param-wise L2 norm of a pytree."""
  return jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1)) ** 2, pytree)


def total_tree_norm_sql2(pytree):
  """Compute the overall L2 norm of a pytree."""
  sql2_norms = tree_norm_sql2(pytree)
  return jax.tree_util.tree_reduce(operator.add, sql2_norms, 0)


def array_append(full_array, to_append):
  """Append a scalar to an array."""
  to_append = jnp.expand_dims(to_append, axis=0)
  return jnp.concatenate((full_array, to_append))


def timed(f):
  """Decorator to time the execution of a function."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    retval = f(*args, **kwargs)
    return retval, time.time() - start_time

  return wrapper


def set_up_loggers(train_dir, xm_work_unit=None):
  """Creates a logger for eval metrics as well as initialization metrics."""
  csv_path = os.path.join(train_dir, 'measurements.csv')
  pytree_path = os.path.join(train_dir, 'training_metrics')
  metrics_logger = MetricLogger(
      csv_path=csv_path,
      pytree_path=pytree_path,
      xm_work_unit=xm_work_unit,
      events_dir=train_dir)

  init_csv_path = os.path.join(train_dir, 'init_measurements.csv')
  init_json_path = os.path.join(train_dir, 'init_scalars.json')
  init_logger = MetricLogger(
      csv_path=init_csv_path,
      json_path=init_json_path,
      xm_work_unit=xm_work_unit)
  return metrics_logger, init_logger


def get_summary_tree(training_metrics_grabber):
  """Extracts desired training statistics from the grabber state.

  Currently this function will compute the scalar aggregate gradient variance
  for every weight matrix of the model. Future iterations of this function
  may depend on the metrics_grabber config.

  Args:
    training_metrics_grabber: TrainingMetricsGrabber object.

  Returns:
    A dict of different aggregate training statistics.
  """
  unreplicated_metrics_tree = jax.tree_map(
      lambda x: x[0], training_metrics_grabber.state['param_tree_stats'])

  # Example key: Layer1/conv1/kernel/
  # NOTE: jax.tree_map does not work here, because tree_map will additionally
  # flatten the node state, while model_utils.flatten_dict will consider the
  # node object a leaf.
  flat_metrics = model_utils.flatten_dict(unreplicated_metrics_tree)

  # Grab just the gradient_variance terms.
  def _reduce_node(node):
    # Var[g] = E[g^2] - E[g]^2
    grad_var_ema = node.grad_sq_ema - jnp.square(node.grad_ema)
    update_var_ema = node.update_sq_ema - jnp.square(node.update_ema)
    return {
        'grad_var': grad_var_ema.sum(),
        'param_norm': node.param_norm,
        'update_var': update_var_ema.sum(),
        'update_ratio': update_var_ema.sum() / node.param_norm,
    }

  return {k: _reduce_node(flat_metrics[k]) for k in flat_metrics}


def run_in_parallel(function, list_of_kwargs_to_function, num_workers):
  """Run a function on a list of kwargs in parallel with ThreadPoolExecutor.

  Adapted from code by mlbileschi.
  Args:
    function: a function.
    list_of_kwargs_to_function: list of dictionary from string to argument
      value. These will be passed into `function` as kwargs.
    num_workers: int.

  Returns:
    list of return values from function.
  """
  if num_workers < 1:
    raise ValueError(
        'Number of workers must be greater than 0. Was {}'.format(num_workers))

  with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    futures = []
    logging.info(
        'Adding %d jobs to process pool to run in %d parallel '
        'threads.', len(list_of_kwargs_to_function), num_workers)

    for kwargs in list_of_kwargs_to_function:
      f = executor.submit(function, **kwargs)
      futures.append(f)

    for f in concurrent.futures.as_completed(futures):
      if f.exception():
        # Propagate exception to main thread.
        raise f.exception()

  return [f.result() for f in futures]


def should_eval(global_step, eval_frequency, eval_steps):
  if eval_steps:
    return global_step in eval_steps
  return global_step % eval_frequency == 0


def add_log_file(logfile):
  """Replicate logs to an additional logfile.

  The caller is responsible for closing the logfile.
  Args:
    logfile: A file opened for appending to replicate logs to.
  """
  handler = logging.StreamHandler(logfile)
  handler.setFormatter(absl_logging.PythonFormatter())

  absl_logger = logging.getLogger('absl')
  absl_logger.addHandler(handler)


def load_pytrees(pytree_path):
  # Note that this assumes each pytree is some nested dict of arrays, which
  # don't need to be individually restored.
  state_dict = checkpoint.load_latest_checkpoint(
      pytree_path, prefix='training_metrics')
  if state_dict:
    pytree = state_dict['pytree']
    return [pytree[str(i)] for i in range(len(pytree))]
  return []


# TODO(gdahl,gilmer): Use atomic writes to avoid file corruptions due to
# preemptions.
class MetricLogger(object):
  """Used to log all measurements during training.

  Note: Writes are not atomic, so files may become corrupted if preempted at
  the wrong time.
  """

  def __init__(self,
               csv_path='',
               json_path='',
               pytree_path='',
               events_dir=None,
               **logger_kwargs):
    """Create a recorder for metrics, as CSV or JSON.


    Args:
      csv_path: A filepath to a CSV file to append to.
      json_path: An optional filepath to a JSON file to append to.
      pytree_path: Where to save trees of numeric arrays.
      events_dir: Optional. If specified, save tfevents summaries to this
        directory.
      **logger_kwargs: Optional keyword arguments, whose only valid parameter
        name is an optional XM WorkUnit used to also record metrics to XM as
        MeasurementSeries.
    """
    self._measurements = {}
    self._csv_path = csv_path
    self._json_path = json_path
    self._pytree_path = pytree_path
    if logger_kwargs:
      if len(logger_kwargs.keys()) > 1 or 'xm_work_unit' not in logger_kwargs:
        raise ValueError(
            'The only logger_kwarg that should be passed to MetricLogger is '
            'xm_work_unit.')
      self._xm_work_unit = logger_kwargs['xm_work_unit']
    else:
      self._xm_work_unit = None

    self._tb_metric_writer = None
    if events_dir:
      self._tb_metric_writer = metric_writers.create_default_writer(events_dir)

  def append_scalar_metrics(self, metrics):
    """Record a dictionary of scalar metrics at a given step.

    Args:
      metrics: a Dict of metric names to scalar values. 'global_step' is the
        only required key.
    """
    if gfile.exists(self._csv_path):
      with gfile.GFile(self._csv_path) as csv_file:
        measurements = pd.read_csv(csv_file)
      measurements = measurements.append([metrics])
    else:
      measurements = pd.DataFrame([metrics], columns=sorted(metrics.keys()))
    # TODO(gdahl,gilmer): Should this be an atomic file?
    with gfile.GFile(self._csv_path, 'w') as csv_file:
      measurements.to_csv(csv_file, index=False)
    if self._xm_work_unit:
      for name, value in metrics.items():
        if name not in self._measurements:
          self._measurements[name] = self._xm_work_unit.get_measurement_series(
              label=name)
        self._measurements[name].create_measurement(
            objective_value=value, step=metrics['global_step'])

    if self._tb_metric_writer:
      self._tb_metric_writer.write_scalars(
          step=int(metrics['global_step']), scalars=metrics)
      # This gives a 1-2% slowdown in steps_per_sec on cifar-10 with batch
      # size 512. We could only flush at the end of training to optimize this.
      self._tb_metric_writer.flush()

  def write_pytree(self, pytree):
    """Record a serializable pytree to disk, overwriting any previous state.

    Args:
      pytree: Any serializable pytree.
    """
    state = dict(pytree=pytree)
    checkpoint.save_checkpoint(
        self._pytree_path,
        step='',
        state=state,
        prefix='training_metrics',
        max_to_keep=None)

  def append_pytree(self, pytree):
    """Append and record a serializable pytree to disk.

    The pytree will be saved to disk as a list of pytree objects. Everytime
    this function is called, it will load the previous saved state, append the
    next pytree to the list, then save the appended list.

    Args:
      pytree: Any serializable pytree.
    """
    # Read the latest (and only) checkpoint, then append the new state to it
    # before saving back to disk.
    old_state = flax_checkpoints.restore_checkpoint(
        self._pytree_path, target=None, prefix='training_metrics')
    # Because we pass target=None, checkpointing will return the raw state
    # dict, where 'pytree' is a dict with keys ['0', '1', ...] instead of a
    # list.
    if old_state:
      state_list = old_state['pytree']
      state_list = [state_list[str(i)] for i in range(len(state_list))]
    else:
      state_list = []
    state_list.append(pytree)

    self.write_pytree(state_list)

  def append_json_object(self, json_obj):
    """Append a json serializable object to the json file."""

    if not self._json_path:
      raise ValueError('Attempting to write to a null json path')
    if gfile.exists(self._json_path):
      with gfile.GFile(self._json_path) as json_file:
        json_objs = json.loads(json_file.read())
      json_objs.append(json_obj)
    else:
      json_objs = [json_obj]
    # TODO(gdahl,gilmer): Should this be an atomic file?
    with gfile.GFile(self._json_path, 'w') as json_file:
      json_file.write(json.dumps(json_objs))


def _summary_str(param):
  total_norm = jnp.linalg.norm(param.reshape(-1))
  return '{} - {} - {}'.format(str(param.shape), param.size, total_norm)


def log_pytree_shape_and_statistics(pytree, json_path=None):
  """Logs the shape and norm of every array in the pytree."""
  if not pytree:
    absl_logging.info('Empty pytree')
    return

  if json_path:
    shape_dict = jax.tree_map(lambda x: x.shape, pytree).pretty_repr()
    with gfile.GFile(json_path, 'w') as json_file:
      json_file.write(shape_dict)

  absl_logging.info('Printing model param shapes.')
  shape_dict = jax.tree_map(_summary_str, pytree)
  absl_logging.info(shape_dict.pretty_repr())
  total_params = jax.tree_util.tree_reduce(
      operator.add, jax.tree_map(lambda x: x.size, pytree))
  absl_logging.info('Total params: %d', total_params)
