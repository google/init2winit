# coding=utf-8
# Copyright 2026 The init2winit Authors.
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
import copy
import functools
import json
import logging
import operator
import os.path
import time

from absl import logging as absl_logging
from clu import metric_writers
import flax
import flax.linen as nn
from init2winit import checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from tensorflow.io import gfile


exists = gfile.exists



class TrainingDivergedError(Exception):
  pass


def tree_norm_sql2(pytree):
  """Compute the param-wise squared L2 norm of a pytree."""
  return jax.tree.map(lambda x: jnp.linalg.norm(x.reshape(-1)) ** 2, pytree)


def total_tree_norm_sql2(pytree):
  """Compute the overall squared L2 norm of a pytree."""
  sql2_norms = tree_norm_sql2(pytree)
  return jax.tree_util.tree_reduce(operator.add, sql2_norms, 0)


def total_tree_norm_l2(pytree):
  """Compute the overall L2 norm of a pytree."""
  return jnp.sqrt(total_tree_norm_sql2(pytree))


def total_tree_sum(pytree):
  """Compute the overall sum of a pytree."""
  sums = jax.tree.map(jnp.sum, pytree)
  return jax.tree_util.tree_reduce(operator.add, sums, 0)


def array_append(full_array, to_append):
  """Append to an array."""
  to_append = jnp.expand_dims(to_append, axis=0)
  return jnp.concatenate((full_array, to_append))


def reduce_to_scalar(value):
  """Reduce an numpy array to a scalar by extracting the first element."""
  if isinstance(value, np.ndarray) or isinstance(value, jnp.ndarray):
    value = value.item(0)
  return value


def dtype_from_str(dtype_string):
  # We use strings to avoid having to import jnp into the config files.
  if dtype_string == 'float32':
    return jnp.float32
  elif dtype_string == 'float64':
    return jnp.float64
  elif dtype_string == 'bfloat16':
    return jnp.bfloat16
  else:
    raise ValueError('Invalid dtype: {}'.format(dtype_string))


def timed(f):
  """Decorator to time a function. Not correct on async-dispatch JAX code."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    retval = f(*args, **kwargs)
    return retval, time.time() - start_time

  return wrapper


def set_up_loggers(train_dir, xm_work_unit=None):
  """Creates a logger for eval metrics as well as initialization metrics."""
  csv_path = os.path.join(train_dir, 'measurements.csv')
  metrics_logger = MetricLogger(
      csv_path=csv_path,
      xm_work_unit=xm_work_unit,
      events_dir=train_dir)

  init_csv_path = os.path.join(train_dir, 'init_measurements.csv')
  init_json_path = os.path.join(train_dir, 'init_scalars.json')
  init_logger = MetricLogger(
      csv_path=init_csv_path,
      json_path=init_json_path,
      xm_work_unit=xm_work_unit)
  return metrics_logger, init_logger


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


class PytreeMetricLogger(object):
  """Used to log pytree metrics during training.
  
  This class is used to log pytree metrics during training. It is similar to
  MetricLogger, but it is designed to log pytree metrics.
  """

  def __init__(
      self,
      pytree_path,
  ):
    self._pytree_path = pytree_path

    orbax_file_options = ocp.checkpoint_manager.FileOptions(
        path_permission_mode=0o775,
        cns2_storage_options=ocp.options.Cns2StorageOptions(
            choose_store_cell=True,
        ),
    )
    self._orbax_checkpoint_manager = ocp.CheckpointManager(
        self._pytree_path,
        options=ocp.CheckpointManagerOptions(
            file_options=orbax_file_options,
            max_to_keep=1, create=True,
        ),
    )

  def write_pytree(self, pytree, step=0):
    """Record a serializable pytree to disk at the given step.

    Args:
      pytree: Any serializable pytree
      step: Integer. The global step.
    """
    state = dict(pytree=pytree)
    checkpoint.save_checkpoint(
        step,
        state=state,
        orbax_checkpoint_manager=self._orbax_checkpoint_manager,
    )

  def wait_until_pytree_checkpoint_finished(self):
    self._orbax_checkpoint_manager.wait_until_finished()

  def latest_pytree_checkpoint_step(self):
    return self._orbax_checkpoint_manager.latest_step()

  def load_latest_pytree(self, target=None):
    """Load pytree from checkpoint."""
    if target:
      target = dict(pytree=target)
    logging.info('target: %s', target)
    loaded_target = checkpoint.load_latest_checkpoint(
        target=target,
        orbax_checkpoint_manager=self._orbax_checkpoint_manager,
    )
    if loaded_target:
      return loaded_target['pytree']
    else:
      return target


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
               events_dir=None,
               **logger_kwargs):
    """Create a recorder for metrics, as CSV or JSON.


    Args:
      csv_path: A filepath to a CSV file to append to.
      json_path: An optional filepath to a JSON file to append to.
      events_dir: Optional. If specified, save tfevents summaries to this
        directory.
      **logger_kwargs: Optional keyword arguments, whose only valid parameter
        name is an optional XM WorkUnit used to also record metrics to XM as
        MeasurementSeries.
    """
    self._measurements = {}
    self._csv_path = csv_path
    self._json_path = json_path
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
    try:
      with gfile.GFile(self._csv_path) as csv_file:
        measurements = pd.read_csv(csv_file)
        measurements = pd.concat([measurements, pd.DataFrame([metrics])])
    except (pd.errors.EmptyDataError, gfile.FileError) as e:
      measurements = pd.DataFrame([metrics], columns=sorted(metrics.keys()))
      if isinstance(e, pd.errors.EmptyDataError):
        # TODO(ankugarg): Identify root cause for the corrupted file.
        # Most likely it's preemptions or file-write error.
        logging.info('Measurements file is empty. Create a new one, starting '
                     'with metrics from this step.')
    # TODO(gdahl,gilmer): Should this be an atomic file?
    with gfile.GFile(self._csv_path, 'w') as csv_file:
      measurements.to_csv(csv_file, index=False)

    if 'global_step' in metrics:
      metrics['global_step'] = int(reduce_to_scalar(metrics['global_step']))

    if self._xm_work_unit and self._xm_work_unit.id > -1:
      for name, value in metrics.items():
        if name not in self._measurements:
          self._measurements[name] = self._xm_work_unit.get_measurement_series(
              label=name)
        try:
          self._measurements[name].create_measurement(
              objective_value=reduce_to_scalar(value),
              step=metrics['global_step'],
          )
        except TypeError as e:
          logging.info('Failed to create measurement for %s: %s', name, value)
          raise e

    if self._tb_metric_writer:
      self._tb_metric_writer.write_scalars(
          step=metrics['global_step'], scalars=metrics)
      # This gives a 1-2% slowdown in steps_per_sec on cifar-10 with batch
      # size 512. We could only flush at the end of training to optimize this.
      self._tb_metric_writer.flush()

  def append_json_object(self, json_obj):
    """Append a json serializable object to the json file."""

    if not self._json_path:
      raise ValueError('Attempting to write to a null json path')
    if exists(self._json_path):
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
    shape_dict = flax.core.pretty_repr(jax.tree.map(lambda x: x.shape, pytree))
    with gfile.GFile(json_path, 'w') as json_file:
      json_file.write(shape_dict)

  absl_logging.info('Printing model param shapes.')
  shape_dict = jax.tree.map(_summary_str, pytree)
  absl_logging.info(flax.core.pretty_repr(shape_dict))
  total_params = jax.tree_util.tree_reduce(
      operator.add, jax.tree.map(lambda x: x.size, pytree))
  absl_logging.info('Total params: %d', total_params)


def tabulate_model(model, hps):
  """Logs a table of the flax module model parameters.

  Args:
    model: init2winit BaseModel
    hps: ml_collections.config_dict.config_dict.ConfigDict
  """
  tabulate_fn = nn.tabulate(model.flax_module, jax.random.PRNGKey(0),
                            console_kwargs={'force_terminal': False,
                                            'force_jupyter': False,
                                            'width': 240},
                            )
  fake_inputs_hps = copy.copy(hps)
  fake_inputs_hps.batch_size = 2
  fake_inputs = model.get_fake_inputs(fake_inputs_hps)
  # Currently only two models implement the get_fake_batch.
  # Only attempt to log if we get a valid fake_input_batch.
  if fake_inputs:
    absl_logging.info(tabulate_fn(*fake_inputs, train=False,))


def edit_distance(source, target):
  """Computes edit distance between source string and target string.

  This function assumes words are seperated by a single space.

  Args:
    source: source string.
    target: target string.

  Returns:
    Edit distance between source string and target string.
  """
  source = source.split()
  target = target.split()

  num_source_words = len(source)
  num_target_words = len(target)

  distance = np.zeros((num_source_words + 1, num_target_words + 1))

  for i in range(num_source_words + 1):
    for j in range(num_target_words + 1):
      # If first string is empty, only option is to
      # insert all words of second string
      if i == 0:
        distance[i][j] = j  # Min. operations = j

      # If second string is empty, only option is to
      # remove all characters of second string
      elif j == 0:
        distance[i][j] = i  # Min. operations = i

      # If last characters are same, ignore last char
      # and recur for remaining string
      elif source[i - 1] == target[j - 1]:
        distance[i][j] = distance[i - 1][j - 1]

      # If last character are different, consider all
      # possibilities and find minimum
      else:
        distance[i][j] = 1 + min(
            distance[i][j - 1],  # Insert
            distance[i - 1][j],  # Remove
            distance[i - 1][j - 1])  # Replace

  return distance[num_source_words][num_target_words]


def combine_gathered(x):
  if len(x.shape) < 3:
    raise ValueError('Rank of input array to combine_gathered must be > 3')

  n_device, n_batch, *lengths = x.shape
  flattened = x.reshape(n_device * n_batch, *lengths)

  return flattened


def use_mock_tpu_backend() -> bool:
  """Helper function to determine if mock TPU backend is used."""
  return str(jax.devices()[0]).startswith(
      ('MOCK_TPU', 'MegaScalePjRtDevice(wrapped=MOCK_TPU')
  )
