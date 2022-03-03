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
from typing import Any, Dict

from absl import logging as absl_logging
from clu import metric_writers
from flax import serialization
from flax import struct
from flax.training import checkpoints as flax_checkpoints
from init2winit import checkpoint
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
import numpy as np
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
  unreplicated_metrics_tree = jax.tree_map(lambda x: x[0],
                                           training_metrics_grabber.state)

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


@struct.dataclass
class _MetricsLeafState:
  # These won't actualy be np arrays, just for type checking.
  grad_ema: np.ndarray
  grad_sq_ema: np.ndarray
  param_norm: np.ndarray
  update_ema: np.ndarray
  update_sq_ema: np.ndarray


def _update_param_stats(leaf_state, param_gradient, update, new_param, config):
  """Updates the leaf state with the new layer statistics.

  Args:
    leaf_state: A _MetricsLeafState.
    param_gradient: The most recent gradient at the given layer.
    update: The optimizer update at the given layer.
    new_param: The updated layer parameters.
    config: See docstring of TrainingMetricsGrabber.

  Returns:
    Updated leaf state containing the new layer statistics.
  """
  ema_beta = config['ema_beta']
  param_grad_sq = jnp.square(param_gradient)

  grad_sq_ema = ema_beta * leaf_state.grad_sq_ema + (1.0 -
                                                     ema_beta) * param_grad_sq
  grad_ema = ema_beta * leaf_state.grad_ema + (1.0 - ema_beta) * param_gradient

  update_sq = jnp.square(update)
  update_ema = ema_beta * leaf_state.update_ema + (1.0 - ema_beta) * update
  update_sq_ema = ema_beta * leaf_state.update_sq_ema + (1.0 -
                                                         ema_beta) * update_sq

  param_norm = jnp.linalg.norm(new_param.reshape(-1))

  return _MetricsLeafState(
      grad_ema=grad_ema,
      grad_sq_ema=grad_sq_ema,
      param_norm=param_norm,
      update_ema=update_ema,
      update_sq_ema=update_sq_ema)


def _validate_config(config):
  if 'ema_beta' not in config:
    raise ValueError('Eval config requires field ema_beta')


@struct.dataclass
class TrainingMetricsGrabber:
  """Flax object used to grab gradient statistics during training.

  This class will be passed to the trainer update function, and can be used
  to record statistics of the gradient during training. The API is
  new_metrics_grabber = training_metrics_grabber.update(grad, batch_stats).
  Currently, this records an ema of the model gradient and squared gradient.
  The this class is configured with the config dict passed to create().
  Current keys:
    ema_beta: The beta used when computing the exponential moving average of the
      gradient variance as follows:
      var_ema_{i+1} = (beta) * var_ema_{i} + (1-beta) * current_variance.
  """

  state: Any
  config: Dict[str, Any]

  # This pattern is needed to maintain functional purity.
  @staticmethod
  def create(model_params, config):
    """Build the TrainingMetricsGrabber.

    Args:
      model_params: A pytree containing model parameters.
      config: Dictionary specifying the grabber configuration. See class doc
        string for relevant keys.

    Returns:
      The build grabber object.
    """

    def _node_init(x):
      return _MetricsLeafState(
          grad_ema=jnp.zeros_like(x),
          grad_sq_ema=jnp.zeros_like(x),
          param_norm=0.0,
          update_ema=jnp.zeros_like(x),
          update_sq_ema=jnp.zeros_like(x),
      )

    _validate_config(config)
    gradient_statistics = jax.tree_map(_node_init, model_params)

    return TrainingMetricsGrabber(gradient_statistics, config)

  def update(self, model_gradient, old_params, new_params):
    """Computes a number of statistics from the model params and update.

    Statistics computed:
      Per layer update variances and norms.
      Per layer gradient variance and norms.
      Per layer param norms.
      Ratio of parameter norm to update and update variance.

    Args:
      model_gradient: A pytree of the same shape as the model_params pytree that
        was used when the metrics_grabber object was created.
      old_params: The params before the param update.
      new_params: The params after the param update.

    Returns:
      An updated class object.
    """
    grads_flat, treedef = jax.tree_flatten(model_gradient)
    new_params_flat, _ = jax.tree_flatten(new_params)
    old_params_flat, _ = jax.tree_flatten(old_params)

    # flatten_up_to here is needed to avoid flattening the _MetricsLeafState
    # nodes.
    state_flat = treedef.flatten_up_to(self.state)
    new_states_flat = [
        _update_param_stats(state, grad, new_param - old_param, new_param,
                            self.config) for state, grad, old_param, new_param
        in zip(state_flat, grads_flat, old_params_flat, new_params_flat)
    ]

    return self.replace(state=jax.tree_unflatten(treedef, new_states_flat))

  def state_dict(self):
    return serialization.to_state_dict(
        {'state': serialization.to_state_dict(self.state)})

  def restore_state(self, state, state_dict):
    """Restore the state from the state dict.

    Allows for checkpointing the class object.

    Args:
      state: the class state.
      state_dict: the state dict containing the desired new state of the object.

    Returns:
      The restored class object.
    """

    state = serialization.from_state_dict(state, state_dict['state'])
    return self.replace(state=state)


serialization.register_serialization_state(
    TrainingMetricsGrabber,
    TrainingMetricsGrabber.state_dict,
    TrainingMetricsGrabber.restore_state,
    override=True)


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
