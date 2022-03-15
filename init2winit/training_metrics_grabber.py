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

"""Grab training metrics throughout training."""

from typing import Any, Dict

from flax import serialization
from flax import struct
from init2winit.utils import array_append
from init2winit.utils import total_tree_norm_sql2
import jax
import jax.numpy as jnp
import numpy as np


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


def _update_param_tree_stats(param_tree_stats, model_gradient,
                             old_params, new_params, config):
  """Update the parameter tree statistics.

  Args:
    param_tree_stats: (pytree) The current parameter tree statistics, as a
      pytree with a _MetricsLeafState as each leaf.
    model_gradient: (pytree) The current gradient.
    old_params: (pytree) The params before the param update.
    new_params: (pytree) The params after the param update.
    config: (ConfigDict) The TrainingMetricsGrabber config.

  Returns:
    new_param_tree_stats: (pytree) The new parameter tree statistics, as a
      pytree of the same form as `param_tree_stats`.
  """

  grads_flat, treedef = jax.tree_flatten(model_gradient)
  new_params_flat, _ = jax.tree_flatten(new_params)
  old_params_flat, _ = jax.tree_flatten(old_params)
  updates_flat = [new_param - old_param for (
      old_param, new_param) in zip(old_params_flat, new_params_flat)]

  # flatten_up_to here is needed to avoid flattening the _MetricsLeafState
  # nodes.
  param_tree_stats_flat = treedef.flatten_up_to(param_tree_stats)
  new_param_tree_stats_flat = [
      _update_param_stats(state, grad, update, new_param, config)
      for (state, grad, update, new_param) in zip(
          param_tree_stats_flat, grads_flat, updates_flat, new_params_flat)
  ]
  return jax.tree_unflatten(treedef, new_param_tree_stats_flat)


# TODO(jeremycohen, gilmer) maybe we should be logging the global step
def _update_global_stats(global_stats, train_cost,
                         old_params, new_params):
  """Update the global statistics.

  Args:
    global_stats: (dict) The current global statistics, as a dict with keys
      'train_cost_series', 'param_normsq_series', and 'update_normsq_series.'
    train_cost: (float) The train cost before the update.
    old_params: (pytree) The params before the param update.
    new_params: (pytree) The params after the param update.

  Returns:
    new_global_stats: (dict) The new global statistics, as a dict with the
      same keys as `global_stats`.
  """

  param_update = jax.tree_map(lambda x, y: x - y, new_params, old_params)

  param_normsq = total_tree_norm_sql2(old_params)
  param_update_normsq = total_tree_norm_sql2(param_update)

  train_cost_series = global_stats['train_cost_series']
  new_train_cost_series = array_append(train_cost_series, train_cost)

  param_normsq_series = global_stats['param_normsq_series']
  new_param_normsq_series = array_append(param_normsq_series, param_normsq)

  update_normsq_series = global_stats['update_normsq_series']
  new_update_normsq_series = array_append(update_normsq_series,
                                          param_update_normsq)

  return {
      'train_cost_series': new_train_cost_series,
      'param_normsq_series': new_param_normsq_series,
      'update_normsq_series': new_update_normsq_series
  }


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

    global_stats = {'train_cost_series': jnp.zeros(0),
                    'param_normsq_series': jnp.zeros(0),
                    'update_normsq_series': jnp.zeros(0)}

    param_tree_stats = jax.tree_map(_node_init, model_params)

    state = {'global_stats': global_stats,
             'param_tree_stats': param_tree_stats}

    return TrainingMetricsGrabber(state, config)

  def update(self, train_cost, model_gradient, old_params, new_params):
    """Computes a number of statistics from the model params and update.

    Global statistics computed:
      Train cost
      Parameter squared norm
      Update squared norm

    Parameter tree statistics computed:
      Per layer update variances and norms.
      Per layer gradient variance and norms.
      Per layer param norms.
      Ratio of parameter norm to update and update variance.

    Args:
      train_cost: (float) The train cost before the update
      model_gradient: A pytree of the same shape as the model_params pytree that
        was used when the metrics_grabber object was created.
      old_params: The params before the param update.
      new_params: The params after the param update.

    Returns:
      An updated class object.
    """
    new_global_stats = _update_global_stats(
        self.state['global_stats'], train_cost, old_params, new_params)
    new_param_tree_stats = _update_param_tree_stats(
        self.state['param_tree_stats'], model_gradient, old_params, new_params,
        self.config)
    new_state = {
        'global_stats': new_global_stats,
        'param_tree_stats': new_param_tree_stats
    }
    return self.replace(state=new_state)

  @staticmethod
  def to_state_dict(grabber):
    """Serialize a TraningMetricsGrabber.

    This function is called by flax.serialization.to_state_dict.

    Args:
      grabber: (TrainingMetricsGrabber) A TrainingMetricsGrabber to be
        serialized.

    Returns:
      a dict representing the TrainingMetricsGrabber
    """
    return serialization.to_state_dict(
        {'state': serialization.to_state_dict(grabber.state)})

  @staticmethod
  def from_state_dict(target, state_dict):
    """Restore a serialized TrainingMetricsGrabber.

    This function is called by flax.serialization.from_state_dict.

    Args:
      target: (TrainingMetricsGrabber) The "target" TrainingMetricsGrabber
        to be populated with contents from the state dict.
      state_dict: (dict) A dictionary, originating from to_state_dict(), whose
        contents should populate the target TrainingMetricsGrabber.

    Returns:
      the target TrainingMetricsGrabber populated with contents from state_dict.
    """
    state = serialization.from_state_dict(target.state, state_dict['state'])
    return target.replace(state=state)


serialization.register_serialization_state(
    TrainingMetricsGrabber,
    TrainingMetricsGrabber.to_state_dict,
    TrainingMetricsGrabber.from_state_dict,
    override=True)
