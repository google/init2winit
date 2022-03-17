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
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

# The TrainingMetricsGrabber configs will be these defaults overridden
# by any overrides passsed to TrainingMetricsGrabber.create.
# See the class doc string for an overview of the keys and what they mean.
DEFAULT_CONFIG = ConfigDict({'ema_beta': 0.9})


@struct.dataclass
class _MetricsLeafState:
  # These won't actualy be np arrays, just for type checking.
  grad_ema: np.ndarray
  grad_sq_ema: np.ndarray
  param_norm: np.ndarray
  update_ema: np.ndarray
  update_sq_ema: np.ndarray


def _advance_ema(cur_ema, new_val, beta):
  """Advance an exponential moving average (EMA)."""
  return beta * cur_ema + (1 - beta) * new_val


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
  update_sq = jnp.square(update)

  grad_ema = _advance_ema(leaf_state.grad_ema, param_gradient, ema_beta)
  grad_sq_ema = _advance_ema(leaf_state.grad_sq_ema, param_grad_sq, ema_beta)
  update_ema = _advance_ema(leaf_state.update_ema, update, ema_beta)
  update_sq_ema = _advance_ema(leaf_state.update_sq_ema, update_sq, ema_beta)

  param_norm = jnp.linalg.norm(new_param.reshape(-1))

  return _MetricsLeafState(
      grad_ema=grad_ema,
      grad_sq_ema=grad_sq_ema,
      param_norm=param_norm,
      update_ema=update_ema,
      update_sq_ema=update_sq_ema)


@struct.dataclass
class TrainingMetricsGrabber:
  """Flax object used to grab gradient statistics during training.

  This class will be passed to the trainer update function, and can be used
  to record statistics of the gradient during training. The API is
  new_metrics_grabber = training_metrics_grabber.update(grad, batch_stats).
  Currently, this records an ema of the model gradient and squared gradient.
  The class is configured with the override dict passed to create() as well as
  the DEFAULT_CONFIG specificied above.
  Current keys:
    ema_beta: The beta used when computing the exponential moving average of the
      gradient variance as follows:
      var_ema_{i+1} = (beta) * var_ema_{i} + (1-beta) * current_variance.
  """

  state: Any
  config: Dict[str, Any]

  # This pattern is needed to maintain functional purity.
  @staticmethod
  def create(model_params, config_overrides):
    """Build the TrainingMetricsGrabber.

    Args:
      model_params: A pytree containing model parameters.
      config_overrides: Dictionary specifying config overrides.
        See class doc string for relevant keys.

    Returns:
      The build grabber object.
    """
    config = ConfigDict(DEFAULT_CONFIG)
    config.update(config_overrides)

    def _node_init(x):
      return _MetricsLeafState(
          grad_ema=jnp.zeros_like(x),
          grad_sq_ema=jnp.zeros_like(x),
          param_norm=0.0,
          update_ema=jnp.zeros_like(x),
          update_sq_ema=jnp.zeros_like(x),
      )

    gradient_statistics = jax.tree_map(_node_init, model_params)

    return TrainingMetricsGrabber(gradient_statistics, config.to_dict())

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
