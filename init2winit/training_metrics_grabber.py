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

import operator

from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

# The TrainingMetricsGrabber configs will be these defaults overridden
# by any overrides passsed to TrainingMetricsGrabber.create.
# See the class doc string for an overview of the keys and what they mean.
DEFAULT_CONFIG = ConfigDict({
    'ema_beta': 0.9,
    'enable_train_cost': False,
    'enable_param_norms': False,
    'enable_update_norms': False,
    'enable_ema': False,
})


def make_training_metrics(num_train_steps, **config_overrides):
  """Creates functions for managing training metrics.

  Training metrics are handled in a functional, "jax-onic" way, similar to
  optax optimizers: there is a state pytree (whose precise structure depends
  on the config settings), and a set of functions to manipulate that state.

  The three functions are:
    (1) an initializer, which initializes the training metrics state
       (given the network param shapes);
    (2) an updater, which updates the training metrics state; and
    (3) a summarizer, which summarizes the training metrics state into a summary
      tree.

  The behavior of these functions is customizable via the configs.  The
  final configs used to configure the training metrics functionality are a
  combination of (1) the default configs in DEFAULT_CONFIG, and (2) the
  config overrides passed as arguments to this function.

  The config keys and their meanings are:
    - enable_train_cost (bool): if true, the metrics state will have a field
        "train_cost" which is a jnp array of length num_train_steps, and which
        stores the train cost at every step of training (padded by zeros).
    - enable_param_norms (bool): if true, the metrics state will have a field
        "param_norms" which is a pytree in the shape of the model params whose
        leaves are jnp arrays of length num_train_steps.
    - enable_update_norms (bool) if true, the metrics state will have a field
        "update_norms" which is a pytree in the shape of the model params whose
        leaves are jnp arrays of length num_train_steps.
    - enable_ema (bool): if true, the metrics state will have fields "grad_ema",
        "grad_sq_ema", "update_ema", and "update_sq_ema" containing
        exponential moving averages of the gradient, update, elementwise squared
        gradient, and elementwise squared update; and the summary tree will
        contain estimates of the gradient variance and update variance.
    - ema_beta (float): if enable_ema=true, the EMA's will use this value for
        their "beta" averaging parameter.

  Args:
    num_train_steps: (int) the number of steps of training.  We use this to
      determine the shape of the arrays that store per-step time series.
    **config_overrides: optional overrides for the training_metrics configs.
      Config keys which are not overridden will retain their default values.

  Returns:
    init_fn: (function) initializes the training metrics state
    update_fn: (function) updates the training metrics state
    summarize_fn: (function): summarizes the training metrics state
  """

  config = ConfigDict(DEFAULT_CONFIG)
  config.update(config_overrides)

  def init_fn(params):
    """Initialize the training metrics state.

    Args:
      params: (pytree) A pytree of model parameters.  Used for its shape
        information.

    Returns:
      metrics_state: (pytree): The initial training metrics state.  This is
        a pytree whose keys are the different training metrics; many of the
        corresponding values are pytrees of the same shape as the model params,
        though some are just scalars.
    """
    metrics_state = {}
    metrics_state['param_norm'] = jax.tree_map(lambda x: 0.0, params)
    if config['enable_train_cost']:
      metrics_state['train_cost'] = jnp.zeros(num_train_steps)
    if config['enable_param_norms']:
      metrics_state['param_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    if config['enable_update_norms']:
      metrics_state['update_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    if config['enable_ema']:
      metrics_state['grad_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['grad_sq_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['update_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['update_sq_ema'] = jax.tree_map(jnp.zeros_like, params)
    return metrics_state

  def update_fn(metrics_state, step, train_cost, grad, old_params, new_params):
    """Update the training metrics state.

    Args:
      metrics_state: (pytree) The current training metrics state.
      step: (int) the global step of training.
      train_cost: (float) The current train cost.
      grad: (pytree, of same shape as params): The current gradient.
      old_params: (pytree, of same shape as params): The parameters before the
        update.
      new_params: (pytree, of same shape as params): The parameters after the
        update.

    Returns:
      next_metrics_state: (pytree) The next training metrics state.
    """
    update = jax.tree_map(lambda x, y: x - y, old_params, new_params)
    grad_sq = jax.tree_map(jnp.square, grad)
    update_sq = jax.tree_map(jnp.square, update)

    param_norm = jax.tree_map(_compute_leaf_norms, old_params)

    next_metrics_state = {}
    next_metrics_state['param_norm'] = param_norm
    if config['enable_train_cost']:
      next_metrics_state['train_cost'] = metrics_state['train_cost'].at[
          step].set(train_cost)
    if config['enable_param_norms']:
      next_metrics_state['param_norms'] = _set_pytree_idx(
          metrics_state['param_norms'], param_norm, step)
    if config['enable_update_norms']:
      update_norm = jax.tree_map(_compute_leaf_norms, update)
      next_metrics_state['update_norms'] = _set_pytree_idx(
          metrics_state['update_norms'], update_norm, step)
    if config['enable_ema']:
      beta = config['ema_beta']
      next_metrics_state['grad_ema'] = _advance_ema(
          metrics_state['grad_ema'], grad, beta)
      next_metrics_state['grad_sq_ema'] = _advance_ema(
          metrics_state['grad_sq_ema'], grad_sq, beta)
      next_metrics_state['update_ema'] = _advance_ema(
          metrics_state['update_ema'], update, beta)
      next_metrics_state['update_sq_ema'] = _advance_ema(
          metrics_state['update_sq_ema'], update_sq, beta)

    return next_metrics_state

  def summarize_fn(metrics_state):
    """Construct a summary tree based on the current training metrics state.

    Args:
      metrics_state: (pytree) The current training metrics state.

    Returns:
      summary_tree: (pytree) A summary of the training metrics state.
    """

    # this dict will map from "summary key" to "pytree of same shape as params"
    summary = {}

    summary['param_norm'] = metrics_state['param_norm']

    if config['enable_ema']:

      def compute_var(first_moment, second_moment):
        return (second_moment - first_moment**2).sum()

      summary['grad_var'] = jax.tree_map(compute_var,
                                         metrics_state['grad_ema'],
                                         metrics_state['grad_sq_ema'])

      summary['update_var'] = jax.tree_map(compute_var,
                                           metrics_state['update_ema'],
                                           metrics_state['update_sq_ema'])

      summary['update_ratio'] = jax.tree_map(operator.truediv,
                                             summary['update_var'],
                                             metrics_state['param_norm'])

    # This dict will map from "summary key" to "flattened pytree of same shape
    # as params."
    flat_summary = _map_values(model_utils.flatten_dict, summary)

    return flat_summary

  return init_fn, update_fn, summarize_fn


def _map_values(f, dictionary):
  """Create a new dict by mapping all the values in a dict through f."""
  return {k: f(v) for (k, v) in dictionary.items()}


def _advance_ema(cur_ema, new_val, beta):
  """Advance an exponential moving average."""
  return jax.tree_map(lambda cur, new: beta * cur + (1 - beta) * new,
                      cur_ema,
                      new_val)


def _compute_leaf_norms(pytree):
  """Compute the norm of all leaves in a pytree."""
  return jax.tree_map(lambda leaf: jnp.linalg.norm(leaf.reshape(-1)), pytree)


def _set_pytree_idx(pytree_of_arrs, new_pytree, idx):
  """Incorporate a new pytree into a pytree of arrays.

  Args:
    pytree_of_arrs: (pytree) a pytree of float arrays
    new_pytree: (pytree) a pytree of floats
    idx: (int) an index

  Returns:
    a pytree where we set the "idx" index of each leaf in pytree_of_arrs to
      the corresponding leaf in new_pytree.

  """
  def set_arr(arr, new_value):
    return arr.at[idx].set(new_value)
  return jax.tree_map(set_arr, pytree_of_arrs, new_pytree)
