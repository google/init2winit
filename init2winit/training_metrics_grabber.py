# coding=utf-8
# Copyright 2023 The init2winit Authors.
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

from flax.core import freeze
from init2winit.hessian.precondition import make_diag_preconditioner
from init2winit.model_lib import model_utils
from init2winit.optimizer_lib import utils as optimizer_utils
from init2winit.utils import total_tree_norm_l2
from init2winit.utils import total_tree_norm_sql2
from init2winit.utils import total_tree_sum
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
    'enable_gradient_norm': False,
    'enable_all_gradient_norms': False,
    'enable_batch_stats_norm': False,
    'enable_all_batch_stats_norms': False,
    'enable_update_norm': False,
    'enable_update_norms': False,
    'enable_preconditioner_normsq': False,
    'enable_semip_grad_normsq': False,
    'enable_ema': False,
    'optstate_sumsq_fields': [],
    'optstate_sumsq_param_wise_fields': [],
    'optstate_sum_fields': [],
    'optstate_sum_param_wise_fields': [],
    'enable_grafting_norms': False,
})


def make_training_metrics(num_train_steps, hps, **config_overrides):
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
    - enable_gradient_norm (bool) if true, the metrics state will have a field
        "gradient_norm" which is a jnp array of length num_train_steps
        containing a time series of the overall gradient norm.
    - enable_update_norm (bool) if true, the metrics state will have a field
        "update_norm" which is a jnp array of length num_train_steps containing
        a time series of the overall update norm.
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
    - optstate_sumsq_fields (list of str): record the squared Euclidean norm of
        each of these fields in the optimizer state.  If this list is non-empty,
        the metrics state will have a field "optstate_normsq" which is a dict
        where each key is a field name in optstate_normsq_fields, and each
        value is a jnp array of length num_time_steps containing the time
        series of the normsq of this optstate field.
    - optstate_sum_fields (list of str): record the sum of each of these fields
        in the optimizer state.  If this list is non-empty, the metrics state
        will have a field "optstate_sum" which is a dict where each key is a
        field name in optstate_sum_fields, and each value is a jnp array
        of length num_time_steps containing the time series of the sum
        of this optstate field.
    - enable_preconditioner_normsq (bool): if true, the metrics state will have
        a field "preconditioner_normsq" which is a jnp array of length
        num_train_steps containing a time series of the squared L2 norm of the
        preconditioner.  Adaptive optimizers only.  See the function
        make_diag_preconditioner() in hessian/precondition.py for more
        info on which optimizers are supported.
    - enable_semip_grad_normsq (bool): if true, the metrics state will have
        a field "semip_grad_normsq" which is a jnp array of length
        num_train_steps containing a time series of the squared L2 norm of the
        "semi-preconditioned" gradient.  Adaptive optimizers only.
    - enable_grafting_norms (bool): if true, the metrics state will have two 
        fields "mag_norms" and "dir_norms" which are pytrees in the shape of the
        model params whose leaves are jnp arrays of length num_train_steps. This
        will only work when you are using the grafting operation through 
        the kitchen_sink API.

  Args:
    num_train_steps: (int) the number of steps of training.  We use this to
      determine the shape of the arrays that store per-step time series.
    hps (ConfigDict): the init2winit hps.
    **config_overrides: optional overrides for the training_metrics configs.
      Config keys which are not overridden will retain their default values.

  Returns:
    init_fn: (function) initializes the training metrics state
    update_fn: (function) updates the training metrics state
    summarize_fn: (function): summarizes the training metrics state
  """

  config = ConfigDict(DEFAULT_CONFIG)
  config.update(config_overrides)

  def init_fn(params, batch_stats):
    """Initialize the training metrics state.

    Args:
      params: (pytree) A pytree of model parameters.  Used for its shape
        information.
      batch_stats: (pytree) A pytree of batch stats.  Used for its shape
        information.

    Returns:
      metrics_state: (pytree): The initial training metrics state.  This is
        a pytree whose keys are the different training metrics; many of the
        corresponding values are pytrees of the same shape as the model params,
        though some are just scalars.
    """
    metrics_state = {}
    metrics_state['param_norm'] = jnp.zeros(num_train_steps)
    if config['enable_train_cost']:
      metrics_state['train_cost'] = jnp.zeros(num_train_steps)
    if config['enable_param_norms']:
      metrics_state['param_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    if config['enable_batch_stats_norm']:
      metrics_state['batch_stats_norm'] = jnp.zeros(num_train_steps)
    if config['enable_all_batch_stats_norms']:
      metrics_state['all_batch_stats_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), batch_stats)
    if config['enable_gradient_norm']:
      metrics_state['gradient_norm'] = jnp.zeros(num_train_steps)
    if config['enable_all_gradient_norms']:
      metrics_state['all_gradient_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    if config['enable_update_norm']:
      metrics_state['update_norm'] = jnp.zeros(num_train_steps)
    if config['enable_update_norms']:
      metrics_state['update_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    if config['enable_ema']:
      metrics_state['grad_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['grad_sq_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['update_ema'] = jax.tree_map(jnp.zeros_like, params)
      metrics_state['update_sq_ema'] = jax.tree_map(jnp.zeros_like, params)
    if config['optstate_sumsq_fields']:
      metrics_state['optstate_sumsq'] = {
          field_name: jnp.zeros(num_train_steps)
          for field_name in config['optstate_sumsq_fields']
      }
    if config['optstate_sumsq_param_wise_fields']:
      metrics_state['optstate_sumsq_param_wise'] = {
          field_name: jax.tree_map(lambda x: jnp.zeros(num_train_steps), params)
          for field_name in config['optstate_sumsq_param_wise_fields']
      }
    if config['optstate_sum_fields']:
      metrics_state['optstate_sum'] = {
          field_name: jnp.zeros(num_train_steps)
          for field_name in config['optstate_sum_fields']
      }
    if config['optstate_sum_param_wise_fields']:
      metrics_state['optstate_sum_param_wise'] = {
          field_name: jax.tree_map(lambda x: jnp.zeros(num_train_steps), params)
          for field_name in config['optstate_sum_param_wise_fields']
      }
    if config['enable_preconditioner_normsq']:
      metrics_state['preconditioner_normsq'] = jnp.zeros(num_train_steps)
    if config['enable_semip_grad_normsq']:
      metrics_state['semip_grad_normsq'] = jnp.zeros(num_train_steps)
    if config['enable_grafting_norms']:
      metrics_state['mag_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
      metrics_state['dir_norms'] = jax.tree_map(
          lambda x: jnp.zeros(num_train_steps), params)
    return metrics_state

  def update_fn(metrics_state, step, train_cost, grad, old_params, new_params,
                optimizer_state, batch_stats):
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
      optimizer_state: the optax optimizer state.
      batch_stats: batch stats

    Returns:
      next_metrics_state: (pytree) The next training metrics state.
    """
    param_norm = jax.tree_map(_compute_leaf_norms, old_params)
    grad_norm = jax.tree_map(_compute_leaf_norms, grad)
    batch_stats_norm = jax.tree_map(_compute_leaf_norms, batch_stats)
    if (config['enable_update_norm'] or config['enable_update_norms'] or
        config['enable_ema']):
      update = jax.tree_map(lambda x, y: x - y, old_params, new_params)

    next_metrics_state = {}
    next_metrics_state['param_norm'] = metrics_state['param_norm'].at[
        step].set(total_tree_norm_l2(param_norm))
    if config['enable_train_cost']:
      next_metrics_state['train_cost'] = metrics_state['train_cost'].at[
          step].set(train_cost)
    if config['enable_param_norms']:
      next_metrics_state['param_norms'] = _set_pytree_idx(
          metrics_state['param_norms'], param_norm, step)
    if config['enable_batch_stats_norm']:
      next_metrics_state['batch_stats_norm'] = metrics_state[
          'batch_stats_norm'].at[step].set(
              total_tree_norm_l2(batch_stats))
    if config['enable_all_batch_stats_norms']:
      next_metrics_state['all_batch_stats_norms'] = _set_pytree_idx(
          metrics_state['all_batch_stats_norms'], batch_stats_norm, step)
    if config['enable_gradient_norm']:
      next_metrics_state['gradient_norm'] = metrics_state['gradient_norm'].at[
          step].set(total_tree_norm_l2(grad))
    if config['enable_all_gradient_norms']:
      next_metrics_state['all_gradient_norms'] = _set_pytree_idx(
          metrics_state['all_gradient_norms'], grad_norm, step)
    if config['enable_update_norm']:
      next_metrics_state['update_norm'] = metrics_state['update_norm'].at[
          step].set(total_tree_norm_l2(update))
    if config['enable_update_norms']:
      update_norm = jax.tree_map(_compute_leaf_norms, update)
      next_metrics_state['update_norms'] = _set_pytree_idx(
          metrics_state['update_norms'], update_norm, step)
    if config['enable_ema']:
      beta = config['ema_beta']
      grad_sq = jax.tree_map(jnp.square, grad)
      update_sq = jax.tree_map(jnp.square, update)
      next_metrics_state['grad_ema'] = _advance_ema(
          metrics_state['grad_ema'], grad, beta)
      next_metrics_state['grad_sq_ema'] = _advance_ema(
          metrics_state['grad_sq_ema'], grad_sq, beta)
      next_metrics_state['update_ema'] = _advance_ema(
          metrics_state['update_ema'], update, beta)
      next_metrics_state['update_sq_ema'] = _advance_ema(
          metrics_state['update_sq_ema'], update_sq, beta)
    if config['optstate_sumsq_fields']:
      next_metrics_state['optstate_sumsq'] = {}
      for field_name in config['optstate_sumsq_fields']:
        field = optimizer_utils.extract_field(optimizer_state, field_name)
        if field is None:
          raise ValueError('optimizer state has no field {}'.format(field_name))
        field_normsq = total_tree_norm_sql2(field)
        next_metrics_state['optstate_sumsq'][field_name] = metrics_state[
            'optstate_sumsq'][field_name].at[step].set(field_normsq)
    if config['optstate_sumsq_param_wise_fields']:
      next_metrics_state['optstate_sumsq_param_wise'] = {}
      for field_name in config['optstate_sumsq_param_wise_fields']:
        field = optimizer_utils.extract_field(optimizer_state, field_name)
        if field is None:
          raise ValueError('optimizer state has no field {}'.format(field_name))
        field_normsq = jax.tree_map(_compute_leaf_norms, field)
        field_normsqs = jax.tree_map(jnp.square, field_normsq)
        next_metrics_state['optstate_sumsq_param_wise'][field_name] = (
            _set_pytree_idx(
                metrics_state['optstate_sumsq_param_wise'][field_name],
                field_normsqs,
                step,
            )
        )
    if config['optstate_sum_fields']:
      next_metrics_state['optstate_sum'] = {}
      for field_name in config['optstate_sum_fields']:
        field = optimizer_utils.extract_field(optimizer_state, field_name)
        if field is None:
          raise ValueError('optimizer state has no field {}'.format(field_name))
        field_normsq = total_tree_sum(field)
        next_metrics_state['optstate_sum'][field_name] = metrics_state[
            'optstate_sum'][field_name].at[step].set(field_normsq)
    if config['optstate_sum_param_wise_fields']:
      next_metrics_state['optstate_sum_param_wise'] = {}
      for field_name in config['optstate_sum_param_wise_fields']:
        field = optimizer_utils.extract_field(optimizer_state, field_name)
        if field is None:
          raise ValueError('optimizer state has no field {}'.format(field_name))
        field_sums = jax.tree_map(jnp.sum, field)
        next_metrics_state['optstate_sum_param_wise'][field_name] = (
            _set_pytree_idx(
                metrics_state['optstate_sum_param_wise'][field_name],
                field_sums,
                step,
            )
        )
    if (config['enable_preconditioner_normsq'] or
        config['enable_semip_grad_normsq']):
      preconditioner = freeze(
          make_diag_preconditioner(hps['optimizer'], hps['opt_hparams'],
                                   optimizer_state, ConfigDict({})))
      if config['enable_preconditioner_normsq']:
        normsq = total_tree_norm_sql2(preconditioner)
        next_metrics_state['preconditioner_normsq'] = metrics_state[
            'preconditioner_normsq'].at[step].set(normsq)
      if config['enable_semip_grad_normsq']:
        semip_grad = jax.tree_map(lambda g, p: g / (p**0.5),
                                  grad, preconditioner)
        semip_grad_normsq = total_tree_norm_sql2(semip_grad)
        next_metrics_state['semip_grad_normsq'] = metrics_state[
            'semip_grad_normsq'].at[step].set(semip_grad_normsq)
    if config['enable_grafting_norms']:
      mag_norm = optimizer_utils.extract_field(optimizer_state, 'mag_norm')
      if mag_norm is None:
        raise ValueError('optimizer state has no field {}'.format('mag_norm'))
      mag_norm = freeze(mag_norm)
      next_metrics_state['mag_norms'] = _set_pytree_idx(
          metrics_state['mag_norms'], mag_norm, step)
      dir_norm = optimizer_utils.extract_field(optimizer_state, 'dir_norm')
      if dir_norm is None:
        raise ValueError('optimizer state has no field {}'.format('dir_norm'))
      dir_norm = freeze(dir_norm)
      next_metrics_state['dir_norms'] = _set_pytree_idx(
          metrics_state['dir_norms'], dir_norm, step)

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
