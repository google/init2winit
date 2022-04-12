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

r"""Debugging tool for identifying problematic layers in a network.

"""

import os

import flax
from init2winit.checkpoint import load_pytree
from init2winit.utils import array_append
from init2winit.utils import tree_norm_sql2
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile


def tag_residual_activations(module,
                             identity_path,
                             other_path,
                             name='residual'):
  """Used in inspecting the forward pass and residual networks.

  Residual connections involve adding x + F(x) for some function F. This
  function is used to log both ||x||_2 and ||F(x)||_2. This calls flax.sow
  and will store the recorded norms in the "residual_activations" collection.

  Args:
    module: When tagging activations within a flax module, pass a pointer to
      the module object itself. The resulting intermediates tree will resemble
      the flax subtree that this is tagged in.
    identity_path: The x part of a residual connection.
    other_path: The F(x) part.
    name: Used to further specify a named key in the sown path.
  """
  res_norm = jnp.linalg.norm(identity_path.reshape(-1))
  add_norm = jnp.linalg.norm(other_path.reshape(-1))
  module.sow(
      'residual_activations',
      name,
      jnp.array((res_norm, add_norm)),
      reduce_fn=lambda x, y: y)


def pmap_then_unreplicate(leaf_fn, axis_name='batch'):
  """Performs tree computation on device then maps back to host.

  This function is useful to avoid checking for use_pmap throughout the model
  debugger. In the pmap case we map replicated inputs to unreplicated outputs,
  grabbing the values on device 0.

  Args:
    leaf_fn: A function to compute on the leaves of the pytree.
    axis_name: The axis to pmap over.

  Returns:
    tree_fn: Applies the pmapped leaf_fn to a replicated pytree, then returns
      the values on device 0.
  """
  p_leaf_fn = jax.pmap(leaf_fn, axis_name=axis_name)
  def tree_fn(*args):
    vals = p_leaf_fn(*args)
    vals_on_host = jax.tree_map(lambda x: x[0], vals)
    return vals_on_host
  return tree_fn


def append_pytree_leaves(full_pytree, to_append):
  """Appends all leaves in the to_append pytree to the full_pytree.

  We assume full_pytree and to_append have the same structure. The leaves of
  full_pytree will have shape (num_previous_appends, *to_append_leaf_shape).
  For example if full_pytree = {'a': np.ones((2, 10))} and
  to_append = {'a': np.ones(10)}. Then append_pytree(full_pytree, to_append)
  returns {'a':, np.ones(3, 10)}. If full_pytree is None, then in the above
  example returns {'a': np.ones(1, 10)}.

  Args:
    full_pytree: pytree of all previously appended pytrees.
    to_append: pytree with same structure of leaves to be appended to
      full_pytree.

  Returns:
     A pytree where the leaves of to_append have been concatenate onto the
       leaves of full_pytree.
  """
  if not full_pytree:
    return jax.tree_map(lambda x: np.expand_dims(x, axis=0), to_append)

  return jax.tree_multimap(lambda x, y: array_append(y, x), to_append,
                           full_pytree)


def create_forward_pass_stats_fn(flax_module,
                                 capture_activation_norms=False,
                                 sown_collection_names=None):
  """Creates a function which grabs intermediate values from forward pass.

  If capture_activation_norms=True then we run the forward pass with
  capture_intermeidates=True and then return all activation norms collected.
  We will also collect and return any additional intermediate statistics
  that have been added via flax.sow. To ensure that these are run in the
  the forward pass, add the collection name to the sown_collection_names list.

  Args:
    flax_module: A flax module.
    capture_activation_norms: If true then all intermediate activation norms
      will be returned in the key 'intermediate_norms'.
    sown_collection_names: Any additional stats logged via flax.sow can be
      captured by providing a list of the collection names we wish to capture.

  Returns:
    get_forward_pass_statistics: A function mapping params, batch to a dict
      of interal statistics. The keys of the dictionary will be all keys in
      sown_collection_names. Additionally if capture_activation_norms=True then
      the key 'intermediate_norms' will be included.
  """

  # Note, we need to pass 'batch_stats' here in order to run BN in train mode.
  # We don't actually need the updated batch_stats though.
  mutables = ['intermediates', 'batch_stats']
  if sown_collection_names is not None:
    mutables.extend(sown_collection_names)
  def get_forward_pass_statistics(params, batch):
    _, forward_pass_statistics = flax_module.apply(
        {'params': params},
        batch,
        capture_intermediates=capture_activation_norms,
        mutable=mutables,
        train=True)
    forward_pass_statistics = flax.core.unfreeze(forward_pass_statistics)

    # We run in train mode but throw away the updated batch stats because we
    # do not want to log them.
    if 'batch_stats' in forward_pass_statistics:
      forward_pass_statistics.pop('batch_stats')
    if 'intermediates' in forward_pass_statistics:
      # This calculation corresponds to the average q-value across the batch.
      forward_pass_statistics['intermediate_norms'] = jax.tree_map(
          lambda x: jnp.linalg.norm(x.reshape(-1)) ** 2 / x.size,
          forward_pass_statistics['intermediates'])

      # Don't want to store the full activations.
      forward_pass_statistics.pop('intermediates')

    return forward_pass_statistics
  return get_forward_pass_statistics


def _check_tuple(x):
  return isinstance(x, tuple) and len(x) == 1


def _maybe_remove_tuple(x):
  if _check_tuple(x):
    return x[0]
  return x


def remove_leaf_tuples(tree):
  return jax.tree_map(_maybe_remove_tuple,
                      tree,
                      is_leaf=_check_tuple)


class ModelDebugger:
  """Debugging tool for internal layers of a model."""

  def __init__(self,
               forward_pass=None,
               use_pmap=True,
               save_every=1,
               metrics_logger=None):
    """Used to inspect a models forward and backward pass.

    The following keys are required in config -
      use_pmap: Whether or not to call jax.pmap on the forward pass.

    Args:
      forward_pass: A function mapping batch to a dict of intermediate values.
      use_pmap: Boolean which determines whether or not computations are meant
        to be pmapped. If true, then full_eval will expect all pytrees to be
        replicated.
      save_every: Stored metrics will be saved to disk every time
        step % save_every == 0
      metrics_logger: utils.MetricsLogger object. If provided then all
        calculations will be saved to disk.
    """
    if metrics_logger and (metrics_logger._json_path is None):
      raise ValueError('To use the ModelDebugger with a metrics_logger, a json'
                       ' path must be specified when building metrics_logger')
    self._save_every = save_every
    self._metrics_logger = metrics_logger
    self._use_pmap = use_pmap
    self.forward_pass = None

    # In both the pmap case and non-pmap case, _tree_norm_fn_sql2 returns
    # unreplicated results on the host cpu.
    if use_pmap:
      self._tree_norm_fn_sql2 = pmap_then_unreplicate(tree_norm_sql2)
      if forward_pass is not None:
        self.forward_pass = pmap_then_unreplicate(forward_pass)
    else:
      self._tree_norm_fn_sql2 = tree_norm_sql2
      self.forward_pass = forward_pass

    self._stored_metrics = {}

    # In the case of preemption we want to restore prior metrics.
    if metrics_logger:
      metrics_file = os.path.join(metrics_logger._pytree_path,
                                  'training_metrics')
      if gfile.exists(metrics_file):
        self._stored_metrics = load_pytree(metrics_file)

  def _grab_statistics(self,
                       step,
                       params=None,
                       grad=None,
                       update=None,
                       grad_norms_sql2=None,
                       update_norms_sql2=None,
                       param_norms_sql2=None):
    """Computes layerwise gradient and parameter norm statistics."""
    metrics_dict = {'step': step}

    def maybe_compute_and_add_sql2_to_metrics_dict(variable_tree,
                                                   norm_tree_sql2, key):
      if variable_tree and not norm_tree_sql2:
        norm_tree_sql2 = self._tree_norm_fn_sql2(variable_tree)
      if norm_tree_sql2:
        metrics_dict['{}_norms_sql2'.format(key)] = norm_tree_sql2
        metrics_dict['global_{}_norm_sql2'.format(key)] = sum(
            jax.tree_leaves(norm_tree_sql2))

    for tup in zip([params, grad, update],
                   [param_norms_sql2, grad_norms_sql2, update_norms_sql2],
                   ['param', 'grad', 'update']):
      maybe_compute_and_add_sql2_to_metrics_dict(*tup)

    return metrics_dict

  def _maybe_save_metrics(self, step):
    save_dict = self._stored_metrics.copy()
    if self._save_every:
      if step % self._save_every == 0:
        self._metrics_logger.write_pytree(save_dict)
    else:
      self._metrics_logger.write_pytree(save_dict)

  @property
  def stored_metrics(self):
    return self._stored_metrics

  def full_eval(self,
                step,
                params=None,
                grad=None,
                update=None,
                param_norms_sql2=None,
                grad_norms_sql2=None,
                update_norms_sql2=None,
                extra_scalar_metrics=None,
                batch=None):
    """Computes statistics of the forward and backward pass and save to disk.

    Currently what is written to disk is a pytree, with a dict at the top level
    with some subset of the following keys:

    step: A vector of shape (num_evals), indicating the steps at which full_eval
      was called.
    grad_norms_sql2: A pytree with same structure as grad, the leaves will have
      shape (num_evals), where num_evals is the number of times that this
      function is called. Each entry will be the
      squared l2 norm of the corresponding variable in grad. This will be
      computed if either the argument grad or grad_norms_sql2 is supplied.
      grad is assumed to be the gradient pytree, and the norms will be computed
      from it. Optionally, if the caller can already efficiently compute the
      tree norms, they can supply it directly with the grad_norms_sql2 arg.
    params_norms_sql2: Same as grad_norms_sql2 but for the param pytree
      instead.
    update_norms_sql2: Same as the grad_norms_sql2 but for the update pytree
      instead.
    **extra_scalar_kwargs: If extra_scalar_metrics is supplied, then we will
      save additional keys, one per key in extra_scalar_metrics. The shape of
      the value will be (num_evals).

    Args:
      step: Current global step in training.
      params: Optional pytree of the model parameters.
      grad: Optional pytree of the loss gradient.
      update: Optional pytree of the optimizer update.
      param_norms_sql2: Optional pytree of the square l2 param norms, this will
        be used instead of the params argument when logging the parameter norms.
      grad_norms_sql2 : Optional pytree of the square l2 grad norms.
      update_norms_sql2: Optional pytree of the square l2 param norms.
      extra_scalar_metrics: A dict of any addional metrics to log.
      batch: A batch of data to use in grabbing forward pass statistics. Only
        used when self.forward_pass is not None.

    Returns:
      Dictionary of all computed metrics. Note, if self.metrics_logger exists
        then we additionally write this dictionary to disk.
    """
    all_metrics = {'step': step}
    if any([
        params, grad, update, param_norms_sql2, grad_norms_sql2,
        update_norms_sql2
    ]):
      metrics_dict = self._grab_statistics(
          step,
          params=params,
          grad=grad,
          update=update,
          grad_norms_sql2=grad_norms_sql2,
          update_norms_sql2=update_norms_sql2,
          param_norms_sql2=param_norms_sql2)
      all_metrics.update(metrics_dict)

    if extra_scalar_metrics:
      all_metrics.update(extra_scalar_metrics)

    if self.forward_pass:
      if batch is None:
        raise ValueError(
            'Must supply a batch when computing forward pass stats.')
      forward_stats = self.forward_pass(params, batch)
      all_metrics.update(forward_stats)

    # When using metrics_logger.append_pytree(pytree, append=True), what is
    # saved is a list of pytrees with scalar leaves, one leaf per layer norm
    # that's computed. This was costly when computing this every step of
    # training. Instead what we do here is keep a single pytree with arrays as
    # leaves, where the length of each array is the number of evals. This is
    # a more compact representation, and also allows for decoupling of
    # save_frequency and eval_frequency to avoid writing to disk every step.
    # In order to properly do this, the model_debugger object needs to be
    # stateful and keep track of this tree in between saves.

    self._stored_metrics = append_pytree_leaves(
        self._stored_metrics,
        remove_leaf_tuples(flax.core.unfreeze(all_metrics)))

    if self._metrics_logger and jax.host_id() == 0:
      self._maybe_save_metrics(step)

    return all_metrics
