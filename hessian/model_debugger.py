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

r"""Debugging tool for identifying problematic layers in a network.

"""

import functools

import flax
import flax.linen as nn
from init2winit.model_lib import partition_tree
from init2winit.utils import array_append
from init2winit.utils import tree_norm_sql2
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

exists = gfile.exists


def qvalue(array):
  return jnp.linalg.norm(array.reshape(-1))**2 / array.size


def cvalue(activations):
  """Return c-values for the activations."""
  activations = activations.astype(jnp.float32)
  # reshape to (batch_size, activation_dim)
  activations = activations.reshape((activations.shape[0], -1))
  norms = jnp.linalg.norm(activations, axis=1)
  xxt = activations.dot(activations.T)
  diag = jnp.diag(1.0 / norms)
  cvalues = jnp.dot(jnp.dot(diag, xxt), diag)
  # avg cosine_sim
  return jnp.mean(cvalues)


def tag_qcvalue(module, activations, name):
  qc_value = (qvalue(activations), cvalue(activations))
  module.sow('qcvalues', name, qc_value)


def tag_residual_activations(module,
                             identity_path,
                             other_path,
                             name='residual'):
  """Used in inspecting the forward pass and residual networks.

  Residual connections involve adding x + F(x) for some function F. This
  function is used to log both ||x||_2 and ||F(x)||_2. This calls flax.sow
  and will store the recorded norms in the "residual_activations" collection.

  Args:
    module: When tagging activations within a flax module, pass a pointer to the
      module object itself. The resulting intermediates tree will resemble the
      flax subtree that this is tagged in.
    identity_path: The x part of a residual connection.
    other_path: The F(x) part.
    name: Used to further specify a named key in the sown path.
  """
  res_values_q = qvalue(identity_path)
  add_values_q = qvalue(other_path)
  module.sow(
      'qvalues',
      name + 'q',  # avoid scope collision with the cvalue
      jnp.array((res_values_q, add_values_q)),
      reduce_fn=lambda x, y: y)

  res_values_c = cvalue(identity_path)
  add_values_c = cvalue(other_path)
  module.sow(
      'cvalues',
      name + 'c',
      jnp.array((res_values_c, add_values_c)),
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
    vals_on_host = jax.tree.map(lambda x: x[0], vals)
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
    return jax.tree.map(lambda x: np.expand_dims(x, axis=0), to_append)

  return jax.tree.map(lambda x, y: array_append(y, x), to_append, full_pytree)


def create_forward_pass_stats_fn(apply_fn,
                                 capture_activation_norms=False,
                                 sown_collection_names=None):
  """Creates a function which grabs intermediate values from forward pass.

  If capture_activation_norms=True then we run the forward pass with
  capture_intermeidates=True and then return all activation norms collected.
  We will also collect and return any additional intermediate statistics
  that have been added via flax.sow. To ensure that these are run in the
  the forward pass, add the collection name to the sown_collection_names list.

  Args:
    apply_fn: See model_lib/base_model.apply_on_batch.
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
  def get_forward_pass_statistics(params, batch, rng):
    _, forward_pass_statistics = apply_fn(
        params=params,
        batch_stats=None,  # We run in train mode so don't need batch_stats.
        batch=batch,
        rngs={'dropout': rng},
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
      forward_pass_statistics['intermediate_qvalue'] = jax.tree.map(
          qvalue, forward_pass_statistics['intermediates'])
      forward_pass_statistics['intermediate_cvalue'] = jax.tree.map(
          cvalue, forward_pass_statistics['intermediates'])

      # Don't want to store the full activations.
      forward_pass_statistics.pop('intermediates')

    return forward_pass_statistics
  return get_forward_pass_statistics


def _check_tuple(x):
  return isinstance(x, tuple)


def _maybe_remove_tuple(x):
  if _check_tuple(x):
    return x[0]
  return x


def remove_leaf_tuples(tree):
  return jax.tree.map(_maybe_remove_tuple, tree, is_leaf=_check_tuple)


#### Utilities for the skip anlysis
def unflatten(d):
  return flax.core.freeze(
      flax.traverse_util.unflatten_dict(
          {tuple(k.split('/')): v for k, v in d.items()}))


# Not a JAX Array Type - just an empty holder of static information:
# The boolean in this class is what is actually toggled when we skip a layers
# backward pass.
@flax.struct.dataclass
class _Meta:
  active: bool = flax.struct.field(pytree_node=False)


def skip():
  # pylint: disable=protected-access
  self = nn.module._context.module_stack[-1]
  # pylint: enable=protected-access
  if self.has_variable('moduleflags', 'flag'):
    flag = self.get_variable('moduleflags', 'flag')
    return not flag.active
  else:
    return False


# This function uses stop_gradient tricks to replace a layer's Jacobian with the
# identity matrix. Currently the function only works for layers where
# input_shape == output_shape, this can be generalized later.
def skip_bwd(fn):
  """Decorator for selectively turning off the backward pass."""

  @functools.wraps(fn)
  def shunt_backwards(self, x):
    if skip():
      # if shape changes we'd do:
      #   proj(x) + lax.stop_gradient(fn(self, x) - proj(x))
      return x + lax.stop_gradient(fn(self, x) - x)
    else:
      return fn(self, x)

  return shunt_backwards


def build_skip_flags(paths_to_skip):
  """Construct the dictionary to be passed to module.apply for skipping layers.

  Args:
    paths_to_skip: '/' separated strings indicating the exact module to skip.
      The path will be equivalent to what is given in the flattened pytree,
      where instead of tuples as the flattened keys, we have the string of keys
      joined via '/'.

  Returns:
    A dictionary which can be passed to the flax.module.apply to turn off the
      specified tagged layers.
  """
  flat = {}
  for p in paths_to_skip:
    prefix = f'{p}/' if p else ''
    flat[prefix + 'flag'] = _Meta(False)
  return flax.core.freeze({'moduleflags': unflatten(flat)})


# TOOD(gilmer): Currently the model debugger does not properly handle internal
# activation of recurrant models. We will only grab the first instance of
# a flax module's call in the case that a module is called multiple times with
# shared weights in a recurrent model.
class ModelDebugger:
  """Debugging tool for internal layers of a model."""

  def __init__(self,
               forward_pass=None,
               grad_fn=None,
               use_pmap=True,
               save_every=1,
               metrics_logger=None,
               skip_flags=None,
               skip_groups=None):
    """Used to inspect a models forward and backward pass.

    The following keys are required in config -
      use_pmap: Whether or not to call jax.pmap on the forward pass.

    Args:
      forward_pass: A function mapping batch to a dict of intermediate values.
      grad_fn: A function mapping batch and parameters to the loss gradient.
      use_pmap: Boolean which determines whether or not computations are meant
        to be pmapped. If true, then full_eval will expect all pytrees to be
        replicated.
      save_every: Stored metrics will be saved to disk every time step %
        save_every == 0
      metrics_logger: utils.MetricsLogger object. If provided then all
        calculations will be saved to disk.
      skip_flags: A list of strings of modules to selectively turn off when
        doing the skip analysis on the backward pass.
      skip_groups: A list of registered functions (defined in partition_tree.py)
        which map the model parameter tree to a subset of model layers. This can
        be used to turn off many layers jointly e.g. "how much do all of the
        attention layer jacobians contribute to the vanishing gradient problem"?
    """
    if metrics_logger and (metrics_logger._json_path is None):
      raise ValueError('To use the ModelDebugger with a metrics_logger, a json'
                       ' path must be specified when building metrics_logger')
    self._save_every = save_every
    self._metrics_logger = metrics_logger
    self._use_pmap = use_pmap
    self.forward_pass = None
    self.grad_fn = None
    self.skip_flags = [] if skip_flags is None else skip_flags
    self.skip_groups = [] if skip_groups is None else skip_groups

    # In both the pmap case and non-pmap case, _tree_norm_fn_sql2 returns
    # unreplicated results on the host cpu.
    if use_pmap:
      self._tree_norm_fn_sql2 = pmap_then_unreplicate(tree_norm_sql2)
      if forward_pass is not None:
        self.forward_pass = pmap_then_unreplicate(forward_pass)
      if grad_fn is not None:
        # Regular pmap here to comply with _grab_statistics
        self.grad_fn = jax.pmap(grad_fn, axis_name='batch')
    else:
      self._tree_norm_fn_sql2 = tree_norm_sql2
      self.forward_pass = forward_pass
      self.grad_fn = grad_fn

    self._stored_metrics = {}

    # In the case of preemption we want to restore prior metrics.
    if metrics_logger and metrics_logger.latest_pytree_checkpoint_step():
      self._stored_metrics = metrics_logger.load_latest_pytree()

  def _grab_statistics(self,
                       step,
                       batch=None,
                       rng=None,
                       params=None,
                       grad=None,
                       update=None,
                       grad_norms_sql2=None,
                       update_norms_sql2=None,
                       param_norms_sql2=None):
    """Computes layerwise gradient and parameter norm statistics."""
    metrics_dict = {'step': step}
    if grad is None and grad_norms_sql2 is None and self.grad_fn is not None:
      grad = self.grad_fn(params, batch, rng)

    for tup in zip([params, grad, update],
                   [param_norms_sql2, grad_norms_sql2, update_norms_sql2],
                   ['param', 'grad', 'update']):
      variable_tree, norm_tree_sql2, key = tup
      if variable_tree and not norm_tree_sql2:
        norm_tree_sql2 = self._tree_norm_fn_sql2(variable_tree)
      if norm_tree_sql2:
        metrics_dict['{}_norms_sql2'.format(key)] = norm_tree_sql2
        metrics_dict['global_{}_norm_sql2'.format(key)] = sum(
            jax.tree.leaves(norm_tree_sql2))

    return metrics_dict

  def _maybe_save_metrics(self, step):
    save_dict = self._stored_metrics.copy()
    if self._save_every:
      if step % self._save_every == 0:
        self._metrics_logger.write_pytree(save_dict, step=step)
    else:
      self._metrics_logger.write_pytree(save_dict, step=step)

  @property
  def stored_metrics(self):
    return self._stored_metrics

  def run_skip_analysis(self, params, batch, rng):
    """Runs a perturbative analysis of the model gradient.

    NOTE: Currently the skip analysis only works for layers which have the same
    input shape as output shape. Layers which change the shape will be handled
    in an upcoming CL.

    For each layer in config['skip_flags'] compute the backward pass with
    that layer swapped to the identity function. Useful for flagging model
    components which have large/vanishing jacobian singular values. Because
    the skip_analysis can be configured to run on separate steps we additionally
    supply the step here. To use the skip analysis do the following steps:

    1. For the module you would like to skip, tag the module's __call__ method
       with the @skip_bwd decorator.
    2. Set config['skip_flags'] = ['path/to/module/to/skip'], a list of strings
       indicating the sequence of layers to skip. The analysis will skip each
       layer individually and return the gradient norms resulting from that
       layer being skipped. When skipping multiple layers at once, use
       config['skip_groups'] which is a list of keys in a registry of functions
       defined in partition_tree.py. Each of these functions maps
       the model parameters to a list of keys. This is most useful when skipping
       many layers at once, for example skipping all of the post residual BN in
       a 200L resnet (200 BN's skipped at once).
    3. The output of the skip analysis is contained in results['skip_analysis'],
       which is a dictionary with keys =
       union(config['skip_flags', 'skip_groups']). Each key points to per layer
       gradient norms resulting from calling the backward pass with the target
       layers skipped.

    For a full example of how to use the skip analysis, see the unit test in
    test_model_debugger.py.

    Args:
      params: Pytree of model params.
      batch: Batch of data.
      rng: jax.random.PRNGKey

    Returns:
      A dictionary with a key for every layer specified in config['skip_flags']
      as well as every key in config['skip_groups'].
      Each key maps to the layerwise l2 gradient norms of the backward with
      that layer skipped (replaced with identity).
    """
    grad_dict = {}
    for flag in self.skip_flags:
      # Here we turn off each flag individually so we pass a list of len 1.
      flags = build_skip_flags([flag])
      new_g = self.grad_fn(params, batch, rng, module_flags=flags)
      grad_dict[flag] = self._tree_norm_fn_sql2(new_g)

    for flag_group in self.skip_groups:
      flags = partition_tree.get_skip_analysis_fn(flag_group)(params)
      flags = build_skip_flags(flags)
      new_g = self.grad_fn(params, batch, rng, module_flags=flags)
      grad_dict[flag_group] = self._tree_norm_fn_sql2(new_g)

    # Additionally store the regular gradient.
    new_g = self.grad_fn(params, batch, rng)
    grad_dict['unmodified_gradient'] = self._tree_norm_fn_sql2(new_g)
    return {'skip_analysis': grad_dict}

  def full_eval(self,
                step,
                params=None,
                grad=None,
                update=None,
                fwd_pass_summaries=None,
                param_norms_sql2=None,
                grad_norms_sql2=None,
                update_norms_sql2=None,
                extra_scalar_metrics=None,
                batch=None,
                rng=None):
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
      fwd_pass_summaries: Optional pytree of any statistics computed from the
        foward pass. Note, everything will be logged to disk, so be careful
        about tree with large leaves (e.g. if fwd_pass_summaries is the tree
        directly return from capture_intermediates=True, then we would save to
        disk the full activation tensors).
      param_norms_sql2: Optional pytree of the square l2 param norms, this will
        be used instead of the params argument when logging the parameter norms.
      grad_norms_sql2: Optional pytree of the square l2 grad norms.
      update_norms_sql2: Optional pytree of the square l2 param norms.
      extra_scalar_metrics: A dict of any addional metrics to log.
      batch: A batch of data to use in grabbing forward pass statistics. Only
        used when self.forward_pass is not None.
      rng: A jax.random.PRNGKey (replicated if use_pmap=True).

    Returns:
      Dictionary of all computed metrics. Note, if self.metrics_logger exists
        then we additionally write this dictionary to disk.
    """
    all_metrics = {'step': step}
    if any([
        params, grad, update, param_norms_sql2, grad_norms_sql2,
        update_norms_sql2, self.grad_fn
    ]):
      metrics_dict = self._grab_statistics(
          step=step,
          rng=rng,
          batch=batch,
          params=params,
          grad=grad,
          update=update,
          grad_norms_sql2=grad_norms_sql2,
          update_norms_sql2=update_norms_sql2,
          param_norms_sql2=param_norms_sql2)
      all_metrics.update(metrics_dict)

    if extra_scalar_metrics:
      all_metrics.update(extra_scalar_metrics)

    if self.skip_flags or self.skip_groups:
      all_metrics.update(self.run_skip_analysis(params, batch, rng))

    if self.forward_pass and not fwd_pass_summaries:
      if batch is None:
        raise ValueError(
            'Must supply a batch when computing forward pass stats.')

      fwd_pass_summaries = self.forward_pass(params, batch, rng)

    if fwd_pass_summaries:
      all_metrics.update(fwd_pass_summaries)

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

    if self._metrics_logger and jax.process_index() == 0:
      self._maybe_save_metrics(step)

    return all_metrics
