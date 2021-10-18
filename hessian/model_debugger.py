# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

"""Debugging tool for identifying problematic layers in a network.

NOTE: The cmap and layer jacobian calculations only work for linen based models
and thus aren't compatible with i2w models at HEAD.
"""
import flax
from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_CONFIG = {
    'cmap': True,
    'use_pmap': True,
}

OUTPUT_PREFIX = 'fn_out'


def get_layer_idx(fn_out_key):
  """Maps 'fn_out{layer_idx}' -> layer_idx."""
  return int(fn_out_key[len(OUTPUT_PREFIX):])


def sort_keys(keys):
  return sorted(keys, key=lambda x: get_layer_idx(x[-1]))


def get_flat_output_dict(fn_out_dict):
  """Used in the cmap calculation to flatten the activation output tree."""
  flat_dict = flax.traverse_util.flatten_dict(unfreeze(fn_out_dict))
  sorted_keys = sort_keys(
      list(flat_dict.keys()))
  return flat_dict, sorted_keys


def get_cmat(m1):
  """Returns the matrix of cosine similarities of the provided data matrix."""
  m1 = m1.astype(np.float32)
  norms = np.array([np.linalg.norm(m1[i]) for i in range(len(m1))])
  xxt = m1.dot(m1.T)
  diag = np.diag(1.0 / norms)
  cmat = np.dot(np.dot(diag, xxt), diag)
  return cmat, norms


def batch_flatten(m, sharded=False):
  if sharded:
    batch_dim = m.shape[0] * m.shape[1]
  else:
    batch_dim = m.shape[0]
  return m.reshape(batch_dim, -1)


def tree_norm(pytree):
  return jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1)), pytree)


def get_tree_norm_fn(use_pmap, axis='batch'):
  if use_pmap:
    p_tree_norm = jax.pmap(tree_norm, axis_name=axis)
    def tree_norm_fn(pytree):
      norms = p_tree_norm(pytree)
      norms_on_device = jax.tree_map(lambda x: x[0], norms)
      return norms_on_device
    return tree_norm_fn
  else:
    return tree_norm


# TODO(gilmer): Add support for more general naming, addiotionally consider a
# global counter.
def tag_layer(self, x, layer_idx):
  self.sow(
      'fn_out',
      '{}{}'.format(OUTPUT_PREFIX, layer_idx),
      x,
      reduce_fn=lambda x, y: y)


class ModelDebugger:
  """Debugging tool for internal layers of a model."""

  def __init__(self, forward_pass, inputs, config, metrics_logger=None):
    """Used to inspect a models forward and backward pass.

    If config['cmap'] is set to True then we X X^T for specified internal
    activations of the model. Essentially, X X^T is the correlation matrix of
    the flattened internal representations, so the resulting shape is
    [n_examples, n_examples]. This is an empirical calculation of the cmap
    analysis done in https://arxiv.org/pdf/2110.01765.pdf. In extreme cases,
    the cmap can identify different failure modes of NN optimization (see page
    77 of Martens et. al.).

    Instructions for computing the cmap statistics:

    1. Inside the model, one can tag the activations of a layer by adding
      model_debugger.tag_layer(self, x, layer_idx) where layer_idx is a counter
      in [0, num_tagged_layers-1]. One must tag layers starting at index 0,
      skipping indices will result in an error.
    2. Set config['cmap'] = True.
    3. Provide a forward_pass function when constructing the model_debugger.
      This should always be the non pmapped function. The forward pass should
      satisfy the API forward_pass(params, False, inputs, {}). If
      config['use_pmap'] then we instead call
      jax.pmap(forward_pass)(params, False, inputs, {}) to compute the internal
      activations.

    The output of the cmap calc will be written to the metrics pytree under keys
    'c{}'.format(layer_idx). These can be loaded for analysis in colab.

    The following keys are required in config -
      use_pmap: Whether or not to call jax.pmap on the forward pass.
      cmap: Whether or not to perform the cmap calculation.

    Args:
      forward_pass: See above documation for the API this should satisfy.
      inputs: Single batch of inputs that will be used in the cmap calculation.
      config: See above for required keys.
      metrics_logger: utils.MetricsLogger object. If provided then all
        calculations will be saved to disk.
    """
    if metrics_logger and (metrics_logger._json_path is None):
      raise ValueError('To use the ModelDebugger with a metrics_logger, a json'
                       ' path must be specified when building metrics_logger')
    self.metrics_logger = metrics_logger
    self.config = config
    self.inputs = inputs
    self.forward_pass = forward_pass
    self.p_forward_pass = jax.pmap(
        forward_pass, axis_name='batch', static_broadcasted_argnums=1)
    self.tree_norm_fn = get_tree_norm_fn(config['use_pmap'])

  def grab_cmap_stats(self, params):
    """Iterates over all keys in mutated['fn_out'], grabbing their cmap."""
    metrics_dict = {}
    # TODO(gilmer): Keeping all registered activations in memory is prohibitive
    # for some models. As such, we can save memory by doing one forward pass per
    # tagged layer, at the expense of compute.
    if self.config['use_pmap']:
      _, mutated = self.p_forward_pass(
          params, False, self.inputs, {})
    else:
      _, mutated = self.forward_pass(
          params, False, self.inputs, {})

    flat_output_dict, flat_output_keys = get_flat_output_dict(mutated['fn_out'])
    for idx, key in enumerate(flat_output_keys):
      new_b = batch_flatten(
          flat_output_dict[key], sharded=self.config['use_pmap'])
      c2 = get_cmat(new_b)[0].reshape(-1)
      metrics_dict['c{}'.format(idx)] = c2
    return metrics_dict

  def grab_statistics(self,
                      step,
                      params=None,
                      grad=None,
                      update=None,
                      param_norms=None,
                      grad_norms=None,
                      update_norms=None,
                      ):
    """Computes layerwise gradient and parameter norm statistics."""
    metrics_dict = {'step': step}
    if grad or grad_norms:
      if grad_norms is None:
        grad_norms = self.tree_norm_fn(grad)
      metrics_dict['grad_norms'] = grad_norms
      metrics_dict['grad_norm'] = sum(jax.tree_leaves(grad_norms))
    if params or param_norms:
      if param_norms is None:
        param_norms = self.tree_norm_fn(params)
      metrics_dict['param_norms'] = param_norms
      metrics_dict['param_norm'] = sum(jax.tree_leaves(param_norms))
    if update:
      if update_norms is None:
        update_norms = self.tree_norm_fn(update_norms)
      metrics_dict['update_norms'] = update_norms
      metrics_dict['update_norm'] = sum(jax.tree_leaves(update_norms))

    return metrics_dict

  def full_eval(self,
                step,
                params,
                grad=None,
                update=None,
                param_norms=None,
                grad_norms=None,
                update_norms=None):
    """Computes statistics of the forward and backward pass.

    Args:
      step: Current global step in training.
      params: (Required), pytree of the model parameters.
      grad: Optional pytree of the loss gradient.
      update: Optional pytree of the optimizer update.
      param_norms: Optional pytree of the param_norms, this will be used instead
        of the params argument when logging the parameter norms.
      grad_norms : Optional pytree of the grad_norms, this will be used instead
        of the grad argument when logging the gradient norms.
      update_norms: Optional pytree of the update_norm, used instead
        of the update argument when logging the update norms.

    Returns:
      Dictionary of all computed metrics. Note, if self.metrics_logger exists
        then we additionally write this dictionary to disk.
    """
    all_metrics = {'step': step}
    if any([params, grad, update, param_norms, grad_norms, update_norms]):
      metrics_dict = self.grab_statistics(
          step,
          params=params,
          grad=grad,
          update=update,
          param_norms=param_norms,
          grad_norms=grad_norms,
          update_norms=update_norms,
      )
      all_metrics.update(metrics_dict)

    if self.config['cmap']:
      metrics_dict = self.grab_cmap_stats(params)
      all_metrics.update(metrics_dict)

    if self.metrics_logger:
      if jax.host_id() == 0:
        self.metrics_logger.append_pytree(all_metrics)
    return all_metrics
