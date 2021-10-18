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

"""Common code used by different models."""

import functools

from absl import logging
from flax import nn
from flax import optim
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'id': lambda x: x,
    'tanh': jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
    'silu': jax.nn.silu,
    'leaky_relu': jax.nn.leaky_relu,
}

lecun_normal = functools.partial(
    initializers.variance_scaling,
    mode='fan_in',
    distribution='truncated_normal')

INITIALIZERS = {
    'delta_orthogonal': initializers.delta_orthogonal,
    'orthogonal': initializers.orthogonal,
    'lecun_normal': lecun_normal,
    'xavier_uniform': initializers.xavier_uniform,
}


class ScalarMultiply(nn.base.Module):
  """Layer which multiplies by a single scalar."""

  def apply(self, x, scale_init=initializers.ones):
    return x * self.param('scale', (), scale_init)


def get_normalizer(normalizer, train):
  """Maps a string to the given normalizer function.

  Args:
    normalizer: One of ['batch_norm', 'layer_norm', 'none'].
    train: Boolean indiciating if we are running in train or inference mode
      for batch norm.

  Returns:
    The normalizer function.

  Raises:
    ValueError if normalizer not recognized.
  """

  if normalizer == 'batch_norm':
    return nn.BatchNorm.partial(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5)
  elif normalizer in ['layer_norm', 'pre_layer_norm', 'post_layer_norm']:
    return nn.LayerNorm
  elif normalizer == 'none':
    def identity(x, name=None):
      del name
      return x
    return identity
  else:
    raise ValueError('Unknown normalizer: {}'.format(normalizer))


def apply_label_smoothing(one_hot_targets, label_smoothing):
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`. This weighted
  mixture of `one_hot_targets` with the uniform distribution is the same as is
  done in [1] and the original label smoothing paper [2]. Another way of
  performing label smoothing is to take `label_smoothing` mass from the
  on-values and distribute it to the off-values; in other words, transform the
  on-values to `1.0 - label_smoothing` and the off-values to
  `label_smoothing / (num_classes - 1)`. This was the style used in [3].
  In order to use this second style with this codebase, one can set the
  `label_smoothing` hyperparameter to the value from the Shallue, Lee et al.
  paper and set the hyperparameter `use_shallue_label_smoothing=True`.

  #### References
  [1]: Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton.
  "When does label smoothing help?." Advances in Neural Information Processing
  Systems. 2019.
  https://arxiv.org/abs/1906.02629
  [2]:  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and
  Zbigniew Wojna. "Rethinking the inception architecture for computer vision."
  In Proceedings of the IEEE conference on computer vision and pattern
  recognition, pages 2818–2826, 2016.
  https://arxiv.org/abs/1512.00567
  [3]: Shallue, Christopher J. and Lee, Jaehoon, et al.
  "Measuring the Effects of Data Parallelism on Neural Network Training."
  Journal of Machine Learning Research 20 (2019): 1-49.
  http://jmlr.org/papers/v20/18-789.html

  Args:
    one_hot_targets: one-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: a scalarin [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def _sync_local_batch_norm_stats_helper(x):
  """Average local BN EMAs along the 0-th axis, tile back to the same shape."""
  shape = x.shape
  # The state is replicated across devices on the first axis, so the
  # per-device EMAs are on the second.
  if shape[0] == 1:
    # In the case that the per_device_batch_size == virtual_batch_size.
    return x
  # Average across per-device EMAs.
  x = jnp.mean(x, axis=(0,))
  # Add back in the EMA axis.
  x = jnp.expand_dims(x, axis=0)
  tiling = [shape[0]] + [1] * (len(shape) - 1)
  # Recreate an EMA for each subbatch by tiling along the EMA axis.
  x = jnp.tile(x, tiling)
  return x


def sync_local_batch_norm_stats(batch_stats):
  """Sync the multiple local BN EMAs when using virual bs < per-device bs."""
  with batch_stats.mutate() as locally_synced_batch_stats:
    for state_values in locally_synced_batch_stats.as_dict().values():
      filtered_values = {
          k: v for k, v in state_values.items()
          if k.startswith('batch_norm_running_')
      }
      # This will overwrite values in state_values with those from the tree_map.
      state_values.update(
          jax.tree_map(_sync_local_batch_norm_stats_helper, filtered_values))
    return locally_synced_batch_stats


def sync_batchnorm_stats(state):
  # TODO(jekbradbury): use different formula for running variances?
  state = sync_local_batch_norm_stats(state)
  return lax.pmean(state, axis_name='batch')


def cross_device_avg(pytree):
  return jax.tree_map(lambda x: lax.pmean(x, 'batch'), pytree)


def l2_regularization(params, l2_decay_rank_threshold):
  """Computes the squared l2 norm of the given parameters.

  This function will only filter for parameters with
  rank >= l2_decay_rank_threshold. So if this threshold is set to 2, then all
  1d (and lower) parameter arrays, including all bias and batch norm params,
  will be ignored in this computation.


  Args:
    params: Pytree containing parameters.
    l2_decay_rank_threshold: The calculation will only include parameters with
       param.ndim >= l2_decay_rank_threshold. Set to 2 to ignore all bias and
       batch_norm params in the model.

  Returns:
    weight_l2: the squared l2 norm of all params matching the threshold.
  """
  weight_penalty_params = jax.tree_leaves(params)
  weight_l2 = sum([
      jnp.sum(x**2)
      for x in weight_penalty_params
      if x.ndim >= l2_decay_rank_threshold
  ])
  return weight_l2


def flatten_dict(nested_dict, sep='/'):
  """Flattens the nested dictionary.

  For example, if the dictionary is {'outer1': {'inner1': 1, 'inner2': 2}}.
  This will return {'/outer1/inner1': 1, '/outer1/inner2': 2}. With sep='/' this
  will match how flax optim.ParamTreeTraversal flattens the keys, to allow for
  easy filtering when using traversals. Requires the nested dictionaries
  contain no cycles.

  Args:
    nested_dict: A nested dictionary.
    sep: The separator to use when concatenating the dictionary strings.
  Returns:
    The flattened dictionary.
  """
  # Base case at a leaf, in which case nested_dict is not a dict.
  if not isinstance(nested_dict, dict):
    return {}
  return_dict = {}
  for key in nested_dict:
    flat_dict = flatten_dict(nested_dict[key], sep=sep)
    if flat_dict:
      for flat_key in flat_dict:
        return_dict[sep + key + flat_key] = flat_dict[flat_key]
    else:  # nested_dict[key] is a leaf.
      return_dict[sep + key] = nested_dict[key]
  return return_dict


def rescale_layers(flax_module, layer_rescale_factors):
  """Rescales the model variables by given multiplicative factors.

  Args:
    flax_module: A flax module where params is a nested dictionary.
    layer_rescale_factors: A dictionary mapping flat keys to a multiplicative
      rescale factor. The corresponding params in the module pytree will be
      changed from x -> a * x for rescale factor a. The keys of the dictionary
      must be of the form described in the flatten_keys documentation.

  Returns:
    A new flax module with the corresponding params rescaled.
  """
  all_keys = flatten_dict(flax_module.params).keys()
  logging.info('All keys:')
  for key in all_keys:
    logging.info(key)

  for key in layer_rescale_factors:
    if key not in all_keys:
      raise ValueError('Module does not have key: {}'.format(key))
    logging.info('Rescaling %s by factor %f', key, layer_rescale_factors[key])
    traversal = optim.ModelParamTraversal(lambda path, _: path == key)
    flax_module = traversal.update(lambda x: x * layer_rescale_factors[key],
                                   flax_module)
  return flax_module
