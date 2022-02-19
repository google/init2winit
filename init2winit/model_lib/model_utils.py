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

from typing import Any, Callable, Iterable

from absl import logging
from flax import linen as nn
from flax import optim
from init2winit.model_lib import normalization
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

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


class ScalarMultiply(nn.Module):
  """Layer which multiplies by a single scalar."""
  scale_init: Any = initializers.ones

  @nn.compact
  def __call__(self, x):
    return x * self.param('scale', self.scale_init, ())


def get_normalizer(normalizer,
                   train,
                   batch_size=None,
                   virtual_batch_size=None,
                   total_batch_size=None,
                   dtype=jnp.float32,
                   data_format='NHWC'):
  """Maps a string to the given normalizer function.

  We return a function that returns the normalization module, deferring the
  creation of the module to when the returned function is called. This is done
  because if we returned the module directly and then used it multiple times
  in the model (as is common in our codebase), the same module variables would
  be reused across call sites. This means that the common use case will be like:

    maybe_normalize = model_utils.get_normalizer(self.normalizer, train)
    x = maybe_normalize()(x)

  Relevant Flax Linen documentation for how to handle the train argument:
  https://flax.readthedocs.io/en/latest/design_notes/arguments.html. We take it
  as an input to get_normalizer because it is only used in the BatchNorm case
  (the alternative of only passing it as maybe_normalize(train) would result
  in an error in the LayerNorm case).

  Args:
    normalizer: One of ['batch_norm', 'virtual_batch_norm', 'layer_norm',
      'none'].
    train: Boolean indiciating if we are running in train or inference mode
      for batch norm.
    batch_size: only used for virtual batch norm, the batch size.
    virtual_batch_size: only used for virtual batch norm, the virtual batch
      size.
    total_batch_size: only used for virtual batch norm when using gradient
      accumulation, the total batch size used to calculate gradients with.
    dtype: data type used for normalizer inputs and outputs.
    data_format: only used for virtual batch norm, used to determine the batch
      axis.

  Returns:
    A function that when called will create the normalizer module/function.

  Raises:
    ValueError if normalizer not recognized.
  """
  if normalizer == 'batch_norm':
    return functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=dtype)
  elif normalizer == 'virtual_batch_norm':
    return functools.partial(
        normalization.VirtualBatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        data_format=data_format,
        total_batch_size=total_batch_size,
        dtype=dtype)
  elif normalizer in ['layer_norm', 'pre_layer_norm', 'post_layer_norm']:
    return functools.partial(nn.LayerNorm, dtype=dtype)
  elif normalizer == 'none':
    def identity_wrapper(*args, **kwargs):
      del args
      del kwargs
      def identity(x, *args, **kwargs):
        del args
        del kwargs
        return x
      return identity
    return identity_wrapper
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


def sync_batchnorm_stats(state):
  # TODO(jekbradbury): use different formula for running variances?
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


def rescale_layers(params, layer_rescale_factors):
  """Rescales the model variables by given multiplicative factors.

  Args:
    params: a dict of trainable model parameters.
    layer_rescale_factors: A dictionary mapping flat keys to a multiplicative
      rescale factor. The corresponding params in the module pytree will be
      changed from x -> a * x for rescale factor a. The keys of the dictionary
      must be of the form described in the flatten_keys documentation.

  Returns:
    A new flax module with the corresponding params rescaled.
  """
  all_keys = flatten_dict(params).keys()
  logging.info('All keys:')
  for key in all_keys:
    logging.info(key)

  for key in layer_rescale_factors:
    logging.info('Rescaling %s by factor %f', key, layer_rescale_factors[key])
    traversal = optim.ModelParamTraversal(lambda path, _: path == key)
    params = traversal.update(lambda x: x * layer_rescale_factors[key], params)
  return params
