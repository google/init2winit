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

"""Registry for the available loss functions we can use for training models."""
from flax import linen as nn
import jax
import jax.numpy as jnp


def sigmoid_binary_cross_entropy(logits, targets, weights=None):
  """Computes the sigmoid binary cross entropy between logits and targets.

  Args:
    logits: float array of shape (batch, output_shape)
    targets: float array of shape (batch, output_shape)
    weights: None or float array of shape (batch,) or shape (batch,
      output_shape)

  Returns:
    float value of sigmoid binary cross entropy between logits and targets
  """
  # Ensure logits and targets are 2d, even if there's only one label per example
  if len(logits.shape) == 1 or len(targets.shape) == 1:
    raise ValueError('logits and targets must be 2d')

  if weights is None:
    weights = jnp.ones_like(targets)
  elif weights.shape == targets.shape[:1]:
    # Add a dimension if labels are per-example so that multiplication can be
    # broadcasted.
    weights = weights[:, None]

  per_label_normalization = jnp.nan_to_num(1 / weights.sum(axis=0), nan=1.0)
  weights = weights * per_label_normalization

  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)

  return -jnp.sum((targets * log_p + (1 - targets) * log_not_p) * weights)


def sigmoid_mean_squared_error(logits, targets, weights=None):
  """Computes the sigmoid mean squared error between logits and targets.

  Args:
    logits: float array of shape (batch, output_shape)
    targets: float array of shape (batch, output_shape)
    weights: None or float array of shape (batch,)

  Returns:
    float value of sigmoid mean squared error between logits and targets
  """
  loss = jnp.sum(
      jnp.square(nn.sigmoid(logits) - targets).reshape(targets.shape[0], -1),
      axis=-1)
  if weights is None:
    weights = jnp.ones(loss.shape[0])
  weights = weights / sum(weights)
  return jnp.sum(jnp.dot(loss, weights))


def weighted_unnormalized_cross_entropy(logits, one_hot_targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  This computes sum_(x,y) ce(x, y) for a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
   logits: [batch, length, num_classes] float array.
   one_hot_targets: one hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).

  Returns:
    Cross entropy loss computed per example, shape [batch, ...].
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))

  loss = -jnp.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)
  if weights is not None:
    if weights.ndim != one_hot_targets.ndim - 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s one_hot_targets' %
          (str(weights.shape), str(one_hot_targets.shape)))
    loss = loss * weights

  return loss


def weighted_cross_entropy(logits, one_hot_targets, weights=None):
  """Same as weighted_unnormalized, but additionally takes the mean."""
  if weights is None:
    normalization = one_hot_targets.shape[0]
  else:
    normalization = weights.sum()
  unnormalized_cross_entropy = weighted_unnormalized_cross_entropy(
      logits, one_hot_targets, weights)
  return jnp.sum(unnormalized_cross_entropy) / normalization


# TODO(cheolmin): add mean_squared_error
_ALL_LOSS_FUNCTIONS = {
    'sigmoid_mean_squared_error': (sigmoid_mean_squared_error, jax.nn.sigmoid),
    'sigmoid_binary_cross_entropy':
        (sigmoid_binary_cross_entropy, jax.nn.sigmoid),
    'cross_entropy': (weighted_cross_entropy, jax.nn.softmax),
}


def get_loss_fn(loss_name):
  """Get the corresponding loss function based on the loss_name.

  Args:
    loss_name: (str) e.g. cross_entropy.

  Returns:
    The loss function.
  Raises:
    ValueError if loss is unrecognized.
  """
  try:
    return _ALL_LOSS_FUNCTIONS[loss_name][0]
  except KeyError:
    raise ValueError('Unrecognized loss function: {}'.format(loss_name))


def get_output_activation_fn(loss_name):
  """Get the corresponding output activation function based on the loss_name.

  Args:
    loss_name: (str) e.g. cross_entropy.

  Returns:
    The output activation function.
  Raises:
    ValueError if loss is unrecognized.
  """
  try:
    return _ALL_LOSS_FUNCTIONS[loss_name][1]
  except KeyError:
    raise ValueError('Unrecognized loss function: {}'.format(loss_name))
