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

"""Registry for the available loss functions we can use for training models.

Loss functions take a batch of (logits, targets, weights) as input and
return the mean of function values. This is to make trainer.py more agnostic to
the details of the padding and masking.
"""

from clu import metrics
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sklearn.metrics


def unnormalized_sigmoid_binary_cross_entropy(logits, targets, weights=None):
  """Computes the sigmoid binary cross entropy per example.

  Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
    weights: None or float array of shape (batch,).

  Returns:
    Sigmoid binary cross entropy computed per example, shape (batch,).
  """
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  losses = -jnp.sum(
      (targets * log_p + (1 - targets) * log_not_p).reshape(
          targets.shape[0], -1),
      axis=-1)
  if weights is not None:
    if weights.shape[0] != losses.shape[0]:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s losses' %
          (str(weights.shape), str(losses.shape)))
    losses = losses * weights

  return losses


def sigmoid_binary_cross_entropy(logits, targets, weights=None):
  """Computes the sigmoid binary cross entropy between logits and targets.

  Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
    weights: None or float array of shape (batch,) or shape (batch,
      output_shape).

  Returns:
    float value of sigmoid binary cross entropy between logits and targets.
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


def unnormalized_sigmoid_mean_squared_error(logits, targets, weights=None):
  """Computes the sigmoid mean squared error per example.

  Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
    weights: None or float array of shape (batch,).

  Returns:
    Sigmoid mean squared error computed per example, shape (batch,).
  """
  losses = jnp.sum(
      jnp.square(nn.sigmoid(logits) - targets).reshape(targets.shape[0], -1),
      axis=-1)
  if weights is not None:
    if weights.shape[0] != losses.shape[0]:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s losses' %
          (str(weights.shape), str(losses.shape)))
    losses = losses * weights

  return losses


def sigmoid_mean_squared_error(logits, targets, weights=None):
  """Same as unnormalized_sigmoid_mean_squared_error, but takes the mean."""
  if weights is None:
    normalization = targets.shape[0]
  else:
    normalization = weights.sum()
  unnormalized_sigmoid_mse = unnormalized_sigmoid_mean_squared_error(
      logits, targets, weights)

  return jnp.sum(unnormalized_sigmoid_mse) / normalization


def weighted_unnormalized_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  This computes sum_(x,y) ce(x, y) for a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: one hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).

  Returns:
    Cross entropy loss computed per example, shape [batch, ...].
  """
  if logits.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s targets' %
        (str(logits.shape), str(targets.shape)))

  loss = -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  if weights is not None:
    if weights.ndim != targets.ndim - 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s targets' %
          (str(weights.shape), str(targets.shape)))
    loss = loss * weights

  return loss


def weighted_cross_entropy(logits, targets, weights=None):
  """Same as weighted_unnormalized, but additionally takes the mean."""
  if weights is None:
    normalization = targets.shape[0]
  else:
    normalization = weights.sum()
  unnormalized_cross_entropy = weighted_unnormalized_cross_entropy(
      logits, targets, weights)
  return jnp.sum(unnormalized_cross_entropy) / normalization


def ctc_loss(logits, logit_paddings, labels, label_paddings, blank_id=0):
  return optax.ctc_loss(logits, logit_paddings, labels, label_paddings,
                        blank_id)


# TODO(cheolmin): add mean_squared_error
_ALL_LOSS_FUNCTIONS = {
    'sigmoid_mean_squared_error': (sigmoid_mean_squared_error, jax.nn.sigmoid),
    'sigmoid_binary_cross_entropy':
        (sigmoid_binary_cross_entropy, jax.nn.sigmoid),
    'cross_entropy': (weighted_cross_entropy, jax.nn.softmax),
    'ctc': (ctc_loss, jax.nn.log_softmax)
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
  except KeyError as loss_fn_not_found_error:
    raise ValueError('Unrecognized loss function: {}'.format(
        loss_name)) from loss_fn_not_found_error


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
  except KeyError as activation_fn_not_found_error:
    raise ValueError('Unrecognized loss function: {}'.format(
        loss_name)) from activation_fn_not_found_error


# Following the Flax OGB example:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/train.py
@flax.struct.dataclass
class MeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('logits', 'targets', 'weights'))):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    targets = self.values['targets']
    logits = self.values['logits']
    weights = self.values['weights']

    assert logits.shape == targets.shape == weights.shape
    assert len(logits.shape) == 2
    assert np.logical_or(weights == 1, weights == 0).all()
    weights = weights.astype(np.bool)

    probs = jax.nn.sigmoid(logits)
    num_tasks = targets.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if np.sum(targets[:, task] == 0) > 0 and np.sum(targets[:,
                                                              task] == 1) > 0:
        is_labeled = weights[:, task]
        average_precisions[task] = sklearn.metrics.average_precision_score(
            targets[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)
