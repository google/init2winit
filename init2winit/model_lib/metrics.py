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

"""Registry for the available metrics we can use for evaluating models."""
from init2winit.model_lib import losses
import jax.numpy as jnp


def num_examples(y_pred, y_true, weights=None):
  del y_pred
  if weights is None:
    return y_true.shape[0]
  return weights.sum()


# TODO(gilmer): Consider revising this to support categorical targets.
def weighted_misclassifications(logits, one_hot_targets, weights=None):
  """Compute weighted error rate over the given batch.

  This computes the error rate over a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples) it
  is assumed that weights is a binary mask where 0 indicates that the example
  is null. We assume the trainer will aggregate and divide by number of samples.

  Args:
   logits: [batch, ..., num_classes] float array.
   one_hot_targets: one hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).

  Returns:
    Binary vector indicated which examples are misclassified ([batch, ...]).
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))
  preds = jnp.argmax(logits, axis=-1)
  targets = jnp.argmax(one_hot_targets, axis=-1)
  loss = jnp.not_equal(preds, targets)
  if weights is not None:
    if weights.ndim != one_hot_targets.ndim - 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s one_hot_targets' %
          (str(weights.shape), str(one_hot_targets.shape)))
    loss = loss * weights
  return loss


_METRICS = {
    'binary_autoencoder_metrics': {
        'sigmoid_binary_cross_entropy': losses.sigmoid_binary_cross_entropy,
        'sigmoid_mean_squared_error': losses.sigmoid_mean_squared_error,
        'denominator': num_examples,
    },
    'classification_metrics': {
        'error_rate': weighted_misclassifications,
        'ce_loss': losses.weighted_unnormalized_cross_entropy,
        'denominator': num_examples,
    },
    'binary_classification_metrics': {
        'ce_loss': losses.sigmoid_binary_cross_entropy,
        'denominator': num_examples,
    },
    'regression_metrics': {
        'mse_loss': losses.mean_squared_error,
        'denominator': num_examples,
    }
}


def get_metrics(metrics):
  """Get the metric functions based on the metrics string.

  Args:
    metrics: (str) e.g. classification_metrics.

  Returns:
    A dictionary of metric functions.
  Raises:
    ValueError if the metrics is unrecognized.
  """
  try:
    return _METRICS[metrics]
  except KeyError:
    raise ValueError('Unrecognized metrics bundle: {}'.format(metrics))
