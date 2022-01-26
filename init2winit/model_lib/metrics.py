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

"""Registry for the available metrics we can use for evaluating models."""

from clu import metrics
import flax
from init2winit.model_lib import losses
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class NumExamples(metrics.Metric):
  """Computes the number of examples used for evaluation."""
  count: np.float32

  @classmethod
  def from_model_output(cls, logits, targets, weights, **_):
    if weights is None:
      count = targets.shape[0]
    else:
      count = weights.sum()
    return cls(count=count)

  def merge(self, other):
    return type(self)(count=self.count + other.count)

  def compute(self):
    return self.count


# TODO(gilmer): Consider revising this to support categorical targets.
def weighted_misclassifications(logits, targets, weights=None):
  """Compute weighted error rate over the given batch.

  This computes the error rate over a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples) it
  is assumed that weights is a binary mask where 0 indicates that the example
  is null. We assume the trainer will aggregate and divide by number of samples.

  Args:
   logits: [batch, ..., num_classes] float array.
   targets: one hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).

  Returns:
    Binary vector indicated which examples are misclassified ([batch, ...]).
  """
  if logits.ndim != targets.ndim:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  preds = jnp.argmax(logits, axis=-1)
  pred_targets = jnp.argmax(targets, axis=-1)
  loss = jnp.not_equal(preds, pred_targets)
  if weights is not None:
    if weights.ndim != targets.ndim - 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s one_hot_targets' %
          (str(weights.shape), str(targets.shape)))
    loss = loss * weights
  return loss


# All metrics created via `from_fun` must take three arguments named `logits`,
# `targets`, `weights`, because the arguments to these functions are passed via
# keyword args. We don't use CLU's `mask` argument, we use `weights` for that.
_METRICS = {
    'binary_autoencoder_metrics':
        metrics.Collection.create(
            sigmoid_binary_cross_entropy=metrics.Average.from_fun(
                losses.sigmoid_binary_cross_entropy),
            sigmoid_mean_squared_error=metrics.Average.from_fun(
                losses.sigmoid_mean_squared_error),
            num_examples=NumExamples),
    'classification_metrics':
        metrics.Collection.create(
            error_rate=metrics.Average.from_fun(weighted_misclassifications),
            ce_loss=metrics.Average.from_fun(
                losses.weighted_unnormalized_cross_entropy),
            num_examples=NumExamples),
    'binary_classification_metrics':
        metrics.Collection.create(
            ce_loss=metrics.Average.from_fun(
                losses.sigmoid_binary_cross_entropy),
            num_examples=NumExamples,
            average_precision=losses.MeanAveragePrecision)
}


def get_metrics(metrics_name):
  """Get the metric functions based on the metrics string.

  Args:
    metrics_name: (str) e.g. classification_metrics.

  Returns:
    A dictionary of metric functions.
  Raises:
    ValueError if the metrics is unrecognized.
  """
  try:
    return _METRICS[metrics_name]
  except KeyError:
    raise ValueError('Unrecognized metrics bundle: {}'.format(metrics_name))
