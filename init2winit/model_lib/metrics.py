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

"""Registry for the available metrics we can use for evaluating models.

Metric functions take a batch of (logits, targets, weights) as input and
return a batch of loss values. This is for safe aggregation across
different-sized eval batches.
"""

from clu import metrics
import flax
from init2winit.model_lib import losses
import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics


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


# Following the Flax OGB example:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/train.py
@flax.struct.dataclass
class MeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('logits', 'targets', 'weights'))):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    values = super().compute()
    targets = values['targets']
    logits = values['logits']
    weights = values['weights']
    if weights.shape != targets.shape:
      # This happens if weights are None
      if np.all(np.isnan(weights)):
        weights = None
      # We need weights to be the exact same shape as targets, not just
      # compatible for broadcasting, so multiply by ones of the right shape.
      weights = np.ones(targets.shape) * losses.conform_weights_to_targets(
          weights, targets)

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
      if np.sum(targets[:, task] == 0) > 0 and (
          np.sum(targets[:, task] == 1) > 0):
        is_labeled = weights[:, task]
        average_precisions[task] = sklearn.metrics.average_precision_score(
            targets[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)


# TODO(mbadura): Check if we can use metrics.Average with a mask
def weighted_average_metric(fun):
  """Returns a clu.Metric that uses `weights` to average the values.

  We can't use CLU `metrics.Average` directly, because it would ignore
  `weights`.

  Args:
    fun: function with the API fun(logits, targets, weights)

  Returns:
    clu.Metric that maintains a weighted average of the values.
  """

  @flax.struct.dataclass
  class _Metric(metrics.Metric):
    """Applies `fun` and computes the average."""
    total: np.float32
    weight: np.float32

    @classmethod
    def from_model_output(cls, logits, targets, weights, **_):
      total = fun(logits, targets, weights).sum()
      if weights is None:
        weight = targets.shape[0]
      else:
        weight = weights.sum()
      return cls(total=total, weight=weight)

    def merge(self, other):
      return type(self)(
          total=self.total + other.total, weight=self.weight + other.weight)

    def compute(self):
      return self.total / self.weight

  return _Metric


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


# All metrics used here must take three arguments named `logits`, `targets`,
# `weights`. We don't use CLU's `mask` argument, the metric gets
# that information from `weights`. The total weight for calculating the average
# is `weights.sum()`.
_METRICS = {
    'binary_autoencoder_metrics':
        metrics.Collection.create(
            sigmoid_binary_cross_entropy=weighted_average_metric(
                losses.unnormalized_sigmoid_binary_cross_entropy),
            sigmoid_mean_squared_error=weighted_average_metric(
                losses.unnormalized_sigmoid_mean_squared_error),
            num_examples=NumExamples),
    'classification_metrics':
        metrics.Collection.create(
            error_rate=weighted_average_metric(weighted_misclassifications),
            ce_loss=weighted_average_metric(
                losses.weighted_unnormalized_cross_entropy),
            num_examples=NumExamples),
    'binary_classification_metrics':
        metrics.Collection.create(
            ce_loss=weighted_average_metric(
                losses.unnormalized_sigmoid_binary_cross_entropy),
            num_examples=NumExamples,
            average_precision=MeanAveragePrecision)
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
  except KeyError as metric_not_found_error:
    raise ValueError('Unrecognized metrics bundle: {}'.format(
        metrics_name)) from metric_not_found_error
