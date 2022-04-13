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
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
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
class OGBGMeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(
        ('logits', 'targets', 'weights'))):
  """Computes the mean average precision (mAP) over different tasks on CPU.

  This implements the uncommon feature of allowing both per-example and
  per-class masking, which is required for the OGBG graph NN workload. For most
  use cases the BinaryMeanAveragePrecision metric should be used instead.
  """

  # Matches the official OGBG evaluation scheme for mean average precision.
  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
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

    if not (logits.shape == targets.shape == weights.shape):  # pylint: disable=superfluous-parens
      raise ValueError(
          f'Shape mismatch between logits ({logits.shape}), targets '
          '({targets.shape}), and weights ({weights.shape}).')
    if len(logits.shape) != 2:
      raise ValueError(f'Rank of logits ({logits.shape}) must be 2.')
    if not np.logical_or(weights == 1, weights == 0).all():
      raise ValueError(f'Weights must be {0, 1}, received {weights}.')

    weights = weights.astype(np.bool)

    probs = expit(logits)  # Sigmoid.
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


def _binary_auc_shape_fix_check(x, shape_error_msg):
  """Assert that the input shape is compatible, or fix it."""
  # One-hot targets, assumed to be shape [N, 2].
  if len(x.shape) == 2:
    if x.shape[1] > 2:
      raise ValueError(shape_error_msg)
    if x.shape[1] == 1:
      x = np.squeeze(x, axis=1)
    elif x.shape[1] == 2:
      # Binary AUC wants the labels/probabilities for the positive class, which
      # is the second element in the (n, 2) shaped array.
      x = x[:, 1]
  elif len(x.shape) > 2:
    raise ValueError(shape_error_msg)
  return x


def _binary_auc_shape_fix(targets, logits, weights, metric_name):
  """Ensure shapes are valid and convert them to dense shapes for sklearn.

  If inputs are shape (n, 2), we slice out the second column via x[:, 1]. If the
  inputs are shape (n, 1), we np.squeeze only the second dimension away. If the
  inputs are shape (n.), they are left untouched. If they are any other shape
  then a ValueError is raised.

  Args:
    targets: np.array of target labels, of shape (n,) or (n, 2).
    logits: np.array of model logits, of shape (n,) or (n, 2).
    weights: np.array of example weights, of shape (n,) or (n, 2).
    metric_name: the name of the metrics being checked, used for error messages.

  Returns:
    A triple of (targets, logits, weights) that now all have shape (n,).
  """
  shape_error_msg = (
      f'Inputs for {metric_name} should be of shape (n,) or (n, 2). Received '
      f'targets={targets.shape}, logits={logits.shape}, '
      f'weights={weights.shape}.')
  targets = _binary_auc_shape_fix_check(targets, shape_error_msg)
  logits = _binary_auc_shape_fix_check(logits, shape_error_msg)
  weights = _binary_auc_shape_fix_check(weights, shape_error_msg)
  # This happens if weights are None
  if np.all(np.isnan(weights)):
    weights = None
  # We need weights to be the exact same shape as targets, not just
  # compatible for broadcasting, so multiply by ones of the right shape.
  weights = np.ones(targets.shape) * losses.conform_weights_to_targets(
      weights, targets)
  return targets, logits, weights


@flax.struct.dataclass
class BinaryMeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(
        ('logits', 'targets', 'weights'))):
  """Computes the mean average precision for a binary classifier on CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(
        values['targets'],
        values['logits'],
        values['weights'],
        'BinaryMeanAveragePrecision')
    valid_targets = targets[weights > 0]
    targets_sum = np.sum(valid_targets)
    # Do not compute AUC if positives only have one class.
    if targets_sum == 0 or targets_sum == len(valid_targets):
      return 0.0
    probs = expit(logits)  # Sigmoid.
    return sklearn.metrics.average_precision_score(
        targets, probs, sample_weight=weights)


@flax.struct.dataclass
class BinaryAUCROC(
    metrics.CollectingMetric.from_outputs(
        ('targets', 'logits', 'weights'))):
  """Compute the AUC-ROC for binary classification on the CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(
        values['targets'],
        values['logits'],
        values['weights'],
        'BinaryAUCROC')
    valid_targets = targets[weights > 0]
    targets_sum = np.sum(valid_targets)
    # Do not compute AUC if all labels are the same.
    if targets_sum == 0 or targets_sum == len(valid_targets):
      return 0.0
    positive_probs = expit(logits)  # Sigmoid.
    return sklearn.metrics.roc_auc_score(
        targets, positive_probs, sample_weight=weights)


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
    'binary_classification_metrics_ogbg_map':
        metrics.Collection.create(
            ce_loss=weighted_average_metric(
                losses.unnormalized_sigmoid_binary_cross_entropy),
            num_examples=NumExamples,
            average_precision=OGBGMeanAveragePrecision),
    'binary_classification_metrics':
        metrics.Collection.create(
            ce_loss=weighted_average_metric(
                losses.unnormalized_sigmoid_binary_cross_entropy),
            num_examples=NumExamples,
            average_precision=BinaryMeanAveragePrecision,
            auc_roc=BinaryAUCROC),
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
