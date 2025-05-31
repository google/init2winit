# coding=utf-8
# Copyright 2025 The init2winit Authors.
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
import functools
from absl import logging
from clu import metrics
import flax
from init2winit import utils
from init2winit.model_lib import losses
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
import sklearn.metrics

# pylint: disable=g-import-not-at-top
try:
  from init2winit.dataset_lib import spm_tokenizer
except ImportError:
  logging.warning('SPM tokenizers could not be loaded.')
# pylint: enable=g-import-not-at-top


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
    metrics.CollectingMetric.from_outputs(('logits', 'targets', 'weights'))):
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

    if np.any(np.isnan(logits)):
      raise utils.TrainingDivergedError('NaN detected in logits')

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
          f'({targets.shape}), and weights ({weights.shape}).')
    if len(logits.shape) != 2:
      raise ValueError(f'Rank of logits ({logits.shape}) must be 2.')
    if not np.logical_or(weights == 1, weights == 0).all():
      raise ValueError(f'Weights must be {0, 1}, received {weights}.')

    weights = weights.astype(bool)

    probs = expit(logits)  # Sigmoid.
    num_tasks = targets.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if (np.sum(targets[:, task] == 0) > 0 and
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
    metrics.CollectingMetric.from_outputs(('logits', 'targets', 'weights'))):
  """Computes the mean average precision for a binary classifier on CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(
        values['targets'], values['logits'], values['weights'],
        'BinaryMeanAveragePrecision')

    if np.any(np.isnan(logits)):
      raise utils.TrainingDivergedError('NaN detected in logits')

    # TODO(mbadura): The np.array call should not be needed after numpy upgrade.
    valid_targets = targets[np.array(weights > 0)]
    targets_sum = np.sum(valid_targets)
    # Do not compute AUC if positives only have one class.
    if targets_sum == 0 or targets_sum == len(valid_targets):
      return 0.0
    probs = expit(logits)  # Sigmoid.
    return sklearn.metrics.average_precision_score(
        targets, probs, sample_weight=weights)


@flax.struct.dataclass
class BinaryAUCROC(
    metrics.CollectingMetric.from_outputs(('targets', 'logits', 'weights'))):
  """Compute the AUC-ROC for binary classification on the CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(values['targets'],
                                                     values['logits'],
                                                     values['weights'],
                                                     'BinaryAUCROC')
    # TODO(mbadura): The np.array call should not be needed after numpy upgrade.
    valid_targets = targets[np.array(weights > 0)]
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
  arg_names = fun.__code__.co_varnames

  @flax.struct.dataclass
  class _Metric(metrics.Metric):
    """Applies `fun` and computes the average."""
    total: np.float32
    weight: np.float32

    @classmethod
    def from_model_output(cls, logits, targets, weights, **kwargs):
      valid_kwargs = {
          key: val for key, val in kwargs.items() if key in arg_names
      }
      total = fun(logits, targets, weights, **valid_kwargs).sum()
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


def perplexity_fun():
  """Returns function for calculating perplexity where PPL(x) = e^(CE(x)).

  This is done exponentiating the CE calculated over potentially multiple
  batches.

  Returns:
    Perplexity subclass of _Metric:
      logits: [batch, length, num_classes] float array.
      targets: one hot vector of shape [batch, ..., num_classes].
      weights: None or array of shape [batch x ...] (rank of one_hot_targets
      -1).
      computed per example, shape [batch, ...].
  """
  _ParentMetricClass = weighted_average_metric(
      losses.weighted_unnormalized_cross_entropy)

  @flax.struct.dataclass
  class _PerplexityMetricClass(_ParentMetricClass):

    def compute(self):
      return jnp.exp(self.total / self.weight)

  return _PerplexityMetricClass


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


def uniform_filter(im, size=7):  # pylint: disable=missing-docstring

  def conv(im):
    return jnp.convolve(
        jnp.pad(im, pad_width=size // 2, mode='symmetric'),
        jnp.ones(size),
        mode='valid') / size
  im = jax.vmap(conv, (0,))(im)
  im = jax.vmap(conv, (1,))(im)
  return im.T


def structural_similarity(im1,
                          im2,
                          data_range=1.0,
                          win_size=7,
                          k1=0.01,
                          k2=0.03):
  """Compute the mean structural similarity index between two images.

  NOTE(dsuo): modified from skimage.metrics.structural_similarity.

  Args:
    im1: ndarray Images. Any dimensionality with same shape.
    im2: ndarray Images. Any dimensionality with same shape.
    data_range: float. The data range of the input image (distance between
      minimum and maximum possible values). By default, this is
    win_size: int or None. The side-length of the sliding window used in
      comparison. Must be an odd value. If `gaussian_weights` is True, this is
      ignored and the window size will depend on `sigma`. estimated from the
      image data-type.
    k1: float. Algorithm parameter K1 (see [1]).
    k2: float. Algorithm parameter K2 (see [2]).

  Returns:
    mssim: float
        The mean structural similarity index over the image.

  References
    [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
      (2004). Image quality assessment: From error visibility to
      structural similarity. IEEE Transactions on Image Processing,
      13, 600-612.
      https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
      :DOI:`10.1109/TIP.2003.819861`
  """
  filter_func = functools.partial(uniform_filter, size=win_size)

  num_points = win_size**len(im1.shape)

  # filter has already normalized by num_points
  cov_norm = num_points / (num_points - 1)  # sample covariance

  # compute (weighted) means
  ux = filter_func(im1)
  uy = filter_func(im2)

  # compute (weighted) variances and covariances
  uxx = filter_func(im1 * im1)
  uyy = filter_func(im2 * im2)
  uxy = filter_func(im1 * im2)
  vx = cov_norm * (uxx - ux * ux)
  vy = cov_norm * (uyy - uy * uy)
  vxy = cov_norm * (uxy - ux * uy)

  c1 = (k1 * data_range)**2
  c2 = (k2 * data_range)**2

  a1 = 2 * ux * uy + c1
  a2 = 2 * vxy + c2
  b1 = ux**2 + uy**2 + c1
  b2 = vx + vy + c2

  d = b1 * b2
  s = (a1 * a2) / d

  # to avoid edge effects will ignore filter radius strip around edges
  pad = (win_size - 1) // 2

  # compute (weighted) mean of ssim.
  return jnp.mean(s.at[pad:-pad, pad:-pad].get())


def ssim(logits, targets, weights=None, mean=None, std=None, volume_max=None):
  """Computes example-wise structural similarity for a batch.

  NOTE(dsuo): we use the same (default) arguments to `structural_similarity`
  as in https://arxiv.org/abs/1811.08839.

  Args:
   logits: (batch,) + input.shape float array.
   targets: (batch,) + input.shape float array.
   weights: None or array of shape (batch,).
   mean: (batch,) mean of original images.
   std: (batch,) std of original images.
   volume_max: (batch,) of the volume max for the volumes each example came
     from.

  Returns:
    Structural similarity computed per example, shape [batch, ...].
  """
  if volume_max is None:
    volume_max = jnp.ones(logits.shape[0])

  if mean is None:
    mean = jnp.zeros(logits.shape[0])

  if std is None:
    std = jnp.ones(logits.shape[0])

  mean = mean.reshape((-1,) + (1,) * (len(logits.shape) - 1))
  std = std.reshape((-1,) + (1,) * (len(logits.shape) - 1))
  logits = logits * std + mean
  targets = targets * std + mean
  ssims = jax.vmap(structural_similarity)(logits, targets, volume_max)

  # NOTE(dsuo): map NaN ssims to -1.
  ssims = jnp.where(jnp.isnan(ssims), -1, ssims)

  # NOTE(kasimbeg): map out-of-bounds ssims to 1 and -1, the theoretical
  # maximum and minimum values of SSIM.
  ssims = jnp.where(ssims > 1, jnp.ones_like(ssims), ssims)
  ssims = jnp.where(ssims < -1, jnp.ones_like(ssims) * -1, ssims)

  # This step should come after setting NaNs to -1. Otherwise,
  # padding elements with weight value 0 will be set to NaN * 0 = NaN
  # and then to -1 instead of 0.
  if weights is not None:
    ssims = ssims * weights

  return ssims


def average_ctc_loss():
  """Returns a clu.Metric that computes average CTC loss taking padding into account."""

  @flax.struct.dataclass
  class _Metric(metrics.Metric):
    """Applies `fun` and computes the average."""
    total: np.float32
    weight: np.float32

    @classmethod
    def from_model_output(cls, normalized_loss, **_):
      return cls(total=normalized_loss, weight=1.0)

    def merge(self, other):
      return type(self)(
          total=self.total + other.total, weight=self.weight + other.weight)

    def compute(self):
      return self.total / self.weight

  return _Metric


def compute_wer(decoded,
                decoded_paddings,
                targets,
                target_paddings,
                tokenizer,
                tokenizer_type='SPM'):
  """Computes word error rate."""
  word_errors = 0.0
  num_words = 0.0

  if tokenizer_type == 'SPM':
    decoded_lengths = np.sum(decoded_paddings == 0.0, axis=-1)
    target_lengths = np.sum(target_paddings == 0.0, axis=-1)

    batch_size = targets.shape[0]

    for i in range(batch_size):
      decoded_length = decoded_lengths[i]
      target_length = target_lengths[i]

      decoded_i = decoded[i][:decoded_length]
      target_i = targets[i][:target_length]

      decoded_i = str(tokenizer.detokenize(decoded_i.astype(np.int32)))
      target_i = str(tokenizer.detokenize(target_i.astype(np.int32)))

      target_i = ' '.join(target_i.split())
      target_num_words = len(target_i.split(' '))

      word_errors += utils.edit_distance(decoded_i, target_i)
      num_words += target_num_words
  else:
    raise ValueError('unsupported tokenizer type passed to compute WER.')

  return word_errors, num_words


def wer(hps):
  """Returns a clu.Metric that computes word error rate.

  Args:
    hps: hyperparameters providing vocab path for tokenizer.

  Returns:
    clu.Metric computing word error rate.
  """

  if hps.tokenizer_type == 'SPM':
    tokenizer = spm_tokenizer.load_tokenizer(hps.tokenizer_vocab_path)
  else:
    raise ValueError(
        'WER computation is currently only supported for SPM tokenizer.')

  @flax.struct.dataclass
  class WER(
      metrics.CollectingMetric.from_outputs(
          ('hyps', 'hyp_paddings', 'targets', 'target_paddings'))):
    """Metric computing word error rate."""

    def compute(self):
      values = super().compute()
      # Ensure the arrays are numpy and not jax.numpy.
      values = {k: np.array(v) for k, v in values.items()}

      word_errors, num_words = compute_wer(values['hyps'],
                                           values['hyp_paddings'],
                                           values['targets'],
                                           values['target_paddings'],
                                           tokenizer, hps.tokenizer_type)
      return word_errors / num_words

  return WER


# All metrics used here must take three arguments named `logits`, `targets`,
# `weights`. We don't use CLU's `mask` argument, the metric gets
# that information from `weights`. The total weight for calculating the average
# is `weights.sum()`.
_METRICS = {
    'autoregressive_language_model_metrics':
        metrics.Collection.create(
            error_rate=weighted_average_metric(weighted_misclassifications),
            ce_loss=weighted_average_metric(
                losses.weighted_unnormalized_cross_entropy),
            perplexity=perplexity_fun(),
            num_examples=NumExamples),
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
    'binary_classification_metrics_dlrm_no_auc':
        metrics.Collection.create(
            ce_loss=weighted_average_metric(
                losses.unnormalized_sigmoid_binary_cross_entropy),
            num_examples=NumExamples,
        ),
    'image_reconstruction_metrics':
        metrics.Collection.create(
            l1_loss=weighted_average_metric(
                losses.weighted_unnormalized_mean_absolute_error),
            ssim=weighted_average_metric(ssim),
            num_examples=NumExamples,
        ),
}


def get_metrics(metrics_name, hps=None):
  """Get the metric functions based on the metrics string.

  Args:
    metrics_name: (str) e.g. classification_metrics.
    hps: optional hyperparameters to be used while creating metric classes.

  Returns:
    A dictionary of metric functions.
  Raises:
    ValueError if the metrics is unrecognized.
  """
  if metrics_name == 'ctc_metrics':
    return metrics.Collection.create(ctc_loss=average_ctc_loss(), wer=wer(hps))
  try:
    return _METRICS[metrics_name]
  except KeyError as metric_not_found_error:
    raise ValueError('Unrecognized metrics bundle: {}'.format(
        metrics_name)) from metric_not_found_error
