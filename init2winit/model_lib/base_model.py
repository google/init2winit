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

# Lint as: python3
"""Base class for all classification models."""

from flax import nn
from init2winit.model_lib import losses
from init2winit.model_lib import metrics
from init2winit.model_lib import model_utils
from jax import lax
from jax.nn import one_hot
import jax.numpy as jnp


def _evaluate_batch(flax_module, batch_stats, batch, metrics_bundle,
                    apply_one_hot_in_loss):
  """Evaluates metrics on the given batch.

  Currently we assume each metric_fn in metrics_bundle has the API:
    metric_fn(logits, targets, weights)
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen (calculated by the 'denominator' metric). In this way we
  currently only support metrics of the 1/N sum f(inputs, targets). Note, the
  caller is responsible for dividing by metrics['denominator'] when computing
  the mean of each metric.

  Args:
    flax_module: A flax.nn.Module
    batch_stats: A flax.nn.Collection object tracking batch_stats.
    batch: A dictionary with keys 'inputs', 'targets', 'weights'.
    metrics_bundle: A group of metrics to use for evaluation.
    apply_one_hot_in_loss: Indicates whether or not the targets are one hot
      encoded.

  Returns:
    A dictionary with the same keys as metrics, but mapping to the summed metric
    across the sharded batch_dim.

  """
  with nn.stateful(batch_stats, mutable=False):
    logits = flax_module(batch['inputs'], train=False)
  targets = batch['targets']

  if apply_one_hot_in_loss:
    targets = one_hot(batch['targets'], logits.shape[-1])

  # map the dict values (which are functions) to function(targets, logits)
  weights = batch.get('weights')  # Weights might not be defined.
  eval_batch_size = targets.shape[0]
  if weights is None:
    weights = jnp.ones(eval_batch_size)

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is the we have summed across all samples in the
  # sharded batch.

  evaluated_metrics = {}
  for key in metrics_bundle:
    per_example_metrics = metrics_bundle[key](logits, targets, weights)
    evaluated_metrics[key] = jnp.sum(
        lax.psum(per_example_metrics, axis_name='batch'))

  return evaluated_metrics


def _predict_batch(flax_module,
                   batch_stats,
                   batch,
                   output_activation_fn=None):
  """Compute predictions for a batch of data.

  NOTE: We assume that batch_stats has been sync'ed.

  Args:
    flax_module: A flax.nn.Module
    batch_stats: A flax.nn.Collection object tracking batch_stats.
    batch: A dictionary with keys 'inputs', 'targets', 'weights'.
    output_activation_fn: An output activation function from jax.nn.functions

  Returns:
    An array of shape [batch_size, num_classes] that contains all the logits.
  """
  with nn.stateful(batch_stats, mutable=False):
    logits = flax_module(batch['inputs'], train=False)
  if output_activation_fn:
    return output_activation_fn(logits)
  return logits


class BaseModel(object):
  """Defines commonalities between all models.

  A model is class with five members variables and five member functions.

  Member variables: hps, dataset_meta_data, loss_fn, output_activation_fn,
  metrics_bundle

  hps is a set of hyperpameters in this experiment.
  dataset_meta_data is a set of data that describes the dataset.
  loss_fn is a loss function.
  output_activation_fn is an activation function at the output layer.
  metrics_bundle is a dict mapping metric names to corresponding functions.

  Member functions: __init__, evaluate_batch, predict_batch, training_cost,
  build_flax_module

  __init__ takes hps, dataset_meta_data, loss_name, metrics_name as arguments
  and set them to corresponding member variables. This function uses
  losses.get_loss_fn, losses.output_activation_fn and metrics.get_metrics
  to obtain loss_fn, output_activation_fn, and metrics_bundle
  from loss_name and metrics_name as:

    self.loss_fn = losses.get_loss_fn(loss_name)
    self.output_activation_fn = losses.get_output_activation_fn(loss_name)
    self.metrics_bundle = metrics.get_metrics(metrics_name)

  evaluate_batch compute metrics in self.metrics_bundle for a given batch.
  A dictionary having summed metrics is returned.

  predict_batch computes the logits of a given batch using a given flax_module.
  This function is primarily used in inference_lib.

  training_cost defines a loss with weight decay, where the
  weight decay factor is determined by hps.l2_decay_factor.

  flax_module_def is returned from the build_flax_module function. A typical
  usage pattern will be:

    model, hps = model_lib.get_model('fully_connected')
    ...  # possibly modify the default hps.
    model = model(hps, dataset.meta_data)
    with nn.stateful(batch_stats) as new_batch_stats:
      _, flax_module = model.flax_module_def.create(
          params_rng, inputs, batch_stats=new_batch_stats)

    logits = flax_module(inputs)  # this is how to call the model fprop.
  """

  def __init__(self, hps, dataset_meta_data, loss_name, metrics_name):
    self.hps = hps
    self.dataset_meta_data = dataset_meta_data
    self.loss_fn = losses.get_loss_fn(loss_name)
    self.output_activation_fn = losses.get_output_activation_fn(loss_name)
    self.metrics_bundle = metrics.get_metrics(metrics_name)
    self.flax_module_def = self.build_flax_module()

  def evaluate_batch(self, flax_module, batch_stats, batch):
    """Evaluates metrics under self.metrics_name on the given batch."""
    return _evaluate_batch(flax_module, batch_stats, batch, self.metrics_bundle,
                           self.dataset_meta_data['apply_one_hot_in_loss'])

  def predict_batch(self,
                    flax_module,
                    batch_stats,
                    batch,
                    apply_output_activation_fn=False):
    """Returns predictions from all the model outputs on the given batch."""
    if apply_output_activation_fn:
      return _predict_batch(flax_module, batch_stats, batch,
                            self.output_activation_fn)
    else:
      return _predict_batch(flax_module, batch_stats, batch)

  def training_cost(self, flax_module, batch_stats, batch, dropout_rng):
    """Return loss with an L2 penalty on the weights."""
    with nn.stateful(batch_stats) as new_batch_stats:
      with nn.stochastic(dropout_rng):
        logits = flax_module(batch['inputs'], train=True)
    weights = batch.get('weights')
    targets = batch['targets']
    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = one_hot(targets, logits.shape[-1])
    # Optionally apply label smoothing.
    if self.hps.get('label_smoothing') is not None:
      targets = model_utils.apply_label_smoothing(
          targets, self.hps.get('label_smoothing'))
    total_loss = self.loss_fn(logits, targets, weights)

    if self.hps.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(flax_module.params,
                                              self.hps.l2_decay_rank_threshold)
      total_loss += 0.5 * self.hps.l2_decay_factor * l2_loss

    return total_loss, (new_batch_stats)

  def get_fake_batch(self, hps):
    del hps
    return None

  def build_flax_module(self):
    raise NotImplementedError('Subclasses must implement build_flax_module().')
