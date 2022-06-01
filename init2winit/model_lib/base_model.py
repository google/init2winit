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

"""Base class for all classification models."""

from init2winit.model_lib import losses
from init2winit.model_lib import metrics
from init2winit.model_lib import model_utils
from jax.nn import one_hot
import jax.numpy as jnp


def _evaluate_batch(flax_module, params, batch_stats, batch, metrics_bundle,
                    apply_one_hot_in_loss):
  """Evaluates metrics on the given batch.

  We use the CLU metrics library to evaluate the metrics, and we require that
  each metric_fn in metrics_bundle has the API:
    metric_fn(logits, targets, weights), including the argument names.

  Args:
    flax_module: the Flax linen.nn.Module.
    params: A dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: A dict of non-trainable model state. Passed as
      {'batch_stats': batch_stats} into flax_module.apply().
    batch: A dictionary with keys 'inputs', 'targets', 'weights'.
    metrics_bundle: A group of metrics to use for evaluation.
    apply_one_hot_in_loss: Indicates whether or not the targets are one hot
      encoded.

  Returns:
    A dictionary with the same keys as metrics, but mapping to the summed metric
    across the sharded batch_dim.
  """
  variables = {'params': params, 'batch_stats': batch_stats}
  logits = flax_module.apply(
      variables, batch['inputs'], mutable=False, train=False)
  targets = batch['targets']

  if apply_one_hot_in_loss:
    targets = one_hot(batch['targets'], logits.shape[-1])

  # map the dict values (which are functions) to function(targets, logits)
  weights = batch.get('weights')  # Weights might not be defined.
  eval_batch_size = targets.shape[0]
  if weights is None:
    weights = jnp.ones(eval_batch_size)

  # We don't use CLU's `mask` argument here, we handle it ourselves through
  # `weights`.
  return metrics_bundle.gather_from_model_output(
      logits=logits, targets=targets, weights=weights, axis_name='batch')


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

  flax_module is returned from the build_flax_module function. A typical
  usage pattern will be:
  ```
    model, hps = model_lib.get_model('fully_connected')
    ...  # possibly modify the default hps.
    model = model(hps, dataset.meta_data)
    model_init_fn = jax.jit(functools.partial(flax_module.init, train=False))
    init_dict = model_init_fn(
        {'params': params_rng, 'dropout': dropout_rng},
        fake_input_batch)
    # Trainable model parameters.
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    # this is how to call the model fprop.
    logits, vars = flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        batch,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng},
        train=True))
    new_batch_stats = vars['batch_stats']
  ```

  Note for models without batch norm, `flax_module.apply` will only return a
  single value (logits). See the Flax docs for more info:
  https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply.
  """

  def __init__(self, hps, dataset_meta_data, loss_name, metrics_name):
    self.hps = hps
    self.dataset_meta_data = dataset_meta_data
    self.loss_fn = losses.get_loss_fn(loss_name)
    self.output_activation_fn = losses.get_output_activation_fn(loss_name)
    self.metrics_bundle = metrics.get_metrics(metrics_name, hps)
    self.flax_module = self.build_flax_module()

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates metrics under self.metrics_name on the given batch."""
    return _evaluate_batch(
        self.flax_module,
        params,
        batch_stats,
        batch,
        self.metrics_bundle,
        self.dataset_meta_data['apply_one_hot_in_loss'])

  def apply_on_batch(self, params, batch_stats, batch, **apply_kwargs):
    """Wrapper around flax_module.apply.

    NOTE: We assume that batch_stats has been sync'ed.

    Args:
      params: A dict of trainable model parameters. Passed as {'params': params}
        into flax_module.apply().
      batch_stats: A dict of non-trainable model state. Passed as
        {'batch_stats': batch_stats} into flax_module.apply().
      batch: A dictionary with keys 'inputs', 'targets', 'weights'.
      **apply_kwargs: Any valid kwargs to flax_module.apply.

    Returns:
      An array of shape [batch_size, num_classes] that contains all the logits.
    """
    variables = {'params': params}
    if batch_stats is not None:
      variables['batch_stats'] = batch_stats
    return self.flax_module.apply(variables, batch['inputs'], **apply_kwargs)

  def predict_batch(self,
                    params,
                    batch_stats,
                    batch):
    """Returns predictions from all the model outputs on the given batch."""
    logits = self.apply_on_batch(params, batch_stats, batch, mutable=False)

    return self.output_activation_fn(logits)

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    """Return loss with an optional L2 penalty on the weights."""
    apply_kwargs = {'train': True}

    if batch_stats is not None:
      apply_kwargs['mutable'] = ['batch_stats']
    if dropout_rng is not None:
      apply_kwargs['rngs'] = {'dropout': dropout_rng}

    logits, new_batch_stats = self.apply_on_batch(params, batch_stats, batch,
                                                  **apply_kwargs)
    weights = batch.get('weights')
    return self.training_objective_fn(params, logits, batch['targets'],
                                      weights), new_batch_stats

  def training_objective_fn(self, params, logits, targets, weights):
    """Returns the training objective (loss + regularizer) on a batch of logits.

    Args:
      params: A dict of model parameters. Only used for regularization.
      logits: A jnp.array of shape (batch, output_shape).
      targets: A jnp.array of shape (batch, output_shape).
      weights: None or jnp.array of shape (batch,)

    Returns:
      A training objective value.

    """
    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = one_hot(targets, logits.shape[-1])
    # Optionally apply label smoothing.
    if self.hps.get('label_smoothing') is not None:
      targets = model_utils.apply_label_smoothing(
          targets, self.hps.get('label_smoothing'))

    objective_value = self.loss_fn(logits, targets, weights)

    if self.hps.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(params,
                                              self.hps.l2_decay_rank_threshold)
      objective_value += 0.5 * self.hps.l2_decay_factor * l2_loss

    return objective_value

  def get_fake_batch(self, hps):
    del hps
    return None

  def build_flax_module(self):
    """The flax module must accept a kwarg `train` in `__call__`."""
    raise NotImplementedError('Subclasses must implement build_flax_module().')
