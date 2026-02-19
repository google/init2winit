# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

import copy
import functools
import time

from absl import logging
from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import losses
from init2winit.model_lib import metrics
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
import numpy as np

ParameterType = model_utils.ParameterType
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


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
    targets = jax.nn.one_hot(batch['targets'], logits.shape[-1])

  # map the dict values (which are functions) to function(targets, logits)
  weights = batch.get('weights')  # Weights might not be defined.
  eval_batch_size = targets.shape[0]
  if weights is None:
    weights = jnp.ones(eval_batch_size)

  # We don't use CLU's `mask` argument here, we handle it ourselves through
  # `weights`.
  return metrics_bundle.single_from_model_output(
      logits=logits, targets=targets, weights=weights)


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

  def __init__(
      self,
      hps,
      dataset_meta_data,
      loss_name,
      metrics_name,
      compile_init_on_cpu=False,
  ):
    self.hps = hps
    self.dataset_meta_data = dataset_meta_data
    self._loss_name = loss_name
    self.loss_fn = losses.get_loss_fn(loss_name, hps)
    self.output_activation_fn = losses.get_output_activation_fn(loss_name)
    self.metrics_bundle = metrics.get_metrics(metrics_name, hps)
    self._compile_init_on_cpu = compile_init_on_cpu
    self.flax_module = self.build_flax_module()

  def initialize(self, initializer, hps, rng, metrics_logger):
    """Run the given initializer.

    We initialize in 3 phases. First we run the default initializer that is
    specified by the model constructor. Next we apply any rescaling as specified
    by hps.layer_rescale_factors. Finally we run the black box initializer
    provided by the initializer arg (the default is noop).

    Args:
      initializer: An initializer defined in init_lib.
      hps: A dictionary specifying the model and initializer hparams.
      rng: An rng key to seed the initialization.
      metrics_logger: Used for black box initializers that have learning curves.

    Returns:
      A tuple (model, batch_stats), where model is the initialized
      flax.nn.Model and batch_stats is the collection used for batch norm.
    """
    model_dtype = utils.dtype_from_str(hps.model_dtype)
    # Note that this fake input batch will be optimized away because the init
    # function is jitted. However, this can still cause memory issues if it is
    # large because it is passed in as an XLA argument. Therefore we use a fake
    # batch size of 2 (we do not want to use 1 in case there is any np.squeeze
    # calls that would remove it), because we assume that there is no dependence
    # on the batch size with the model (batch norm reduces across a batch dim of
    # any size). This is similar to how the Flax examples initialize models:
    # https://github.com/google/flax/blob/44ee6f2f4130856d47159dc58981fb26ea2240f4/examples/imagenet/train.py#L70.
    fake_batch_hps = copy.copy(hps)
    fake_batch_hps.batch_size = 2
    fake_inputs = self.get_fake_inputs(fake_batch_hps)  # pylint: disable=assignment-from-none
    if fake_inputs:
      fake_input_batch = fake_inputs
    elif isinstance(hps.input_shape, list):  # Typical case for seq2seq models
      fake_input_batch = [
          np.zeros((2, *x), model_dtype) for x in hps.input_shape]
    else:  # Typical case for classification models
      fake_input_batch = [np.zeros((2, *hps.input_shape), model_dtype)]
    params_rng, init_rng, dropout_rng = jax.random.split(rng, num=3)

    # By jitting the model init function, we initialize the model parameters
    # lazily without computing a full forward pass. For further documentation,
    # see
    # https://flax.readthedocs.io/en/latest/flax.linen.html?highlight=jax.jit#flax.linen.Module.init.
    # We need to close over train=False here because otherwise the jitted init
    # function will convert the train Python bool to a jax boolean, which will
    # mess up Pythonic boolean statements like `not train` inside the model
    # construction.
    # We initialize model params on host to avoid memory issues.

    jit_kwargs = {}
    if self._compile_init_on_cpu:
      jit_kwargs['backend'] = 'cpu'
    logging.info(
        'Compiling model init on %s.',
        'cpu' if self._compile_init_on_cpu else 'device',
    )
    start_time = time.time()
    model_init_fn = jax.jit(
        functools.partial(self.flax_module.init, train=False), **jit_kwargs
    )

    init_dict = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                              *fake_input_batch)

    logging.info(
        'Flax module init call took %f seconds.',
        time.time() - start_time,
    )
    start_time = time.time()
    # Trainable model parameters.
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    if hps.get('layer_rescale_factors'):
      params = model_utils.rescale_layers(params, hps.layer_rescale_factors)
    logging.info('Layers rescaled in %f seconds.', time.time() - start_time)
    start_time = time.time()
    # We don't pass batch_stats to the initializer, the initializer will just
    # run batch_norm in train mode and does not need to maintain the
    # batch_stats.
    # TODO(gilmer): We hardcode here weighted_cross_entropy, but this will need
    # to change for other models. Maybe have meta_loss_inner as an initializer
    # hyper_param?
    # TODO(gilmer): instead of passing in weighted_xent, pass in the model and
    # get the loss from that.
    params = initializer(
        self.loss_fn,
        self.flax_module,
        params,
        hps,
        hps.input_shape,
        hps.output_shape,
        init_rng,
        metrics_logger,
    )
    logging.info('Initializer run in %f seconds.', time.time() - start_time)
    start_time = time.time()

    self._param_shapes = model_utils.param_shapes(params)
    self._param_types = model_utils.param_types(self._param_shapes)
    logging.info(
        'Param shapes and types computed in %f seconds.',
        time.time() - start_time,
    )

    return params, batch_stats

  # Individual models can override this method to specify sharding overrides.
  def get_sharding_overrides(self, mesh):
    """Returns the default sharding overrides for the model."""
    del mesh
    # Indicates all params are replicated.
    # Child model classes can override this dict to partition individual layers.
    param_type_to_sharding = {}

    return param_type_to_sharding

  def _apply_sharding_overrides(
      self, params, mesh, default_shardings, sharding_overrides
  ):
    """Applies the sharding overrides to the default shardings."""

    param_types = model_utils.param_types(params)
    param_shapes = jax.tree_util.tree_map(lambda arr: arr.shape, params)

    def _apply_override(sharding, param_type, param_shape):
      if param_type in sharding_overrides:
        if model_utils.is_shape_compatible_with_sharding(
            param_shape, sharding_overrides[param_type], mesh
        ):
          return sharding_overrides[param_type]
        else:
          raise ValueError(
              f'Param shape {param_shape} is not compatible with sharding'
              f' {sharding_overrides[param_type]}'
          )
      else:
        return sharding

    overriden_shardings = jax.tree_util.tree_map(
        _apply_override,
        default_shardings,
        param_types,
        param_shapes,
    )

    return overriden_shardings

  def get_sharding(self, params, mesh):
    """Returns the overriden sharding annotations for the model.

    The default sharding strategy is to replicate all layers on all devices.
    Models can override get_sharding_overrides() to specify sharding overrides
    for individual layers. The logic for merging the default sharding with the
    overrides lives in this method so child models only have to deal with
    get_sharding_overrides() specifying the overrides with a simple dict
    of the format shown below:

    overrides = {
        ParameterType.EMBEDDING: NamedSharding(mesh, P(None, 'devices')),
        ParameterType.WEIGHT: NamedSharding(mesh, P(None, 'devices')),
    }

    Args:
      params: The model parameters.
      mesh: The mesh to shard over.
    """

    # Replicate all layers on all devices.
    default_shardings = nn.get_sharding(params, mesh)
    sharding_overrides = self.get_sharding_overrides(mesh)

    overriden_shardings = self._apply_sharding_overrides(
        params, mesh, default_shardings, sharding_overrides)
    return overriden_shardings

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
      targets = jax.nn.one_hot(targets, logits.shape[-1])
    # Optionally apply label smoothing only if the loss used
    # is multi-class cross-entropy.
    label_smoothing = self.hps.get('label_smoothing')
    if self._loss_name == 'cross_entropy' and label_smoothing is not None:
      targets = model_utils.apply_label_smoothing(
          targets, label_smoothing
      )

    objective_numerator, objective_denominator = self.loss_fn(
        logits, targets, weights
    )

    # epsilon added to handle empty batch case if we encounter one.
    objective_value = objective_numerator / (objective_denominator + 1e-9)
    if self.hps.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(
          params, self.hps.l2_decay_rank_threshold
      )
      objective_value += 0.5 * self.hps.l2_decay_factor * l2_loss

    return objective_value

  def get_fake_inputs(self, hps):
    del hps
    return None

  @property
  def param_shapes(self):
    """The shapes of the parameters in the model."""
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, model.initialize() should be called '
          'before model.param_shapes!')
    return self._param_shapes

  @property
  def params_types(self):
    """The types of the parameters in the model."""
    if self._param_types is None:
      raise ValueError(
          'This should not happen, model.initialize() should be called '
          'before model.param_types!')
    return self._param_types

  def is_output_params(self, param_key: str) -> bool:
    """Whether a parameter name is the output layer parameters."""
    raise NotImplementedError('Subclasses must implement is_output_params().')

  def build_flax_module(self) -> nn.Module:
    """The flax module must accept a kwarg `train` in `__call__`."""
    raise NotImplementedError('Subclasses must implement build_flax_module().')
