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

"""Defines the MetaInit initializer.

This is a black box initializer which does pretraining optimization to find
weight scales in the NN layers which are better conditioned. These weights
are optimized to bias the weights towards flatter regions in the loss landscape.
"""

import functools
import json
import operator
import time

from absl import logging
from flax import jax_utils
from flax import optim as optimizers
from flax.deprecated import nn
from init2winit.model_lib import model_utils
import jax
from jax import jvp
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


# Small hparams for quicker tests.
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    meta_learning_rate=.1,
    meta_steps=50,
    meta_batch_size=8,
    epsilon=1e-5,
    meta_momentum=0.5,
))


def _count_params(tree):
  return jax.tree_util.tree_reduce(operator.add,
                                   jax.tree_map(lambda x: x.size, tree))


def scale_params(params, scalars):
  return jax.tree_multimap(lambda w, scale: w * scale, params, scalars)


def meta_loss(params_to_loss, scalars, normalized_params, epsilon):
  """Computes the meta_init objective function.

  This is a function of the learned scalars (one for each non-bias term in the
  model), and the original loss function. Note, the meta_loss is only a function
  of the variables with ndim > 1. Be careful how variables are shaped, this is
  potentially brittle, but the easiest way to avoid optimizing the bias terms.

  Args:
    params_to_loss: Function mapping (params) -> scalar valued loss.
    scalars: A pytree of scalars, one for each variable in the model. Note, we
      assume that all bias terms have a scale of 0.
    normalized_params: The pytree of model params. Requires that all params have
      norm equal to 1.
    epsilon: Used for numerical stability to avoid dividing by 0.

  Returns:
    Scalar value of the meta objective.
  """

  # Note, params_to_loss has captured the data for this particular shard.
  grad_loss = jax.grad(params_to_loss)

  scaled_params = scale_params(normalized_params, scalars)

  # The meta loss is a function of the data across all shards and does NOT
  # decompose as a sum over each data point. However the inner loss DOES
  # decompose, so we need to average the gradients of all of the data here.
  g = model_utils.cross_device_avg(grad_loss(scaled_params))

  # Again average across all of the shards.
  hgp = model_utils.cross_device_avg(jvp(grad_loss, [scaled_params], [g])[1])

  nparams = _count_params(g)

  def meta_term(g, hgp):
    ratio = (g-hgp) / (g + epsilon * jax.lax.stop_gradient(2*(g >= 0) - 1))
    return jnp.sum(jnp.abs(ratio - 1))

  return jax.tree_util.tree_reduce(
      operator.add, jax.tree_multimap(meta_term, g, hgp)) / nparams


def normalize(node):
  norm = jnp.linalg.norm(node.reshape(-1))
  if jnp.abs(norm) == 0.0:
    return node
  return node / norm


def _get_non_bias_params(params):
  flat_params = model_utils.flatten_dict(params)
  bias_and_scalar_keys = [
      key for key in flat_params if len(flat_params[key].shape) >= 2
  ]
  return bias_and_scalar_keys


def meta_optimize_scales(loss_fn,
                         fprop,
                         normalized_params,
                         norms,
                         hps,
                         input_shape,
                         output_shape,
                         rng_key,
                         metrics_logger=None,
                         log_every=10):
  """Implements MetaInit initializer.

  Args:
    loss_fn: Loss function.
    fprop: Forward pass of the model with API fprop(params, inputs) -> outputs.
    normalized_params: Pytree of model parameters. We assume that all non-bias
      terms have norm 1, and all bias terms are all 0's.
    norms: The initial guess of the learned norms, this is the starting point of
      meta_init.
    hps: HParam object. Required hparams are meta_learning_rate,
      meta_batch_size, meta_steps, and epsilon.
    input_shape: Must agree with batch[0].shape[1:].
    output_shape: Must agree with batch[1].shape[1:].
    rng_key: jax.PRNGKey, used to seed all randomness.
    metrics_logger: Supply a utils.MetricsLogger object.
    log_every: Log the meta loss every k steps.

  Returns:
    scales: The model scales after optimizing the meta_init loss.
    final_loss: The final meta objective value achieved.
  """
  num_outputs = output_shape[-1]
  if hps.meta_batch_size % jax.device_count() != 0:
    raise ValueError('meta_bs: {}, n_devices: {}'.format(
        hps.meta_batch_size, jax.device_count()))

  def get_batch(rng_key):
    """Return a fake batch of data."""
    meta_input_shape = (
        jax.local_device_count(),
        hps.meta_batch_size // jax.device_count(),
    ) + input_shape
    input_key, target_key = jax.random.split(rng_key)

    inputs = jax.random.normal(input_key, meta_input_shape)
    targets = jax.random.randint(target_key, (
        jax.local_device_count(),
        hps.meta_batch_size // jax.device_count(),
    ), 0, num_outputs)
    targets = jnp.eye(num_outputs)[targets]
    return (inputs, targets)

  # We will only optimize the scalars for model parameters with rank >=2.
  non_bias_and_scalar_keys = _get_non_bias_params(normalized_params)
  if jax.process_index() == 0:
    logging.info('MetaInit will optimize the following parameters:')
    for key in non_bias_and_scalar_keys:
      logging.info(key)
  traversal = optimizers.ModelParamTraversal(
      lambda path, _: path in non_bias_and_scalar_keys)

  # The flax optimizer expects a Model object. There is no fprop for the
  # meta_model, hence the first argument is None.
  meta_model = nn.Model(None, norms)
  # TODO(gilmer,znado): update to optax.
  meta_opt = optimizers.Momentum(hps.meta_learning_rate,
                                 hps.meta_momentum).create(
                                     meta_model, focus=traversal)
  meta_opt = jax_utils.replicate(meta_opt)

  # Make a closure over the static variables (model, normalized_params, hps).
  @functools.partial(jax.pmap, axis_name='batch')
  def update(optimizer, inputs, targets):
    """Update step."""
    params_to_loss = lambda params: loss_fn(fprop(params, inputs), targets)

    def _meta_loss(meta_model):
      return meta_loss(params_to_loss, meta_model.params, normalized_params,
                       hps.epsilon), None
    output_and_grad = optimizer.compute_gradient(_meta_loss)
    grads = model_utils.cross_device_avg(output_and_grad[-1])
    grads = jax.tree_map(jnp.sign, grads)
    optimizer = optimizer.apply_gradient(grads)
    return optimizer, output_and_grad[0]

  training_curve = []
  start = time.perf_counter()
  for i in range(hps.meta_steps):
    batch_rng = jax.random.fold_in(rng_key, i)
    inputs, targets = get_batch(batch_rng)

    meta_opt, loss_value = update(meta_opt, inputs, targets)
    training_curve.append(loss_value)
    if (jax.process_index() == 0 and
        (i % log_every == 0 or (i + 1) == hps.meta_steps)):
      end = time.perf_counter()
      logging.info('Cumulative time (seconds): %d', end-start)
      logging.info('meta_init step %d, loss: %f', i, float(loss_value[0]))
      if metrics_logger is not None:
        metrics_logger.append_scalar_metrics({
            'global_step': i,
            'meta_loss': float(loss_value[0])
        })

  # Create a new model with the learned init
  learned_norms = jax_utils.unreplicate(meta_opt).target.params
  return learned_norms, training_curve


def _log_shape_and_norms(pytree, metrics_logger, key):
  shape_and_norms = jax.tree_map(
      lambda x: (str(x.shape), str(np.linalg.norm(x.reshape(-1)))),
      pytree)
  logging.info(json.dumps(shape_and_norms, sort_keys=True, indent=4))
  if metrics_logger is not None:
    metrics_logger.append_json_object({'key': key, 'value': shape_and_norms})


def meta_init(loss_fn,
              model,
              hps,
              input_shape,
              output_shape,
              rng_key,
              metrics_logger=None,
              log_every=10):
  """Implements MetaInit initializer.

  Args:
    loss_fn: Loss function.
    model: Flax Model class.
    hps: HParam object. Required hparams are meta_learning_rate,
      meta_batch_size, meta_steps, and epsilon.
    input_shape: Must agree with batch[0].shape[1:].
    output_shape: Must agree with batch[1].shape[1:].
    rng_key: jax.PRNGKey, used to seed all randomness.
    metrics_logger: Instance of utils.MetricsLogger
    log_every: Print meta loss every k steps.

  Returns:
    A Flax model with the learned initialization.
  """
  # Pretty print the preinitialized norms with the variable shapes.
  if jax.process_index() == 0:
    logging.info('Preinitialized norms:')
    _log_shape_and_norms(model.params, metrics_logger, key='init_norms')
    # First grab the norms of all weights and rescale params to have norm 1.
    logging.info('Running meta init')
  norms = jax.tree_map(lambda node: jnp.linalg.norm(node.reshape(-1)),
                       model.params)

  normalized_params = jax.tree_map(normalize, model.params)

  learned_norms, _ = meta_optimize_scales(
      loss_fn,
      model.module.call,
      normalized_params,
      norms,
      hps,
      input_shape,
      output_shape,
      rng_key,
      metrics_logger=metrics_logger,
      log_every=log_every)
  new_init = scale_params(normalized_params, learned_norms)

  if jax.process_index() == 0:
    # Pretty print the meta init norms with the variable shapes.
    logging.info('Learned norms from meta_init:')
    _log_shape_and_norms(new_init, metrics_logger, key='meta_init_norms')

  return nn.Model(model.module, new_init)
