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

"""Base class for training algorithms."""

import abc
import collections

from init2winit import schedules
from init2winit.model_lib import model_utils
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import optimizers
import jax
import jax.numpy as jnp
import optax


_GRAD_CLIP_EPS = 1e-6


def optax_update_params_helper(
    params,
    model_state,
    optimizer_state,
    optimizer_update_fn,
    batch,
    lr,
    rng,
    grad_clip,
    training_cost_fn,
):
  """Helper function for updating parameters using optax.

  Args:
    params: The current model parameters.
    model_state: The current state of the model.
    optimizer_state: The current state of the optimizer.
    optimizer_update_fn: The optimizer update function.
    batch: The current batch of data.
    lr: The learning rate.
    rng: The random number generator.
    grad_clip: The gradient clipping value.
    training_cost_fn: The training cost function.

  Returns:
    A tuple containing the new optimizer state, the new model parameters,
    the new model state, and a dictionary of metrics.
  """
  optimizer_state = optimizers.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost_fn(
        params, batch=batch, batch_stats=model_state, dropout_rng=rng
    )

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  if grad_clip:
    scaled_grad = jax.tree.map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad
    )
    grad = jax.lax.cond(
        grad_norm > grad_clip, lambda _: scaled_grad, lambda _: grad, None
    )
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats,
      cost_fn=opt_cost,
      grad_fn=grad_fn,
      value=cost_value,
  )

  update_norm = jnp.sqrt(model_utils.l2_regularization(model_updates, 0))

  new_params = optax.apply_updates(params, model_updates)
  return (
      new_optimizer_state,
      new_params,
      new_batch_stats,
      cost_value,
      grad,
      grad_norm,
      update_norm,
  )


class TrainingAlgorithm(metaclass=abc.ABCMeta):
  """Base class for training algorithms."""

  def __init__(self, hps, model, num_train_steps):
    self.model = model
    self.num_train_steps = num_train_steps
    self.hps = hps
    self.eval_report_metrics = collections.defaultdict()

  @abc.abstractmethod
  def update_params(
      self,
      params,
      model_state,
      optimizer_state,
      batch,
      global_step,
      rng,
      hyperparameters=None,
      workload=None,
      param_types=None,
      loss_type=None,
      train_state=None,
      eval_results=None,
  ):
    """Updates the model parameters.

    Args:
      params: The current model parameters.
      model_state: The current state of the model.
      optimizer_state: The current state of the optimizer.
      batch: The current batch of data.
      global_step: The current training step.
      rng: The random number generator.
      hyperparameters: The hyperparameters for the training.
      workload: The workload being trained.
      param_types: The types of the parameters.
      loss_type: The type of loss function to use.
      train_state: The optional training state.
      eval_results: The optional evaluation results.

    Returns:
      A tuple containing:
        new_optimizer_state: Pytree of optimizer state.
        new_params: Pytree of model parameters.
        new_model_state: Pytree of model state.
        cost_value: The training cost.
        grad: The gradient.
    """

  @abc.abstractmethod
  def init_optimizer_state(
      self,
      workload=None,
      params=None,
      model_state=None,
      hyperparameters=None,
      rng=None,
  ):
    """Initializes the optimizer state.

    Args:
      workload: The workload being trained.
      params: The initial model parameters.
      model_state: The initial state of the model.
      hyperparameters: The hyperparameters for the training.
      rng: The random number generator.

    Returns:
      Optimizer state: Pytree of optimizer state.
    """


class OptaxTrainingAlgorithm(TrainingAlgorithm):
  """Class for training algorithms implemented with optax and defined in optimizer_lib.optimizers.py."""

  def __init__(self, hps, model, num_train_steps):
    super().__init__(hps, model, num_train_steps)
    self._optimizer_state = None
    self._update_fn = None
    self._lr_fn = None
    self.training_cost_fn = model.training_cost

  def update_params(
      self,
      params,
      model_state,
      optimizer_state,
      batch,
      global_step,
      rng,
      hyperparameters=None,
      workload=None,
      param_types=None,
      loss_type=None,
      train_state=None,
      eval_results=None,
  ):
    """Updates the model parameters.

    Args:
      params: The current model parameters.
      model_state: The current state of the model.
      optimizer_state: The current state of the optimizer.
      batch: The current batch of data.
      global_step: The current training step.
      rng: The random number generator.
      hyperparameters: The hyperparameters for the training.
      workload: The workload being trained.
      param_types: The types of the parameters.
      loss_type: The type of loss function to use.
      train_state: The optional training state.
      eval_results: The optional evaluation results.

    Returns:
      A tuple containing:
        new_optimizer_state: Pytree of optimizer state.
        new_params: Pytree of model parameters.
        new_model_state: Pytree of model state.
    """
    del (
        workload,
        hyperparameters,
        param_types,
        loss_type,
        train_state,
        eval_results,
    )  # Unused
    grad_clip = self.hps.opt_hparams.get('grad_clip', None)
    # We pass the lr directly because the lr functions from sehedules.py
    # have numpy dependencies and can't be jitted.
    lr = self._lr_fn(global_step)
    jitted_update_fn = jax.jit(
        optax_update_params_helper,
        static_argnames=(
            'training_cost_fn',
            'optimizer_update_fn',
        ),
        donate_argnums=(0, 1, 2),
    )
    (
        new_optimizer_state,
        new_params,
        new_batch_stats,
        cost_value,
        grad,
        grad_norm,
        update_norm,
    ) = jitted_update_fn(
        params,
        model_state,
        optimizer_state,
        self._update_fn,
        batch,
        lr,
        rng,
        grad_clip,
        self.training_cost_fn,
    )

    self.eval_report_metrics.update(
        learning_rate=lr,
        grad_norm=grad_norm.item(),
        update_norm=update_norm.item(),
    )
    self._optimizer_state = new_optimizer_state

    return new_optimizer_state, new_params, new_batch_stats, cost_value, grad

  def init_optimizer_state(
      self,
      workload=None,
      params=None,
      model_state=None,
      hyperparameters=None,
      rng=None,
  ):
    """Initializes the optimizer state.

    Args:
      workload: The workload being trained.
      params: The initial model parameters.
      model_state: The initial state of the model.
      hyperparameters: The hyperparameters for the training.
      rng: The random number generator.

    Returns:
      Optimizer state: Pytree of optimizer state.
    """
    del workload, model_state, hyperparameters, rng  # Unused
    stretch_factor = 1
    if self.hps.get('total_accumulated_batch_size') is not None:
      stretch_factor = (
          self.hps.total_accumulated_batch_size // self.hps.batch_size
      )

    self._lr_fn = schedules.get_schedule_fn(
        self.hps.lr_hparams,
        max_training_updates=self.num_train_steps // stretch_factor,
        stretch_factor=stretch_factor,
    )

    optimizer_init_fn, optax_optimizer_update_fn = optimizers.get_optimizer(
        self.hps, self.model, batch_axis_name='batch'
    )
    optax_optimizer_state = optimizer_init_fn(params)
    self._optimizer_state = optax_optimizer_state
    self._update_fn = optax_optimizer_update_fn
    return optax_optimizer_state

  # TODO(b/436634470): Consolidate this with the prepare_for_eval API
  def get_ema_eval_params(self, optimizer_state):
    """Extracts the exponential moving average (EMA) parameters from the optimizer state.

    Args:
      optimizer_state: The current state of the optimizer.

    Returns:
      The EMA parameters.

    Raises:
      ValueError: If the EMA parameters cannot be extracted from the optimizer
        state.
    """
    if isinstance(optimizer_state, optax.InjectStatefulHyperparamsState):
      eval_params = optimizer_state.inner_state[0][0].ema
    elif isinstance(
        optimizer_state, gradient_accumulator.GradientAccumulatorState
    ):
      eval_params = optimizer_state.base_state.inner_state[0][0].ema
    else:
      raise ValueError(
          'EMA computation should be the very first transformation in defined'
          ' kitchensink optimizer.'
      )
    return eval_params
