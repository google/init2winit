# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

r"""Linear regression workload for testing different schedules.

Notation:
  P: number of parameters
  D: number of data points

"""

import functools
from typing import Tuple

from flax.training import train_state
from init2winit.projects.optlrschedule.workload import base_workload
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax

partial = functools.partial


def create_dummy_train_state():
  """Creates a dummy TrainState with placeholder values."""
  dummy_params = {'w': jnp.array(0.0)}  # Dummy model parameters
  dummy_tx = optax.sgd(learning_rate=0.0)  # Dummy optimizer
  return train_state.TrainState.create(
      apply_fn=lambda x: x, params=dummy_params, tx=dummy_tx
  )


def mean_squared_error(z: jnp.ndarray) -> jnp.ndarray:
  """Computes mean squared error loss."""
  return 0.5 * jnp.mean(z**2, axis=-1)


def generate_ntk_matrix(
    key: jax.Array,
    spectrum: jnp.ndarray | None = None,
    num_data: int | None = None,
    num_params: int | None = None,
) -> jnp.ndarray:
  """Generates an NTK (Neural Tangent Kernel) matrix with a known spectrum."""
  if spectrum is None:
    weight_matrix = random.normal(key, shape=(num_data, num_params))
    return weight_matrix @ weight_matrix.T / num_data
  else:
    orthogonal_matrix = random.orthogonal(key, len(spectrum))
    return orthogonal_matrix @ jnp.diag(spectrum) @ orthogonal_matrix.T


def get_loss_traj_fn_lax(spectrum, p0, batch_size, num_data):
  """Finite differences version of theoretical average loss using lax.scan.
  
  Args:
    spectrum: spectrum of NTK matrix
    p0: initial normalized residuals - eigenmode projection * spectrum**-1
    batch_size: batch size
    num_data: number of data points
  Returns:
    Function that takes array of learning rates as input, and reurns loss
    trajectory.
  """
  def sing_layer(p, lr):
    p_next = ((1.-lr*spectrum)**2)*p+(lr**2)*(
        (1./batch_size-1./num_data)*spectrum*jnp.sum(spectrum*p))
    return p_next, jnp.mean(spectrum*p_next)/2.
  def loss_traj_fn(lrs):
    _, losses = jax.lax.scan(sing_layer, p0, lrs)
    return losses
  return loss_traj_fn


def initialize_regression(
    param_rng: jax.Array,
    spectrum: jnp.ndarray | None = None,
    init_z: jnp.ndarray | None = None,
    num_traj: int | None = None,
    num_data: int | None = None,
    num_params: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes a linear regression problem with an NTK matrix and initial values."""
  ntk_key, z_key = random.split(param_rng)
  ntk_matrix = generate_ntk_matrix(
      ntk_key, spectrum=spectrum, num_data=num_data, num_params=num_params
  )
  num_data = ntk_matrix.shape[-1]

  z_init = (
      init_z
      if init_z is not None
      else random.normal(z_key, shape=(num_traj or 1, num_data))
  )
  return z_init, ntk_matrix


def _train_step(
    key: jax.Array,
    learning_rate: float,
    z: jnp.ndarray,
    ntk_matrix: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
  """Performs a single training step for linear regression using mini-batch updates."""
  num_data = z.shape[1]

  # Select random indices for mini-batch
  batch_indices = random.choice(key, num_data, (batch_size,), replace=False)

  # Compute batch fraction normalization factor
  batch_norm_factor = batch_size / num_data

  # Update rule using NTK
  return z - (learning_rate / batch_norm_factor) * jnp.einsum(
      'ad,id->ai', z[:, batch_indices], ntk_matrix[:, batch_indices]
  )


class LinearRegression(base_workload.BaseWorkload):
  """Linear regression workload for testing different schedules."""

  def __init__(self, config: base_workload.ConfigType) -> None:
    """Initializes the linear regression training setup.

    Args:
      config: A dictionary containing the following keys and values:
        - batch_size: The batch size for each training step.
        - total_steps: The total number of training steps.
        - spectrum: The spectrum of the NTK matrix.
                    1D NumPy array of shape (num_data,),
                    where num_data is the number of data points.
                    This value should be normalized to sum to 1.
        - init_z: The initial values of the regression problem.
        - num_data: The number of data points in the regression problem.
        - num_params: The number of parameters in the regression problem.
        - return_loss_history: Whether to return the loss history. Set to False
          for compatibility with run_search_decoupled.
        - compute_option: The compute option for the training step.
    """
    super().__init__(config)
    self.batch_size = config['batch_size']
    self.total_steps = config['total_steps']
    self.spectrum = config['spectrum']
    self.init_z = config['init_z']
    self.num_data = config['num_data']
    self.num_params = config['num_params']
    self.return_loss_history = config.get('return_loss_history', False)

    assert config['compute_option'] == 'vmap(jit)'

    # JIT compilation for training step
    self.train_step = jax.jit(partial(_train_step, batch_size=self.batch_size))

    # Vectorized function for parallel training
    self.vmapped_train_and_evaluate = jax.vmap(
        self.train_and_evaluate_model,
        in_axes=(
            0,
            0,
            None,
        ),  # Parallelize over schedules and params_rngs, share data_rng
    )

  def _load_data(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Not used in this subclass."""
    raise NotImplementedError(
        '_load_data method is intentionally left blank as it is not used in'
        ' this subclass.'
    )

  def create_train_state(
      self, init_param_rng: jax.Array
  ) -> train_state.TrainState:
    """Not used in this subclass."""
    raise NotImplementedError(
        'create_train_state method is intentionally left blank as it is not'
        ' used in this subclass.'
    )

  def evaluate_models(
      self, state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
  ):
    """Not used in this subclass."""
    raise NotImplementedError(
        'evaluate_models method is intentionally left blank as it is not used'
        ' in this subclass.'
    )

  def train_and_evaluate_model(
      self,
      schedule: np.ndarray,
      param_rng: jax.Array,
      data_rng: jax.Array,
  ) -> dict[str, jnp.ndarray]:
    """Trains a model using a single learning rate schedule."""

    # Initialize regression problem
    z, ntk_matrix = initialize_regression(
        param_rng,
        spectrum=self.spectrum,
        init_z=self.init_z,
        num_data=self.num_data,
        num_params=self.num_params,
    )

    # Store loss values
    loss_history = jnp.zeros((z.shape[0], self.total_steps + 1))

    # Training loop
    for step in range(self.total_steps):
      step_key = jax.random.fold_in(data_rng, step)
      loss_history = loss_history.at[:, step].set(mean_squared_error(z))
      z = self.train_step(step_key, schedule[step], z, ntk_matrix)

    loss_history = loss_history.at[:, self.total_steps].set(
        mean_squared_error(z)
    )

    if self.return_loss_history:
      return {
          'loss_history': loss_history[0],
          'final_loss': loss_history[0][-1],
      }
    else:  # return_loss_history not compatible with run_search_decoupled
      return {'final_loss': loss_history[0][-1]}

  def train_and_evaluate_models(
      self,
      schedules: np.ndarray,
      params_rngs: jax.Array,
      data_rng: jax.Array,
  ) -> Tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Trains multiple models in parallel using vmap.

    Args:
      schedules: np.ndarray: Learning rate schedules for each model (num_models,
        num_steps).
      params_rngs: jax.Array: Random number generators for each model
        (num_models).
      data_rng: jax.Array: Random number generator for the data.

    Returns:
      dummy_state: A dummy TrainState object (we don't use it, but just keep
      currrent interface).
      results: A dictionary containing the training results for each model.
    """

    dummy_state = create_dummy_train_state()
    results = self.vmapped_train_and_evaluate(schedules, params_rngs, data_rng)
    return dummy_state, results
