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

"""JAX-based Python script for defining a workload structure.

This Python file defines an abstract base class BaseWorkload,
which outlines the structure for training a machine learning model
using JAX and Flax libraries.

The config dictionary is used to specify the training parameters,
such as the number of epochs, batch size, and base learning rate.
The data_parallel flag is used to enable data parallelism across devices.

As an example, the following code snippet shows how to set the workload config:
config = {
    'use_dummy_data': True,      # Use dummy data for testing
    'batch_size': 128,           # Batch size for training
    'total_steps': 100,          # Number of training steps
    'compute_option': 'vanilla'  # Compute option for training without vmap or
    jit
}
"""

import abc
import copy
from typing import Any, Dict, Tuple, Union

from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import sharding
from jax.experimental import mesh_utils
# pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P
import ml_collections
import numpy as np


# pylint: enable=g-importing-member

jax.config.update(
    'jax_default_prng_impl', 'threefry2x32'
)  # Replace PRNG for reproducibility
jax.config.update(
    'jax_threefry_partitionable', True
)  # Enable partitionable PRNG

ConfigType = Union[Dict[str, Any], ml_collections.ConfigDict]


class BaseWorkload(abc.ABC):
  """Abstract base class for workloads, defines basic training structure."""

  def __init__(self, config: ConfigType) -> None:
    """Initialize workload configuration.

    Args:
        config (ConfigType): Configuration dictionary for training parameters.
    """

    self.config = copy.deepcopy(config)
    if isinstance(self.config, ml_collections.ConfigDict):
      self.config = config.to_dict()

    # Add optimizer configuration
    default_optimizer_config = {
        'beta1': 0.9,  # Adam beta1
        'beta2': 0.999,  # Adam beta2
    }
    # Update config with defaults if not provided
    if 'optimizer_config' not in self.config:
      self.config['optimizer_config'] = default_optimizer_config
    else:
      for k, v in default_optimizer_config.items():
        if k not in self.config['optimizer_config']:
          self.config['optimizer_config'][k] = v

    # Add evaluation configuration
    default_eval_config = {
        'eval_mode': 'step',  # 'step' or 'epoch'
        'eval_frequency': 1000,  # Evaluate every N steps or every N epochs
    }
    # Update config with defaults if not provided
    if 'eval_config' not in self.config:
      self.config['eval_config'] = default_eval_config
    else:
      for k, v in default_eval_config.items():
        if k not in self.config['eval_config']:
          self.config['eval_config'][k] = v

    # Create device mesh for data parallelism
    self.num_devices = len(jax.devices())
    if self.config['batch_size'] % self.num_devices != 0:
      raise ValueError('Batch size must be divisible by the number of devices.')
    mesh = sharding.Mesh(
        mesh_utils.create_device_mesh((self.num_devices,)), ('batch',)
    )

    # Sharding strategy: data parallelism across the batch dimension
    self.sharding = sharding.NamedSharding(mesh, P('batch'))
    self.replicated_sharding = sharding.NamedSharding(
        mesh, P()
    )  # Replicate across all devices

  def make_global_array(self, process_local_data):
    """Combines per-host batches into a global batch array."""
    global_shape = (
        process_local_data.shape[0] * jax.process_count(),
        *process_local_data.shape[1:],
    )
    return jax.make_array_from_process_local_data(
        self.sharding, process_local_data, global_shape
    )

  def should_evaluate(self, global_step: int, epoch: int) -> bool:
    """Determine if evaluation should be performed based on current step/epoch.

    Args:
        global_step: Current global step
        epoch: Current epoch

    Returns:
        bool: True if evaluation should be performed
    """
    eval_config = self.config['eval_config']
    eval_mode = eval_config['eval_mode']
    eval_frequency = eval_config['eval_frequency']

    if eval_mode == 'step':
      return (global_step + 1) % eval_frequency == 0
    elif eval_mode == 'epoch':
      return (epoch + 1) % eval_frequency == 0
    else:
      raise ValueError(f'Unknown eval_mode: {eval_mode}')

  @abc.abstractmethod
  def _load_data(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess dataset.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Preprocessed
        training and test datasets.
        - x_train: Training images (NumPy array)
        - y_train: Training labels (NumPy array)
        - x_test: Test images (NumPy array)
        - y_test: Test labels (NumPy array)
    """
    pass

  @abc.abstractmethod
  def create_train_state(
      self, init_param_rng: jax.Array
  ) -> train_state.TrainState:
    """Initialize model parameters and optimizer state.

    Args:
        init_param_rng (jax.Array): Random key for parameter initialization.

    Returns:
        train_state.TrainState: The initialized training state containing model
        parameters and optimizer state.
    """
    pass

  @abc.abstractmethod
  def evaluate_models(
      self,
      state: train_state.TrainState,
      x: jnp.ndarray,
      y: jnp.ndarray,
  ) -> dict[str, jnp.ndarray]:
    """Evaluate the model on test data.

    Args:
        state: Train state where each element in tree_leaves(state.params)
          returns the model is of shape (num_models, ...).
        x (jnp.ndarray): Test images
        y (jnp.ndarray): Test labels

    Returns:
        dict[str, jnp.ndarray]: A dictionary containing the following keys and
        values.
          - The computed loss on (x,y) all models.
          - The computed error on the data (x,y) for all models.
    """
    pass

  @abc.abstractmethod
  def train_and_evaluate_models(
      self,
      schedules: np.ndarray,
      params_rngs: jax.Array,
      data_rng: jax.Array,
  ) -> Tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Train the model using a specific schedule.

    Args:
        schedules (np.ndarray): Array of learning rate schedule.
        params_rngs (jax.Array): JAX random key for randomness during training.
        data_rng (jax.Array): JAX random key for data randomness.

    Returns:
        Tuple[
          train_state.TrainState,  # vmap_states: Contains model states after
          training.
          dict[str, jnp.ndarray] : A dictionary containing the results keys and
          values.
    """
    pass
