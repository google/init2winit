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

r"""Test class for validating workload implementations.

This Python file defines a test class TestWorkloads using absltest, which
includes tests for training workloads in the optlrschedule project, particularly
focusing on the CIFAR-10 training workload.
"""

import logging
from absl.testing import absltest
from init2winit.projects.optlrschedule.workload.cifar10_cnn import (
    Cifar10Training,
)
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_default_matmul_precision', 'float32')


class TestWorkloads(absltest.TestCase):
  """Test class for workloads in the optlrschedule project.

  This class contains tests for different workloads, such as CIFAR-10 training.
  """

  TOTAL_STEPS = 80

  eval_config = {
      'eval_mode': 'step',  # 'step' or 'epoch'
      'eval_frequency': 40,  # Evaluate every N steps or every N epochs
  }

  # Configuration for the model and training
  config = {
      'rng_seed': 0,
      'batch_size': 4,
      'total_steps': TOTAL_STEPS,
      'use_dummy_data': True,
      'compute_option': 'vanilla',
      'eval_config': eval_config,
      'optimizer': 'sgd',
  }

  def test_train_and_evaluate_models_with_jit_vmap_schedules(self):
    """Test function for dummy training."""

    num_devices = len(jax.devices())
    logging.info('num_devices: %d', num_devices)

    num_schedule = 3
    seeds = np.ones((num_schedule,), dtype=int) * self.config['rng_seed']
    params_rngs = jax.vmap(jax.random.key)(seeds)
    loader_rng = jax.random.key(self.config['rng_seed'])

    num_of_steps = 1
    self.config['total_steps'] = num_of_steps
    self.config['compute_option'] = 'jit(vmap)'

    # training w/ lr = 0, 1, 2
    schedules = np.zeros((num_schedule, num_of_steps))
    schedules[0] = np.zeros((num_of_steps,))
    schedules[1] = np.ones((num_of_steps,)) * 1
    schedules[2] = np.ones((num_of_steps,)) * 2

    cifar10_workload = Cifar10Training(self.config)
    (
        states,
        _,
    ) = cifar10_workload.train_and_evaluate_models(
        schedules, params_rngs, loader_rng
    )

    # Calculate the distance between the parameters
    def flatten_params(params):
      flat_params, _ = jax.flatten_util.ravel_pytree(params)
      return flat_params

    flat_params = jax.vmap(lambda state: flatten_params(state.params))(states)

    flat_params_0 = flat_params[0]
    flat_params_1 = flat_params[1]
    flat_params_2 = flat_params[2]

    distance_1 = np.linalg.norm(flat_params_1 - flat_params_0)
    distance_2 = np.linalg.norm(flat_params_2 - flat_params_0)

    # distance_1 should be smaller than distance_2
    self.assertLess(distance_1, distance_2)

  def test_train_and_evaluate_models_with_jit_vmap_prngs(self):
    """Test that variance from different initializations vs schedules.

    Results
    [Seeds] variance:                  0.012265
    [Schedules] variance:              98.640579
    [Seeds] error rate variance:       0.000645
    [Schedules] error rate variance:   0.000742
    """
    num_devices = len(jax.devices())
    logging.info('num_devices: %d', num_devices)

    # Configuration
    num_of_steps = 100  # Longer training to see the effect
    self.config.update({
        'total_steps': num_of_steps,
        'compute_option': 'jit(vmap)',
        'batch_size': 16,
        'use_dummy_data': True,  # Use dummy data for faster testing
    })

    # Test 1: Different initializations with same schedule
    # Create workload
    cifar10_workload = Cifar10Training(self.config)
    seeds = np.array([0, 1, 2, 3, 4])  # Five seeds for the experiment
    num_models = len(seeds)

    params_rngs = jax.vmap(jax.random.key)(seeds)
    loader_rng = jax.random.key(self.config['rng_seed'])

    constant_schedule = np.ones(num_of_steps) * 0.1
    schedules = jax.vmap(lambda _: constant_schedule)(jnp.zeros(num_models))

    # Train models with different initializations
    (init_states, _) = cifar10_workload.train_and_evaluate_models(
        schedules, params_rngs, loader_rng
    )
    del cifar10_workload, schedules, params_rngs, loader_rng

    # Test 2: Same initialization with different schedules
    cifar10_workload = Cifar10Training(self.config)
    seeds = np.array([0, 0, 0, 0, 0])  # Five seeds for the experiment
    params_rngs = jax.vmap(jax.random.key)(seeds)
    loader_rng = jax.random.key(self.config['rng_seed'])

    num_schedules = len(seeds)
    schedules = np.zeros((num_schedules, num_of_steps))
    for i in range(num_schedules):
      schedules[i] = np.ones(num_of_steps) * (
          0.1 * (i + 1)
      )  # Different constant learning rates

    # Train models with different schedules
    (schedule_states, _) = cifar10_workload.train_and_evaluate_models(
        schedules, params_rngs, loader_rng
    )

    # Helper function to calculate parameter distance variance
    def calculate_param_variance(states):
      def flatten_params(params):
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        return flat_params

      # Flatten all parameters
      flat_params = jax.vmap(lambda state: flatten_params(state.params))(states)

      # Calculate pairwise distances
      distances = []
      for i in range(len(flat_params)):
        for j in range(i + 1, len(flat_params)):
          distances.append(np.linalg.norm(flat_params[i] - flat_params[j]))

      return np.var(distances)

    # Calculate variances
    init_variance = calculate_param_variance(init_states)
    schedule_variance = calculate_param_variance(schedule_states)

    logging.info('Initialization variance: %f', init_variance)
    logging.info('Schedule variance: %f', schedule_variance)

    # Assert that initialization variance is smaller than schedule variance
    self.assertLess(init_variance, schedule_variance)


if __name__ == '__main__':
  absltest.main()
