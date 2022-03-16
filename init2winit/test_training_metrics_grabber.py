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

"""Unit tests for trainer.py.

"""

import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from init2winit import checkpoint
from init2winit.shared_test_utilities import pytree_equal
from init2winit.training_metrics_grabber import TrainingMetricsGrabber
import jax
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np


FLAGS = flags.FLAGS


class TrainingMetricsGrabberTest(jtu.JaxTestCase):
  """Tests the logged statistics from training_metrics_grabber."""

  def setUp(self):
    super(TrainingMetricsGrabberTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(TrainingMetricsGrabberTest, self).tearDown()

  def test_grad_var(self):
    model_size = 10
    example_grads = [{
        'layer1': np.ones(model_size),
        'layer2': 3 * np.ones(model_size)
    }, {
        'layer1': 2 * np.ones(model_size),
        'layer2': np.ones(model_size)
    }]
    eval_config = {'ema_beta': 0.5}
    training_metrics_grabber = TrainingMetricsGrabber.create(
        example_grads[0], eval_config)

    for grad in example_grads:
      training_metrics_grabber = training_metrics_grabber.update(
          1.0, grad, example_grads[0], example_grads[0])

    for layer in ['layer1', 'layer2']:
      expected_grad_ema = 1 / 4 * np.zeros(model_size) + 1 / 4 * example_grads[
          0][layer] + 1 / 2 * example_grads[1][layer]

      self.assertArraysAllClose(
          expected_grad_ema,
          training_metrics_grabber.state['param_tree_stats'][layer].grad_ema)

  def test_serialize_in_checkpoint(self):
    """Test that the TrainingMetricsGrabber can be serialized and restored."""

    initial_params = {'foo': jnp.zeros(5), 'bar': jnp.zeros(5)}
    initial_gradient = {'foo': 1*jnp.ones(5), 'bar': 2*jnp.ones(5)}
    new_params = jax.tree_map(lambda p, g: p - 0.1*g,
                              initial_params,
                              initial_gradient)

    initial_grabber = TrainingMetricsGrabber.create(
        initial_params, {'ema_beta': 0.5})
    new_grabber = initial_grabber.update(
        1.0, initial_gradient, initial_params, new_params)

    checkpoint.save_checkpoint(self.test_dir, 1,
                               {'training_metrics_grabber': new_grabber})

    loaded_checkpoint = checkpoint.load_latest_checkpoint(
        self.test_dir, {'training_metrics_grabber': initial_grabber})
    loaded_grabber = loaded_checkpoint['training_metrics_grabber']

    self.assertTrue(pytree_equal(loaded_grabber.state, new_grabber.state))

  def test_global_stats(self):
    """Test that the global stats are computed correctly."""

    lr = 0.1

    # Create mocks.
    params = {'foo': jnp.zeros(5), 'bar': jnp.zeros(5)}
    first_gradient = {'foo': 1*jnp.ones(5), 'bar': 2*jnp.ones(5)}
    first_loss = 4.0
    second_gradient = {'foo': 2*jnp.ones(5), 'bar': 3*jnp.ones(5)}
    second_loss = 2.0

    grabber = TrainingMetricsGrabber.create(params, {'ema_beta': 0.5})

    # Simulate first step of GD.
    new_params = jax.tree_map(lambda p, g: p - lr*g, params, first_gradient)
    grabber = grabber.update(first_loss, first_gradient, params, new_params)
    params = new_params

    # Simulate second step of GD.
    new_params = jax.tree_map(lambda p, g: p - lr*g, params, second_gradient)
    grabber = grabber.update(second_loss, second_gradient, params, new_params)
    params = new_params

    # Assert that the training metrics are as expected.
    expected_first_update_normsq = (lr**2) * ((1**2) * 5 + (2**2) * 5)
    expected_second_update_normsq = (lr**2) * ((2**2) * 5 + (3**2) * 5)

    global_stats = grabber.state['global_stats']

    self.assertArraysEqual(global_stats['train_cost_series'],
                           jnp.array([first_loss, second_loss]))

    self.assertArraysAllClose(
        global_stats['param_normsq_series'],
        jnp.array([0.0, expected_first_update_normsq]),
        atol=1e-7)

    self.assertArraysAllClose(
        global_stats['update_normsq_series'],
        jnp.array([expected_first_update_normsq,
                   expected_second_update_normsq]),
        atol=1e-7)

if __name__ == '__main__':
  absltest.main()
