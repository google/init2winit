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

from absl import flags
from absl.testing import absltest
from init2winit.shared_test_utilities import pytree_equal
from init2winit.training_metrics_grabber import make_training_metrics
import jax
from jax import test_util as jtu
import jax.numpy as jnp


FLAGS = flags.FLAGS


class TrainingMetricsGrabberTest(jtu.JaxTestCase):
  """Tests the logged statistics from training_metrics_grabber."""

  def setUp(self):
    super(TrainingMetricsGrabberTest, self).setUp()
    self.mock_params0 = {'foo': jnp.zeros(5), 'bar': {'baz': jnp.ones(10)}}
    self.mock_grad1 = {'foo': -jnp.ones(5), 'bar': {'baz': -jnp.ones(10)}}
    self.mock_grad2 = {'foo': -2*jnp.ones(5), 'bar': {'baz': -2*jnp.ones(10)}}

    self.mock_cost0 = 1.0
    self.mock_cost1 = 0.5
    self.mock_cost2 = 0.25

    self.num_train_steps = 5

    # Simulate running GD with step size 1.
    self.mock_params1 = jax.tree_map(lambda p, g: p - 1.0 * g,
                                     self.mock_params0,
                                     self.mock_grad1)
    self.mock_params2 = jax.tree_map(lambda p, g: p - 1.0 * g,
                                     self.mock_params1,
                                     self.mock_grad2)

    self.mock_zeros = {'foo': jnp.zeros(5), 'bar': {'baz': jnp.zeros(10)}}

  def test_init(self):
    """Test the training metrics initializer."""

    zeros_like_params = jax.tree_map(jnp.zeros_like, self.mock_params0)
    zeros_scalar_like_params = jax.tree_map(lambda x: 0.0, self.mock_params0)
    zeros_timeseries_like_params = jax.tree_map(
        lambda x: jnp.zeros(self.num_train_steps), self.mock_params0)

    # Test init with everything disabled.
    init_fn, _, _ = make_training_metrics(self.num_train_steps)
    initial_metrics_state = init_fn(self.mock_params0)
    self.assertTrue(
        pytree_equal({'param_norm': zeros_scalar_like_params},
                     initial_metrics_state))

    # Test init with enable_ema = True and enable_train_cost=True.
    init_fn, _, _ = make_training_metrics(self.num_train_steps,
                                          enable_ema=True,
                                          enable_train_cost=True,
                                          enable_param_norms=True)
    initial_metrics_state = init_fn(self.mock_params0)
    self.assertTrue(pytree_equal(initial_metrics_state, {
                'train_cost': jnp.zeros(self.num_train_steps),
                'param_norm': zeros_scalar_like_params,
                'grad_ema': zeros_like_params,
                'grad_sq_ema': zeros_like_params,
                'update_ema': zeros_like_params,
                'update_sq_ema': zeros_like_params,
                'param_norms': zeros_timeseries_like_params
    }))

  def test_train_cost(self):
    """Ensure that the train cost is logged correctly."""
    init_fn, update_fn, _ = make_training_metrics(self.num_train_steps,
                                                  enable_train_cost=True)
    initial_metrics_state = init_fn(self.mock_params0)
    updated_metrics_state = update_fn(initial_metrics_state,
                                      0,
                                      self.mock_cost0,
                                      self.mock_grad1,
                                      self.mock_params0,
                                      self.mock_params1)
    updated_metrics_state = update_fn(updated_metrics_state,
                                      1,
                                      self.mock_cost1,
                                      self.mock_grad2,
                                      self.mock_params1,
                                      self.mock_params2)

    self.assertTrue(
        pytree_equal(
            updated_metrics_state['train_cost'],
            jnp.array([self.mock_cost0, self.mock_cost1, 0.0, 0.0, 0.0])))

  def test_update_param_norm(self):
    """Ensure that the training metrics updater updates param norm correctly."""

    init_fn, update_fn, _ = make_training_metrics(self.num_train_steps)
    initial_metrics_state = init_fn(self.mock_params0)
    updated_metrics_state = update_fn(initial_metrics_state, 0, self.mock_cost0,
                                      self.mock_grad1, self.mock_params0,
                                      self.mock_params1)
    self.assertTrue(
        pytree_equal(updated_metrics_state['param_norm'], {
            'foo': jnp.linalg.norm(self.mock_params0['foo']),
            'bar': {'baz': jnp.linalg.norm(self.mock_params0['bar']['baz'])}
        }))

    updated_metrics_state = update_fn(initial_metrics_state, 1, self.mock_cost1,
                                      self.mock_grad2, self.mock_params1,
                                      self.mock_params2)

    self.assertTrue(
        pytree_equal(updated_metrics_state['param_norm'], {
            'foo': jnp.linalg.norm(self.mock_params1['foo']),
            'bar': {'baz': jnp.linalg.norm(self.mock_params1['bar']['baz'])}
        }))

  def test_update_param_norms(self):
    """Ensure that we update param norms correctly."""

    init_fn, update_fn, _ = make_training_metrics(self.num_train_steps,
                                                  enable_param_norms=True)
    initial_metrics_state = init_fn(self.mock_params0)
    updated_metrics_state = update_fn(initial_metrics_state,
                                      0,
                                      self.mock_cost0,
                                      self.mock_grad1,
                                      self.mock_params0,
                                      self.mock_params1)
    updated_metrics_state = update_fn(updated_metrics_state,
                                      1,
                                      self.mock_cost1,
                                      self.mock_grad2,
                                      self.mock_params1,
                                      self.mock_params2)
    self.assertTrue(
        pytree_equal(
            updated_metrics_state['param_norms'], {
                'foo':
                    jnp.array([
                        jnp.linalg.norm(self.mock_params0['foo']),
                        jnp.linalg.norm(self.mock_params1['foo']),
                        0.0, 0.0, 0.0
                    ]),
                'bar': {
                    'baz':
                        jnp.array([
                            jnp.linalg.norm(self.mock_params0['bar']['baz']),
                            jnp.linalg.norm(self.mock_params1['bar']['baz']),
                            0.0, 0.0, 0.0
                        ])
                }
            }))

  def test_update_grad_ema(self):
    """Ensure that the training metrics updater updates grad ema correctly."""

    init_fn, update_fn, _ = make_training_metrics(self.num_train_steps,
                                                  enable_ema=True,
                                                  ema_beta=0.5)
    initial_metrics_state = init_fn(self.mock_params0)
    updated_metrics_state = update_fn(initial_metrics_state,
                                      0,
                                      self.mock_cost0,
                                      self.mock_grad1,
                                      self.mock_params0,
                                      self.mock_params1)
    updated_metrics_state = update_fn(updated_metrics_state,
                                      1,
                                      self.mock_cost1,
                                      self.mock_grad2,
                                      self.mock_params1,
                                      self.mock_params2)

    self.assertTrue(
        pytree_equal(
            updated_metrics_state['grad_ema'],
            jax.tree_map(lambda x, y, z: 0.25 * x + 0.25 * y + 0.5 * z,
                         self.mock_zeros, self.mock_grad1, self.mock_grad2)))

  def test_summarize(self):
    """Test the training metrics summarizer."""
    _, _, summarize_fn = make_training_metrics(self.num_train_steps,
                                               enable_train_cost=True,
                                               enable_ema=True)
    metrics_state = {
        'train_cost': jnp.array([1.0, 0.5, 0.25, 0.0, 0.0]),
        'param_norm': {
            'foo': 7.0,
            'bar': {'baz': 2.0}
        },
        'grad_ema': {
            'foo': 1 * jnp.ones(5),
            'bar': {'baz': 2 * jnp.ones(10)}
        },
        'grad_sq_ema': {
            'foo': 2 * jnp.ones(5),
            'bar': {'baz': 6 * jnp.ones(10)}
        },
        'update_ema': {
            'foo': 2 * jnp.ones(5),
            'bar': {'baz': 1 * jnp.ones(10)}
        },
        'update_sq_ema': {
            'foo': 6 * jnp.ones(5),
            'bar': {'baz': 2 * jnp.ones(10)}
        },
    }
    tree_summary = summarize_fn(metrics_state)
    self.assertTrue(
        pytree_equal(
            tree_summary, {
                'param_norm': {
                    '/foo': 7.0,
                    '/bar/baz': 2.0
                },
                'grad_var': {
                    '/foo': 5 * (2 - 1**2),
                    '/bar/baz': 10 * (6 - 2**2)
                },
                'update_var': {
                    '/foo': 5 * (6 - 2**2),
                    '/bar/baz': 10 * (2 - 1**2)
                },
                'update_ratio': {
                    '/foo': 5 * (6 - 2**2) / 7.0,
                    '/bar/baz': 10 * (2 - 1**2) / 2.0
                }
            }))

if __name__ == '__main__':
  absltest.main()
