# coding=utf-8
# Copyright 2023 The init2winit Authors.
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
from flax.core import freeze
from init2winit.hessian.precondition import make_diag_preconditioner
from init2winit.shared_test_utilities import pytree_equal
from init2winit.training_metrics_grabber import make_training_metrics
from init2winit.utils import total_tree_norm_l2
from init2winit.utils import total_tree_norm_sql2
from init2winit.utils import total_tree_sum
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from optax import InjectHyperparamsState
from optax import ScaleByAdamState


FLAGS = flags.FLAGS


class TrainingMetricsGrabberTest(absltest.TestCase):
  """Tests the logged statistics from training_metrics_grabber."""

  def setUp(self):
    super(TrainingMetricsGrabberTest, self).setUp()
    self.mock_params0 = freeze({'foo': jnp.zeros(5),
                                'bar': {'baz': jnp.ones(10)}})
    self.mock_batch_stats = freeze({'foo': jnp.zeros(5),
                                    'bar': {'baz': jnp.ones(10)}})
    self.mock_grad1 = freeze({'foo': -jnp.ones(5),
                              'bar': {'baz': -jnp.ones(10)}})
    self.mock_grad2 = freeze({'foo': -2*jnp.ones(5),
                              'bar': {'baz': -2*jnp.ones(10)}})

    self.mock_nu0 = freeze({'foo': 2*jnp.ones(5),
                            'bar': {'baz': 3*jnp.ones(10)}})
    self.mock_nu1 = freeze({'foo': 3*jnp.ones(5),
                            'bar': {'baz': 4*jnp.ones(10)}})

    self.mock_optimizer_state0 = InjectHyperparamsState(
        0,
        None,
        (ScaleByAdamState(0, None, self.mock_nu0),))

    self.mock_optimizer_state1 = InjectHyperparamsState(
        0,
        None,
        (ScaleByAdamState(1, None, self.mock_nu1),))

    self.mock_cost0 = 1.0
    self.mock_cost1 = 0.5
    self.mock_cost2 = 0.25

    self.step_size = 1.0
    self.num_train_steps = 5

    # Simulate running GD with step size 1.
    self.mock_params1 = jax.tree_map(lambda p, g: p - self.step_size * g,
                                     self.mock_params0,
                                     self.mock_grad1)
    self.mock_params2 = jax.tree_map(lambda p, g: p - self.step_size * g,
                                     self.mock_params1,
                                     self.mock_grad2)

    self.mock_zeros = freeze({'foo': jnp.zeros(5),
                              'bar': {'baz': jnp.zeros(10)}})

  def test_init(self):
    """Test the training metrics initializer."""

    zeros_like_params = jax.tree_map(jnp.zeros_like, self.mock_params0)
    zeros_timeseries = jnp.zeros(self.num_train_steps)
    zeros_timeseries_like_params = jax.tree_map(
        lambda x: jnp.zeros(self.num_train_steps), self.mock_params0)

    # Test init with everything disabled.
    init_fn, _, _ = make_training_metrics(self.num_train_steps,
                                          ConfigDict({}))
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    self.assertTrue(
        pytree_equal({'param_norm': zeros_timeseries},
                     initial_metrics_state))

    # Test init with enable_ema = True and enable_train_cost=True.
    init_fn, _, _ = make_training_metrics(self.num_train_steps,
                                          ConfigDict({}),
                                          enable_ema=True,
                                          enable_train_cost=True,
                                          enable_param_norms=True,
                                          enable_gradient_norm=True,
                                          enable_all_gradient_norms=True,
                                          enable_update_norm=True,
                                          enable_update_norms=True,
                                          enable_batch_stats_norm=True,
                                          enable_all_batch_stats_norms=True)
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    self.assertTrue(pytree_equal(initial_metrics_state, {
                'train_cost': zeros_timeseries,
                'param_norm': zeros_timeseries,
                'grad_ema': zeros_like_params,
                'grad_sq_ema': zeros_like_params,
                'update_ema': zeros_like_params,
                'update_sq_ema': zeros_like_params,
                'param_norms': zeros_timeseries_like_params,
                'gradient_norm': zeros_timeseries,
                'all_gradient_norms': zeros_timeseries_like_params,
                'update_norm': zeros_timeseries,
                'update_norms': zeros_timeseries_like_params,
                'batch_stats_norm': zeros_timeseries,
                'all_batch_stats_norms': zeros_timeseries_like_params,},))

  def test_train_cost(self):
    """Ensure that the train cost is logged correctly."""
    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps, ConfigDict({}), enable_train_cost=True
    )
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(
        initial_metrics_state,
        0,
        self.mock_cost0,
        self.mock_grad1,
        self.mock_params0,
        self.mock_params1,
        self.mock_optimizer_state0,
        self.mock_batch_stats,
    )
    updated_metrics_state = update_fn(
        updated_metrics_state,
        1,
        self.mock_cost1,
        self.mock_grad2,
        self.mock_params1,
        self.mock_params2,
        self.mock_optimizer_state1,
        self.mock_batch_stats,
    )

    self.assertTrue(
        pytree_equal(
            updated_metrics_state['train_cost'],
            jnp.array([self.mock_cost0, self.mock_cost1, 0.0, 0.0, 0.0]),
        )
    )

  def test_update_param_norm(self):
    """Ensure that the training metrics updater updates param norm correctly."""

    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps,
        ConfigDict({}),
    )
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(
        initial_metrics_state,
        0,
        self.mock_cost0,
        self.mock_grad1,
        self.mock_params0,
        self.mock_params1,
        self.mock_optimizer_state0,
        self.mock_batch_stats,
    )

    expected_param_norm = jnp.zeros(5)
    expected_param_norm1 = jnp.zeros(5)
    expected_param_norm = expected_param_norm.at[0].set(total_tree_norm_l2(
        self.mock_params0))
    expected_param_norm1 = expected_param_norm1.at[0].set(total_tree_norm_l2(
        self.mock_params0))
    expected_param_norm1 = expected_param_norm1.at[1].set(total_tree_norm_l2(
        self.mock_params1))

    self.assertTrue(
        pytree_equal(
            updated_metrics_state['param_norm'],
            expected_param_norm,
        )
    )

    updated_metrics_state = update_fn(
        updated_metrics_state,
        1,
        self.mock_cost1,
        self.mock_grad2,
        self.mock_params1,
        self.mock_params2,
        self.mock_optimizer_state1,
        self.mock_batch_stats,
    )
    self.assertTrue(
        pytree_equal(
            updated_metrics_state['param_norm'],
            expected_param_norm1,
        )
    )

  def test_update_param_norms(self):
    """Ensure that we update param norms correctly."""

    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps, ConfigDict({}), enable_param_norms=True
    )
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(
        initial_metrics_state,
        0,
        self.mock_cost0,
        self.mock_grad1,
        self.mock_params0,
        self.mock_params1,
        self.mock_optimizer_state0,
        self.mock_batch_stats,
    )
    updated_metrics_state = update_fn(
        updated_metrics_state,
        1,
        self.mock_cost1,
        self.mock_grad2,
        self.mock_params1,
        self.mock_params2,
        self.mock_optimizer_state1,
        self.mock_batch_stats,
    )
    self.assertTrue(
        pytree_equal(
            updated_metrics_state['param_norms'],
            freeze({
                'foo': jnp.array([
                    jnp.linalg.norm(self.mock_params0['foo']),
                    jnp.linalg.norm(self.mock_params1['foo']),
                    0.0,
                    0.0,
                    0.0,
                ]),
                'bar': {
                    'baz': jnp.array([
                        jnp.linalg.norm(self.mock_params0['bar']['baz']),
                        jnp.linalg.norm(self.mock_params1['bar']['baz']),
                        0.0,
                        0.0,
                        0.0,
                    ])
                },
            }),
        )
    )

  def test_update_update_norms(self):
    """Ensure that we update gradient and update norms correctly."""
    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps,
        ConfigDict({}),
        enable_gradient_norm=True,
        enable_update_norm=True,
        enable_update_norms=True,
    )
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(
        initial_metrics_state,
        0,
        self.mock_cost0,
        self.mock_grad1,
        self.mock_params0,
        self.mock_params1,
        self.mock_optimizer_state0,
        self.mock_batch_stats,
    )
    updated_metrics_state = update_fn(
        updated_metrics_state,
        1,
        self.mock_cost1,
        self.mock_grad2,
        self.mock_params1,
        self.mock_params2,
        self.mock_optimizer_state1,
        self.mock_batch_stats,
    )
    self.assertTrue(
        pytree_equal(
            updated_metrics_state['update_norms'],
            freeze({
                'foo': jnp.array([
                    self.step_size * jnp.linalg.norm(self.mock_grad1['foo']),
                    self.step_size * jnp.linalg.norm(self.mock_grad2['foo']),
                    0.0,
                    0.0,
                    0.0,
                ]),
                'bar': {
                    'baz': jnp.array([
                        self.step_size
                        * jnp.linalg.norm(self.mock_grad1['bar']['baz']),
                        self.step_size
                        * jnp.linalg.norm(self.mock_grad2['bar']['baz']),
                        0.0,
                        0.0,
                        0.0,
                    ])
                },
            }),
        )
    )

    self.assertEqual(
        updated_metrics_state['update_norm'][0],
        total_tree_norm_l2(self.mock_grad1),
    )
    self.assertEqual(
        updated_metrics_state['update_norm'][1],
        total_tree_norm_l2(self.mock_grad2),
    )

    self.assertEqual(
        updated_metrics_state['update_norm'][0],
        self.step_size * total_tree_norm_l2(self.mock_grad1),
    )
    self.assertEqual(
        updated_metrics_state['update_norm'][1],
        self.step_size * total_tree_norm_l2(self.mock_grad2),
    )

  def test_update_grad_ema(self):
    """Ensure that the training metrics updater updates grad ema correctly."""

    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps, ConfigDict({}), enable_ema=True, ema_beta=0.5
    )
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(
        initial_metrics_state,
        0,
        self.mock_cost0,
        self.mock_grad1,
        self.mock_params0,
        self.mock_params1,
        self.mock_optimizer_state0,
        self.mock_batch_stats,
    )
    updated_metrics_state = update_fn(
        updated_metrics_state,
        1,
        self.mock_cost1,
        self.mock_grad2,
        self.mock_params1,
        self.mock_params2,
        self.mock_optimizer_state1,
        self.mock_batch_stats,
    )

    self.assertTrue(
        pytree_equal(
            updated_metrics_state['grad_ema'],
            jax.tree_map(
                lambda x, y, z: 0.25 * x + 0.25 * y + 0.5 * z,
                self.mock_zeros,
                self.mock_grad1,
                self.mock_grad2,
            ),
        )
    )

  def test_optstate_sumsq(self):
    """Test that optstate sumsq and sumsq are computed correctly."""
    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps,
        ConfigDict({}),
        optstate_sumsq_fields=['nu'],
        optstate_sum_fields=['nu'])
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    self.assertTrue(pytree_equal(
        initial_metrics_state['optstate_sumsq'], {
            'nu': jnp.zeros(self.num_train_steps)
        }
    ))
    self.assertTrue(pytree_equal(
        initial_metrics_state['optstate_sum'], {
            'nu': jnp.zeros(self.num_train_steps)
        }
    ))
    updated_metrics_state = update_fn(initial_metrics_state,
                                      0,
                                      self.mock_cost0,
                                      self.mock_grad1,
                                      self.mock_params0,
                                      self.mock_params1,
                                      self.mock_optimizer_state0,
                                      self.mock_batch_stats)
    updated_metrics_state = update_fn(updated_metrics_state,
                                      1,
                                      self.mock_cost1,
                                      self.mock_grad2,
                                      self.mock_params1,
                                      self.mock_params2,
                                      self.mock_optimizer_state1,
                                      self.mock_batch_stats)

    self.assertEqual(updated_metrics_state['optstate_sumsq']['nu'][0],
                     total_tree_norm_sql2(self.mock_nu0))
    self.assertEqual(updated_metrics_state['optstate_sumsq']['nu'][1],
                     total_tree_norm_sql2(self.mock_nu1))

    self.assertEqual(updated_metrics_state['optstate_sum']['nu'][0],
                     total_tree_sum(self.mock_nu0))
    self.assertEqual(updated_metrics_state['optstate_sum']['nu'][1],
                     total_tree_sum(self.mock_nu1))

  def test_update_precondition(self):
    """Test that precondition metrics are computed correctly."""
    optimizer = 'adam'
    opt_hparams = ConfigDict({'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8})

    init_fn, update_fn, _ = make_training_metrics(
        self.num_train_steps,
        ConfigDict({
            'optimizer': optimizer,
            'opt_hparams': opt_hparams
        }),
        enable_preconditioner_normsq=True,
        enable_semip_grad_normsq=True)
    initial_metrics_state = init_fn(self.mock_params0, self.mock_batch_stats)
    updated_metrics_state = update_fn(initial_metrics_state,
                                      0,
                                      self.mock_cost0,
                                      self.mock_grad1,
                                      self.mock_params0,
                                      self.mock_params1,
                                      self.mock_optimizer_state0,
                                      self.mock_batch_stats)
    updated_metrics_state = update_fn(updated_metrics_state,
                                      1,
                                      self.mock_cost1,
                                      self.mock_grad2,
                                      self.mock_params1,
                                      self.mock_params2,
                                      self.mock_optimizer_state1,
                                      self.mock_batch_stats)

    pre0 = make_diag_preconditioner(optimizer, opt_hparams,
                                    self.mock_optimizer_state0, ConfigDict({}))
    semip_grad_0 = jax.tree_map(lambda g, p: g / (p**0.5), self.mock_grad1,
                                pre0)

    pre1 = make_diag_preconditioner(optimizer, opt_hparams,
                                    self.mock_optimizer_state1, ConfigDict({}))
    semip_grad_1 = jax.tree_map(lambda g, p: g / (p**0.5), self.mock_grad2,
                                pre1)

    self.assertEqual(updated_metrics_state['preconditioner_normsq'][0],
                     total_tree_norm_sql2(pre0))
    self.assertEqual(updated_metrics_state['preconditioner_normsq'][1],
                     total_tree_norm_sql2(pre1))

    self.assertEqual(updated_metrics_state['semip_grad_normsq'][0],
                     total_tree_norm_sql2(semip_grad_0))
    self.assertEqual(updated_metrics_state['semip_grad_normsq'][1],
                     total_tree_norm_sql2(semip_grad_1))

  def test_summarize(self):
    """Test the training metrics summarizer."""
    _, _, summarize_fn = make_training_metrics(self.num_train_steps,
                                               ConfigDict({}),
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
