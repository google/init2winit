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

"""Unit tests for model_debugger.py.

"""

import os
import shutil
import tempfile

from absl.testing import absltest
import flax
from flax import linen as nn
import checkpoint as checkpoint  # local file import
import hessian.model_debugger as model_debugger  # local file import
import utils as utils  # local file import
import jax.numpy as jnp
import jax.random
import numpy as np

N_CLASSES = 10


def normal_like(x):
  return np.random.normal(size=x.shape)


class TestDense(nn.Module):
  """Layer to test the model_debugger.skip functionality."""

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(features=256)(x)
    y = nn.relu(y)
    out = x + y

    return out


class Linear(nn.Module):
  """A simple linear model."""

  @nn.compact
  def __call__(self, x, train=False):
    y = nn.Dense(features=10, kernel_init=nn.initializers.ones)(x)
    model_debugger.tag_residual_activations(self, x, y)
    return y


def set_up_cnn(seed=0, batch_size=10, replicate=True):
  rng = jax.random.PRNGKey(seed)
  batch = get_batch(batch_size, shard=True)
  cnn = CNN()
  variables = cnn.init(rng, batch[0][0])  # init with unsharded batch
  if replicate:
    rep_variables = flax.jax_utils.replicate(variables)
    return rep_variables


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)

    # return intermediate output
    x = TestDense()(x)

    x = nn.Dense(features=N_CLASSES)(x)
    return x


def get_batch(batch_size, shard=False):
  x = jnp.ones((batch_size, 28, 28, 1))
  x = normal_like(x)
  y = jnp.zeros((batch_size, N_CLASSES))
  if shard:
    x = jnp.expand_dims(x, axis=0)
    y = jnp.expand_dims(y, axis=0)
  return (x, y)


class ModelDebuggerTest(absltest.TestCase):
  """Tests training for 2 epochs on MNIST."""

  def setUp(self):
    super(ModelDebuggerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(ModelDebuggerTest, self).tearDown()

  def test_model_debugger(self):
    """Test training for two epochs on MNIST with a small model."""
    rng = jax.random.PRNGKey(0)
    batch_size = 10
    batch = get_batch(batch_size)
    cnn = CNN()
    variables = cnn.init(rng, batch[0])

    debugger = model_debugger.ModelDebugger(use_pmap=False)
    metrics = debugger.full_eval(
        10, params=variables['params'], grad=variables['params'])
    expected_keys = [
        'step',
        'global_param_norm_sql2',
        'param_norms_sql2',
        'grad_norms_sql2',
        'global_grad_norm_sql2',
    ]
    self.assertEqual(set(expected_keys), set(metrics.keys()))

  def test_create_forward_pass_stats_fn(self):
    """Test that we properly capture intermediate values."""
    rng = jax.random.PRNGKey(0)
    xs = np.random.normal(size=(1, 10, 5))

    lin_model = Linear()
    params = lin_model.init(rng, xs[0])['params']  # init with unsharded batch
    rep_params = flax.jax_utils.replicate(params)

    get_act_stats_fn = model_debugger.create_forward_pass_stats_fn(
        lin_model, capture_activation_norms=True,
        sown_collection_names=['residual_activations'])

    debugger = model_debugger.ModelDebugger(
        forward_pass=get_act_stats_fn,
        use_pmap=True)

    metrics = debugger.full_eval(step=0, params=rep_params, batch=xs)

    expected_output = np.dot(xs[0], params['Dense_0']['kernel'])
    expected_output_norm = np.linalg.norm(expected_output)
    expected_input_norm = float(np.linalg.norm(xs))

    self.assertAlmostEqual(expected_output_norm,
                           metrics['intermediate_norms']['__call__'])
    self.assertAlmostEqual(
        expected_input_norm,
        float(metrics['residual_activations']['residual'][0]), places=5)
    self.assertAlmostEqual(
        expected_output_norm,
        float(metrics['residual_activations']['residual'][1]), places=5)

  def test_model_debugger_pmap(self):
    """Test training for two epochs on MNIST with a small model."""

    rep_variables = set_up_cnn()

    pytree_path = os.path.join(self.test_dir, 'metrics')
    metrics_logger = utils.MetricLogger(
        pytree_path=pytree_path,
        events_dir=self.test_dir)
    debugger = model_debugger.ModelDebugger(
        use_pmap=True, metrics_logger=metrics_logger)

    # eval twice to test the concat
    extra_metrics = {'train_loss': 1.0}
    extra_metrics2 = {'train_loss': 1.0}
    metrics = debugger.full_eval(
        10,
        params=rep_variables['params'],
        grad=rep_variables['params'],
        extra_scalar_metrics=extra_metrics)
    metrics = debugger.full_eval(
        10,
        params=rep_variables['params'],
        grad=rep_variables['params'],
        extra_scalar_metrics=extra_metrics2)
    expected_keys = [
        'step',
        'global_param_norm_sql2',
        'param_norms_sql2',
        'grad_norms_sql2',
        'global_grad_norm_sql2',
        'train_loss',
    ]

    metrics_file = os.path.join(self.test_dir, 'metrics/training_metrics')

    loaded_metrics = checkpoint.load_checkpoint(
        metrics_file)['pytree']

    self.assertEqual(set(expected_keys), set(metrics.keys()))
    expected_shape = ()
    self.assertEqual(metrics['global_grad_norm_sql2'].shape, expected_shape)
    # Test stored metrics is concatenated.
    expected_shape = (2,)
    self.assertEqual(loaded_metrics['global_grad_norm_sql2'].shape,
                     expected_shape)

    # check param norms were saved correctly
    self.assertEqual(
        loaded_metrics['param_norms_sql2']['Conv_0']['kernel'].shape, (2,))
    self.assertEqual(loaded_metrics['train_loss'][0], 1.0)


if __name__ == '__main__':
  absltest.main()
