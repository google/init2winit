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

"""Unit tests for model_debugger.py.

"""

import os
import shutil
import tempfile

from absl.testing import absltest
import flax
from flax import linen as nn
from init2winit import checkpoint
from init2winit import utils
from init2winit.hessian import model_debugger
from init2winit.hessian.model_debugger import skip_bwd
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
    rep_rng = flax.jax_utils.replicate(rng)
    metrics = debugger.full_eval(
        10, params=variables['params'], grad=variables['params'], rng=rep_rng)
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
    rep_rng = flax.jax_utils.replicate(rng)
    xs = np.random.normal(size=(1, 10, 5))

    lin_model = Linear()
    params = lin_model.init(rng, xs[0])['params']  # init with unsharded batch
    rep_params = flax.jax_utils.replicate(params)

    def apply_on_batch(params, batch_stats, batch, **apply_kwargs):
      del batch_stats
      return lin_model.apply({'params': params}, batch, **apply_kwargs)

    get_act_stats_fn = model_debugger.create_forward_pass_stats_fn(
        apply_on_batch,
        capture_activation_norms=True,
        sown_collection_names=['qvalues'])

    debugger = model_debugger.ModelDebugger(
        forward_pass=get_act_stats_fn, use_pmap=True)

    metrics = debugger.full_eval(
        step=0, params=rep_params, batch=xs, rng=rep_rng)

    expected_output = np.dot(xs[0], params['Dense_0']['kernel'])
    expected_q_value = np.linalg.norm(expected_output)**2 / expected_output.size
    expected_c_value = model_debugger.cvalue(expected_output)
    expected_output_norm = np.linalg.norm(
        expected_output)**2 / expected_output.size
    expected_input_norm = float(np.linalg.norm(xs))**2 / xs.size
    expected_keys = [
        'qvalues', 'intermediate_qvalue', 'intermediate_cvalue', 'step',
        'param_norms_sql2', 'global_param_norm_sql2'
    ]

    self.assertEqual(set(expected_keys), set(metrics.keys()))

    self.assertAlmostEqual(
        float(expected_q_value),
        float(metrics['intermediate_qvalue']['__call__'][0]),
        places=5)
    self.assertAlmostEqual(
        float(expected_c_value),
        float(metrics['intermediate_cvalue']['__call__'][0]),
        places=5)
    self.assertAlmostEqual(
        expected_input_norm,
        float(metrics['qvalues']['residualq'][0]),
        places=5)
    self.assertAlmostEqual(
        expected_output_norm,
        float(metrics['qvalues']['residualq'][1]),
        places=5)

  def test_skip_analysis(self):
    """Test that we can selectively turn off layers in the backward pass."""
    # Define a model that multiplies input by 3 * 2 * 2
    rng = jax.random.PRNGKey(0)

    class A(nn.Module):

      @skip_bwd
      @nn.compact
      def __call__(self, x):
        y = B()(x)
        z = C()(y)
        return z

    class B(nn.Module):

      @skip_bwd
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(
            1, kernel_init=nn.initializers.constant(3), use_bias=False)(
                x)
        y = C()(x)
        return y

    class C(nn.Module):

      @skip_bwd
      @nn.compact
      def __call__(self, x):
        x = nn.Dense(
            1, kernel_init=nn.initializers.constant(2), use_bias=False)(
                x)
        return x

    init_rng = jax.random.PRNGKey(0)
    x = jnp.ones(1)

    _, vs = A().apply({}, x, rngs={'params': init_rng}, mutable=True)
    ps = vs['params']

    def fake_loss_fn(ps, x, rng, module_flags=None):
      if module_flags is None:
        vs = {'params': ps}
      else:
        vs = {'params': ps, 'moduleflags': module_flags['moduleflags']}

      out = A().apply(vs, x, rngs={'dropout_rng': rng})
      return jnp.sum(out)

    grad_fn = jax.grad(fake_loss_fn)

    debugger = model_debugger.ModelDebugger(
        use_pmap=True,
        grad_fn=grad_fn,
        skip_flags=['B_0/C_0'],
        skip_groups=['test_group'])
    rep_params = flax.jax_utils.replicate(ps)
    rep_rng = flax.jax_utils.replicate(rng)
    rep_x = flax.jax_utils.replicate(x)
    metrics = debugger.full_eval(
        step=10, params=rep_params, rng=rep_rng, batch=rep_x)
    expected_keys = [
        'step', 'global_param_norm_sql2', 'param_norms_sql2', 'grad_norms_sql2',
        'global_grad_norm_sql2', 'skip_analysis'
    ]
    self.assertEqual(set(expected_keys), set(metrics.keys()))
    skip_dict = metrics['skip_analysis']

    # B_0/C_0 comes from skip_flags = ['B_0/C_0']. 'test_group' comes from
    # skip_groups = ['test_group']. We don't list all keys turned off via
    # test_group.
    expected_keys = ['B_0/C_0', 'unmodified_gradient', 'test_group']
    self.assertEqual(set(expected_keys), set(skip_dict.keys()))

    skipped = skip_dict['B_0/C_0']

    self.assertEqual(skipped['B_0']['C_0']['Dense_0']['kernel'], 0.0)
    self.assertEqual(skipped['B_0']['Dense_0']['kernel'], 4.0)
    self.assertEqual(skipped['C_0']['Dense_0']['kernel'], 36.0)

    skipped = skip_dict['test_group']

    self.assertEqual(skipped['B_0']['C_0']['Dense_0']['kernel'], 0.0)
    self.assertEqual(skipped['B_0']['Dense_0']['kernel'], 1.0)
    self.assertEqual(skipped['C_0']['Dense_0']['kernel'], 0)

    og = skip_dict['unmodified_gradient']

    self.assertEqual(og['B_0']['C_0']['Dense_0']['kernel'], 36.0)
    self.assertEqual(og['B_0']['Dense_0']['kernel'], 16.0)
    self.assertEqual(og['C_0']['Dense_0']['kernel'], 36.0)

  def test_model_debugger_pmap(self):
    """Test training for two epochs on MNIST with a small model."""

    rep_variables = set_up_cnn()

    pytree_path = os.path.join(self.test_dir, 'metrics')
    metrics_logger = utils.MetricLogger(
        pytree_path=pytree_path, events_dir=self.test_dir)

    # Fake grad_fn for testing.
    def grad_fn(params, batch, rng):
      del params, batch, rng
      return rep_variables['params']

    debugger = model_debugger.ModelDebugger(
        use_pmap=True, grad_fn=grad_fn, metrics_logger=metrics_logger)

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
        grad=None,  # use internal gradient comp
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

    loaded_metrics = checkpoint.load_checkpoint(metrics_file)['pytree']

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

    # Test restore of prior metrics.
    new_debugger = model_debugger.ModelDebugger(
        use_pmap=True, metrics_logger=metrics_logger)
    metrics = new_debugger.full_eval(
        10,
        params=rep_variables['params'],
        grad=rep_variables['params'],
        extra_scalar_metrics=extra_metrics2)
    self.assertEqual(
        new_debugger.stored_metrics['param_norms_sql2']['Conv_0']
        ['kernel'].shape, (3,))


if __name__ == '__main__':
  absltest.main()
