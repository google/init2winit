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

"""Unit tests for trainer.py.

"""

import shutil
import tempfile

from absl.testing import absltest
import flax
from flax import linen as nn
import hessian.model_debugger as model_debugger  # local file import
import jax.numpy as jnp
import jax.random
import numpy as np


def normal_like(x):
  return np.random.normal(size=x.shape)


class TestDense(nn.Module):
  """Used to get nested tree structure in the variable tree."""

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=256)(x)
    model_debugger.tag_layer(self, x, 2)
    return x


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    # inject an input from a variable collection
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    model_debugger.tag_layer(self, x, 0)
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    model_debugger.tag_layer(self, x, 1)
    x = x.reshape((x.shape[0], -1))  # flatten

    # return intermediate output
    x = TestDense()(x)

    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

config = {
    'param_norms': True,
    'grad_norms': False,
    'update_norms': False,
    'layer_jacobians': False,
    'lanczos_n_iter': 30,
    'use_pmap': False,
    'verbose': True,
    'cmap': True,
}


def forward_pass(params, inj, inputs, injects):
  if inj:
    mutable = ['fn_out', 'fn_inp', 'inputs']
  else:
    mutable = ['fn_out', 'fn_inp']
  return CNN().apply({
      'params': params,
      'inputs': injects
  },
                     inputs,
                     mutable=mutable)


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
    x = jnp.ones((batch_size, 28, 28, 1))
    cnn = CNN()
    variables = cnn.init(rng, x)

    debugger = model_debugger.ModelDebugger(forward_pass, normal_like(x),
                                            config)
    metrics = debugger.full_eval(
        10, params=variables['params'], grad=variables['params'])
    expected_keys = [
        'step', 'c0', 'c1', 'c2', 'param_norm', 'param_norms', 'grad_norm',
        'grad_norms'
    ]
    self.assertEqual(set(expected_keys), set(metrics.keys()))

  def test_model_debugger_pmap(self):
    """Test training for two epochs on MNIST with a small model."""
    rng = jax.random.PRNGKey(0)
    batch_size = 10
    x = jnp.ones((batch_size, 28, 28, 1))
    cnn = CNN()
    variables = cnn.init(rng, x)
    sharded_x = jnp.ones((1, batch_size, 28, 28, 1))

    pmap_config = config.copy()
    pmap_config['use_pmap'] = True
    rep_variables = flax.jax_utils.replicate(variables)
    debugger = model_debugger.ModelDebugger(forward_pass,
                                            normal_like(sharded_x), pmap_config)
    metrics = debugger.full_eval(
        10, params=rep_variables['params'], grad=rep_variables['params'])
    expected_keys = [
        'step', 'c0', 'c1', 'c2', 'param_norm', 'param_norms', 'grad_norm',
        'grad_norms'
    ]
    self.assertEqual(set(expected_keys), set(metrics.keys()))
    expected_shape = (100,)
    for i in range(3):
      key = 'c{}'.format(i)
      self.assertEqual(metrics[key].shape, expected_shape)

if __name__ == '__main__':
  absltest.main()
