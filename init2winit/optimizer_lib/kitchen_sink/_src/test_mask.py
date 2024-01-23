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

"""Tests for utils."""

from typing import Sequence

from absl.testing import absltest
import chex
import flax
import flax.linen as nn
from init2winit.optimizer_lib.kitchen_sink._src.mask import create_mask
from init2winit.optimizer_lib.kitchen_sink._src.mask import create_weight_decay_mask
import jax
import jax.numpy as jnp
import optax

# pylint:disable=duplicate-key


class Foo(nn.Module):
  """Dummy model."""

  train: bool
  filters: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.filters, (1, 1), use_bias=False, dtype=jnp.float32)(x)
    x = nn.BatchNorm(
        use_running_average=not self.train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=jnp.float32)(
            x)
    return x


class Bar(nn.Module):
  """Dummy model."""

  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, use_bias=True, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x


class CreateMaskTest(chex.TestCase):
  """Test masking."""

  def test_simple(self):
    """Check if the leaf key is `a`."""
    mask = create_mask(lambda path, _: path[-1] == 'a')
    data = {'a': 4, 'b': {'a': 5, 'c': 1}, 'c': {'a': {'b': 1}}}

    truth = {'a': True, 'b': {'a': True, 'c': False}, 'c': {'a': {'b': False}}}

    chex.assert_equal(mask(data), truth)


class CreateWeightDecayMaskTest(chex.TestCase):
  """Test weight decay mask."""

  def test_simple(self):
    """Check that the correct tags are removed."""
    mask = create_weight_decay_mask()
    data = {
        'bias': {
            'b': 4
        },
        'bias': {
            'BatchNorm_0': 4,
            'bias': 5,
            'a': 0
        },
        'BatchNorm_0': {
            'b': 4
        },
        'a': {
            'b': {
                'BatchNorm_0': 0,
                'bias': 0
            },
            'c': 0
        }
    }
    truth = {
        'bias': {
            'b': False
        },
        'bias': {
            'BatchNorm_0': False,
            'bias': False,
            'a': False
        },
        'BatchNorm_0': {
            'b': False
        },
        'a': {
            'b': {
                'BatchNorm_0': True,
                'bias': False
            },
            'c': True
        }
    }

    chex.assert_equal(mask(data), truth)

  @chex.variants(with_jit=True, without_jit=True)
  def test_batch(self):
    """Test that batch layer is indeed ignored.

    Code taken from: https://github.com/google/flax/issues/932
    """
    key = jax.random.PRNGKey(0)
    x = jnp.ones((5, 4, 4, 3))
    y = jax.random.uniform(key, (5, 4, 4, 7))

    foo_vars = flax.core.unfreeze(Foo(filters=7, train=True).init(key, x))
    tx = optax.masked(optax.adam(1e-7), create_weight_decay_mask())

    @self.variant
    def train_step(params, x, y):
      y1, new_batch_stats = Foo(
          filters=7, train=True).apply(
              params, x, mutable=['batch_stats'])

      return jnp.abs(y - y1).sum(), new_batch_stats

    state = self.variant(tx.init)(foo_vars['params'])
    grads, _ = jax.grad(train_step, has_aux=True)(foo_vars, x, y)
    updates, state = self.variant(tx.update)(dict(grads['params']), state)

    chex.assert_trees_all_close(updates['BatchNorm_0'],
                                grads['params']['BatchNorm_0'])

  @chex.variants(with_jit=True, without_jit=True)
  def test_bias(self):
    """Test that biases are ignored."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
    x = jax.random.uniform(key1, (4, 4))

    model = Bar(features=[3, 4, 5])
    params = flax.core.unfreeze(model.init(key2, x))
    y = jax.random.uniform(key1, model.apply(params, x).shape)

    tx = optax.masked(optax.adam(1e-7), create_weight_decay_mask())
    state = tx.init(params)

    def loss(params, x, y):
      pred = model.apply(params, x)
      return jnp.abs(pred - y).sum()

    grads = jax.grad(loss)(params, x, y)
    updates, state = self.variant(tx.update)(dict(grads), state)

    for i in range(3):
      chex.assert_trees_all_close(grads['params'][f'layers_{i}']['bias'],
                                  updates['params'][f'layers_{i}']['bias'])


if __name__ == '__main__':
  absltest.main()
