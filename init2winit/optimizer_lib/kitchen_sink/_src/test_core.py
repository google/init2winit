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

"""Tests for transform_chain."""

from absl.testing import absltest
import chex
import flax
import flax.linen as nn
from init2winit.optimizer_lib.kitchen_sink._src.core import kitchen_sink
import jax
import jax.numpy as jnp
import ml_collections
import optax

# pylint:disable=g-long-lambda


class KitchenSinkSuccessTest(absltest.TestCase):
  """Test transform_chain."""

  def setUp(self):
    super().setUp()
    self.basic_sink = kitchen_sink({
        '0': {
            'element': 'nesterov'
        },
        '1': {
            'element': 'polyak_hb'
        },
    })

  def test_construction(self):
    self.assertTrue(self.basic_sink)

  def test_dummy_step(self):
    """Test dummy step."""
    num_weights = 100
    xs = jnp.ones((num_weights,))
    ys = 1

    optimizer = kitchen_sink({
        '0': {
            'element': 'nesterov'
        },
        '1': {
            'element': 'polyak_hb'
        },
    })
    params = {'w': jnp.ones((num_weights,))}
    opt_state = optimizer.init(flax.core.FrozenDict(params))

    compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x),
                                                      jnp.array(y))
    grads = jax.grad(compute_loss)(params, xs, ys)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    self.assertTrue(params)


class KitchenSinkFailTest(absltest.TestCase):
  """Test transform_chain exceptional behavior."""

  def test_bad_hp(self):
    self.assertRaises(TypeError, kitchen_sink,
                      {'0': {
                          'element': 'nesterov',
                          'hps': {
                              'asdf': 1e-4
                          }
                      }})

  def test_bad_transform_name(self):
    self.assertRaises(ValueError, kitchen_sink,
                      {'0': {
                          'element': 'rasputin'
                      }})


class KitchenSinkMaskTest(absltest.TestCase):
  """Test transform_chain with mask."""

  def test_no_op(self):
    """Test no-op."""
    optimizer = kitchen_sink({
        '0': {
            'element': 'nesterov',
            'mask': lambda p: jax.tree_map(lambda x: x.ndim != 1, p)
        }
    })
    params = {'w': jnp.array([1, 2, 3])}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state, params)

    chex.assert_trees_all_close(params, update)

  def test_mask_dim_1(self):
    """Test mask dimension."""
    optimizer = kitchen_sink({
        '0': {
            'element': 'nesterov',
            'mask': lambda p: jax.tree_map(lambda x: x.ndim != 1, p)
        }
    })
    params = {'w': jnp.array([1, 2, 3]), 'b': jnp.ones((2, 2))}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state, params)

    optimizer2 = kitchen_sink({'0': {'element': 'nesterov'}})
    params2 = {'b': jnp.ones((2, 2))}
    state2 = optimizer2.init(params2)
    update2, state2 = optimizer2.update(params2, state2, params2)

    chex.assert_trees_all_close(update['b'], update2['b'])
    chex.assert_trees_all_close(update['w'], params['w'])


class FromHParamsTest(chex.TestCase):
  """Test construction from opt_hparams ml_collections.ConfigDict."""

  def test_empty_element(self):
    self.assertRaises(ValueError, kitchen_sink,
                      ml_collections.ConfigDict({'0': {}}))

  def test_empty_hps(self):
    kitchen_sink(
        ml_collections.ConfigDict({'0': {
            'element': 'nesterov'
        }}))

  def test_equal_structs(self):
    """Test that initial and updated states have the same tree structure."""
    tx = kitchen_sink(
        ml_collections.ConfigDict({'0': {
            'element': 'nesterov'
        }}))
    params = {'a': 1.}
    gradients = {'a': 2.}
    state = tx.init(params)
    _, new_state = tx.update(gradients, state)
    chex.assert_trees_all_equal_structs(state, new_state)

  def test_empty_mask(self):
    kitchen_sink(
        ml_collections.ConfigDict(
            {'0': {
                'element': 'nesterov',
                'hps': {
                    'decay': 0.9
                }
            }}))

  def test_one_minus(self):
    """Test that we appropriately process / remove  hps."""
    tx = kitchen_sink(
        ml_collections.ConfigDict(
            {'0': {
                'element': 'nesterov',
                'hps': {
                    'decay': 0.9
                }
            }}))
    tx_one_minus = kitchen_sink(
        ml_collections.ConfigDict(
            {'0': {
                'element': 'nesterov',
                'hps': {
                    'one_minus_decay': 0.1
                }
            }}))

    params = {'a': 1.}

    state = tx.init(params)
    updates, state = tx.update(params, state, params)
    result = optax.apply_updates(params, updates)

    state = tx_one_minus.init(params)
    updates, state = tx_one_minus.update(params, state, params)
    result_one_minus = optax.apply_updates(params, updates)

    chex.assert_trees_all_equal(result, result_one_minus)


class GraftCombinatorTest(chex.TestCase):
  """Test grafting."""

  def test_1d(self):
    """Test grafting."""
    x = 10.
    lr = 0.01

    g = kitchen_sink(
        {
            'join': {
                'mag_chain': {
                    'element': 'sgd',
                    'hps': {
                        'learning_rate': -1.
                    },
                },
                'dir_chain': {
                    'element': 'sgd',
                    'hps': {
                        'learning_rate': -2.
                    },
                }
            },
            'by': 'grafting',
        },
        learning_rate=lr)
    state = g.init(x)
    for _ in range(10):
      grad_fn = jax.value_and_grad(lambda x: x**2)
      _, grad = grad_fn(x)
      updates, state = g.update(grad, state)
      dx = 2 * x
      x -= lr * dx
      self.assertAlmostEqual(updates, -lr * dx, places=4)

  def test_nn(self):
    seed = 33
    key = jax.random.PRNGKey(seed)

    class SimpleNN(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(features=100)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(features=100)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

    x = jax.random.normal(key, [20, 2])
    y = jax.random.normal(key, [20, 1])
    lr = 0.001
    model = SimpleNN()
    params = model.init(key, x)

    m = optax.sgd(learning_rate=-1., momentum=0.9)
    d = optax.sgd(learning_rate=-2., momentum=0.9)
    g = kitchen_sink(
        {
            'join': {
                'mag_chain': {
                    'element': 'sgd',
                    'hps': {
                        'learning_rate': -1.,
                        'momentum': 0.9
                    },
                },
                'dir_chain': {
                    'element': 'sgd',
                    'hps': {
                        'learning_rate': -2.,
                        'momentum': 0.9
                    },
                }
            },
            'by': 'grafting',
        },
        learning_rate=lr)
    s_m = m.init(params)
    s_d = d.init(params)
    s_g = g.init(params)

    def loss_fn(params):
      yhat = model.apply(params, x)
      loss = jnp.sum((y - yhat)**2)
      return loss

    for _ in range(10):
      grad_fn = jax.value_and_grad(loss_fn)
      _, grad = grad_fn(params)
      u_m, s_m = m.update(grad, s_m)
      u_d, s_d = d.update(grad, s_d)
      u_g, s_g = g.update(grad, s_g)

      u_m_n = jax.tree_map(jnp.linalg.norm, u_m)
      u_d_n = jax.tree_map(jnp.linalg.norm, u_d)
      u_g2 = jax.tree_map(lambda m, d, dn: -lr * d / (dn + 1e-6) * m, u_m_n,
                          u_d, u_d_n)

      chex.assert_trees_all_close(u_g, u_g2)


if __name__ == '__main__':
  absltest.main()
