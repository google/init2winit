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

"""Tests for kitchen_sink."""

from absl.testing import absltest
import chex
import flax
import flax.linen as nn
from init2winit.optimizer_lib.optmaximus import from_hparams
from init2winit.optimizer_lib.optmaximus import kitchen_sink
import jax
import jax.numpy as jnp
import ml_collections
import optax

# pylint:disable=g-long-lambda


class KitchenSinkSuccessTest(absltest.TestCase):
  """Test kitchen_sink."""

  def test_construction(self):
    self.assertTrue(kitchen_sink(['nesterov', 'polyak_hb'], [{}, {}]))

  def test_construction_no_hps(self):
    self.assertTrue(kitchen_sink(['nesterov', 'polyak_hb']))

  def test_dummy_step(self):
    """Test dummy step."""
    num_weights = 100
    xs = jnp.ones((num_weights,))
    ys = 1

    optimizer = kitchen_sink(['nesterov', 'polyak_hb'], [{}, {}])
    params = {'w': jnp.ones((num_weights,))}
    opt_state = optimizer.init(flax.core.FrozenDict(params))

    compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
    grads = jax.grad(compute_loss)(params, xs, ys)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    self.assertTrue(params)


class KitchenSinkFailTest(absltest.TestCase):
  """Test kitchen_sink exceptional behavior."""

  def test_bad_hp(self):
    self.assertRaises(TypeError, kitchen_sink, ['nesterov', 'polyak_hb'], [{
        'asdf': 1e-4
    }, {}])

  def test_incompatible_length_hp(self):
    self.assertRaises(ValueError, kitchen_sink, ['nesterov'], [{}, {}])

  def test_incompatible_length_mask(self):
    self.assertRaises(ValueError, kitchen_sink, ['nesterov'], [{}, {}], [])

  def test_bad_transform_name(self):
    self.assertRaises(KeyError, kitchen_sink, ['Rasputin'], [{}])


class KitchenSinkMaskTest(absltest.TestCase):
  """Test kitchen_sink with mask."""

  def test_no_op(self):
    """Test no-op."""
    optimizer = kitchen_sink(
        ['nesterov'], masks=[lambda p: jax.tree_map(lambda x: x.ndim != 1, p)])
    params = {'w': jnp.array([1, 2, 3])}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state, params)

    chex.assert_trees_all_close(params, update)

  def test_mask_dim_1(self):
    """Test mask dimension."""
    optimizer = kitchen_sink(
        ['nesterov'], masks=[lambda p: jax.tree_map(lambda x: x.ndim != 1, p)])
    params = {'w': jnp.array([1, 2, 3]), 'b': jnp.ones((2, 2))}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state, params)

    optimizer2 = kitchen_sink(['nesterov'])
    params2 = {'b': jnp.ones((2, 2))}
    state2 = optimizer2.init(params2)
    update2, state2 = optimizer2.update(params2, state2, params2)

    chex.assert_trees_all_close(update['b'], update2['b'])
    chex.assert_trees_all_close(update['w'], params['w'])


class FromHParamsTest(chex.TestCase):
  """Test construction from opt_hparams ml_collections.ConfigDict."""

  def test_empty_element(self):
    self.assertRaises(KeyError, from_hparams,
                      ml_collections.ConfigDict({'0': {}}))

  def test_empty_hps(self):
    from_hparams(ml_collections.ConfigDict({'0': {'element': 'nesterov'}}))

  def test_empty_mask(self):
    from_hparams(
        ml_collections.ConfigDict(
            {'0': {
                'element': 'nesterov',
                'hps': {
                    'decay': 0.9
                }
            }}))

  def test_one_minus(self):
    """Test that we appropriately process / remove one_minus_ hps."""
    tx = from_hparams(
        ml_collections.ConfigDict(
            {'0': {
                'element': 'nesterov',
                'hps': {
                    'decay': 0.9
                }
            }}))
    tx_one_minus = from_hparams(
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

  def test_add_decayed_weights(self):
    """Test no mask gets added for add_decayed_weights."""
    tx_no_mask = from_hparams(
        ml_collections.ConfigDict({
            '0': {
                'element': 'nesterov',
                'hps': {
                    'one_minus_decay': 0.1,
                }
            },
            '1': {
                'element': 'add_decayed_weights',
                'hps': {
                    'weight_decay': 1e-4
                }
            }
        }))
    tx_none_mask = from_hparams(
        ml_collections.ConfigDict({
            '0': {
                'element': 'nesterov',
                'hps': {
                    'one_minus_decay': 0.1,
                }
            },
            '1': {
                'element': 'add_decayed_weights',
                'hps': {
                    'weight_decay': 1e-4,
                    'mask': None
                }
            }
        }))

    params = {'a': 1.}
    state = tx_no_mask.init(params)
    updates, state = tx_no_mask.update(params, state, params)
    result_no_mask = optax.apply_updates(params, updates)

    state = tx_none_mask.init(params)
    updates, state = tx_none_mask.update(params, state, params)
    result_none_mask = optax.apply_updates(params, updates)

    chex.assert_trees_all_equal(result_no_mask, result_none_mask)

  @chex.variants(with_jit=True, without_jit=True, with_pmap=True)
  def test_add_decayed_weights_with_mask(self):
    """Test mask is not added for add_decayed_weights if specified in hps."""

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

    tx = from_hparams(
        ml_collections.ConfigDict({
            '0': {
                'element': 'add_decayed_weights',
                'hps': {
                    'weight_decay': 1e-4,
                    'mask': 'bias_bn'
                }
            }
        }))
    key = jax.random.PRNGKey(0)
    x = jnp.ones((5, 4, 4, 3))
    y = jax.random.uniform(key, (5, 4, 4, 7))

    foo_vars = flax.core.unfreeze(Foo(filters=7, train=True).init(key, x))

    @self.variant
    def train_step(params, x, y):
      y1, new_batch_stats = Foo(
          filters=7, train=True).apply(
              params, x, mutable=['batch_stats'])

      return jnp.abs(y - y1).sum(), new_batch_stats

    state = self.variant(tx.init)(foo_vars['params'])
    grads, _ = jax.grad(train_step, has_aux=True)(foo_vars, x, y)
    updates, state = self.variant(tx.update)(dict(grads['params']), state,
                                             foo_vars['params'])

    chex.assert_trees_all_close(updates['BatchNorm_0'],
                                grads['params']['BatchNorm_0'])


if __name__ == '__main__':
  absltest.main()
