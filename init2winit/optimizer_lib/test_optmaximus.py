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
from init2winit.optimizer_lib.optmaximus import kitchen_sink
import jax
import jax.numpy as jnp
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
    # TODO(dsuo): refactor out setup code
    num_weights = 100
    xs = jnp.ones((num_weights,))
    ys = 1

    optimizer = kitchen_sink(['nesterov', 'polyak_hb'], [{}, {}])
    params = {'w': jnp.ones((num_weights,))}
    opt_state = optimizer.init(params)

    compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
    grads = jax.grad(compute_loss)(params, xs, ys)

    updates, opt_state = optimizer.update(grads, opt_state)
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
    optimizer = kitchen_sink(
        ['nesterov'], masks=[lambda p: jax.tree_map(lambda x: x.ndim != 1, p)])
    params = {'w': jnp.array([1, 2, 3])}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state)

    chex.assert_tree_all_close(params, update)

  def test_mask_dim_1(self):
    """Test mask dimension."""
    optimizer = kitchen_sink(
        ['nesterov'], masks=[lambda p: jax.tree_map(lambda x: x.ndim != 1, p)])
    params = {'w': jnp.array([1, 2, 3]), 'b': jnp.ones((2, 2))}
    state = optimizer.init(params)
    update, state = optimizer.update(params, state)

    optimizer2 = kitchen_sink(['nesterov'])
    params2 = {'b': jnp.ones((2, 2))}
    state2 = optimizer2.init(params2)
    update2, state2 = optimizer2.update(params2, state2)

    chex.assert_tree_all_close(update['b'], update2['b'])
    chex.assert_tree_all_close(update['w'], params['w'])


if __name__ == '__main__':
  absltest.main()
