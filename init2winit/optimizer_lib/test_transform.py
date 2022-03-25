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

"""Tests for transforms."""

from typing import NamedTuple

from absl.testing import absltest
import chex
from init2winit.optimizer_lib.kitchen_sink import transform_chain
import jax
import jax.numpy as jnp
import optax


def _optimizer_loop(optimizer, iterations=5):
  """Helper function for running optimizer loops."""
  params = {'w': jnp.ones((2,))}
  opt_state = optimizer.init(params)
  results = []
  for _ in range(iterations):
    compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
    grads = jax.grad(compute_loss)(params, jnp.array([5.0, 6.0]), 4.0)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    results.append(params)
  return results


class NesterovTest(chex.TestCase):
  """Test correctness of nesterov momentum."""

  def test_correctness(self):
    """Testing correctness via an independent flax.optim run."""

    target_solution = [
        {
            'w': jnp.array([0.40500003, 0.286])
        },
        {
            'w': jnp.array([0.255515, 0.106618])
        },
        {
            'w': jnp.array([0.31884143, 0.18260972])
        },
        {
            'w': jnp.array([0.40163627, 0.28196353])
        },
        {
            'w': jnp.array([0.43924114, 0.32708937])
        },
    ]
    optimizer = transform_chain(['nesterov'], [{
        'decay': 0.7
    }],
                                learning_rate=0.01)
    results = _optimizer_loop(optimizer)
    for target, result in zip(target_solution, results):
      chex.assert_trees_all_close(target, result)


class PolyakHBTest(chex.TestCase):
  """Test correctness of polyak_hb momentum."""

  def test_correctness(self):
    """Testing correctness via an independent flax.optim run."""

    target_solution = [
        {
            'w': jnp.array([0.65, 0.58000004])
        },
        {
            'w': jnp.array([0.26849997, 0.12220004])
        },
        {
            'w': jnp.array([0.09766498, -0.08280197])
        },
        {
            'w': jnp.array([0.17850482, 0.01420582])
        },
        {
            'w': jnp.array([0.38620475, 0.2634457])
        },
    ]
    optimizer = transform_chain(['polyak_hb'], [{
        'decay': 0.7
    }],
                                learning_rate=0.01)
    results = _optimizer_loop(optimizer)
    for target, result in zip(target_solution, results):
      chex.assert_trees_all_close(target, result)


class FirstMomentEMATest(chex.TestCase):
  """Test correctness of first_moment_ema."""

  def test_correctness(self):
    """Testing correctness via independent implementation."""

    def ema(decay, debias=True):

      def init_fn(params):
        del params
        return {'w': jnp.zeros((2,)), 'count': 0}

      def update_fn(updates, state, params=None):
        del params
        state['count'] += 1
        state['w'] = ((1 - decay) * updates['w'] + decay * state['w'])
        if debias:
          update = {'w': state['w'] / (1 - decay**state['count'])}
        else:
          update = {'w': state['w']}
        return update, state

      return optax.GradientTransformation(init_fn, update_fn)

    decay = 0.7
    learning_rate = 0.01
    true_ema = optax.chain(ema(decay), optax.scale(-1. * learning_rate))
    ks_ema = transform_chain(['first_moment_ema'], [{
        'decay': decay,
        'debias': True,
    }],
                             learning_rate=learning_rate)
    targets = _optimizer_loop(true_ema)
    results = _optimizer_loop(ks_ema)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class PreconditionByRMSTest(chex.TestCase):
  """Test correctness of precondition_by_rms."""

  def test_debias_false(self):
    rms_prop = optax.scale_by_rms()
    precondition_by_rms = transform_chain(['precondition_by_rms'], [{
        'eps': 0,
        'eps_root': 1e-8,
        'decay': 0.9,
        'debias': False
    }])
    targets = _optimizer_loop(rms_prop)
    results = _optimizer_loop(precondition_by_rms)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)

  def test_debias_true(self):
    adam = transform_chain(['scale_by_adam'], [{'b1': 0.0}])
    precondition_by_rms = transform_chain(['precondition_by_rms'], [{
        'debias': True
    }])
    targets = _optimizer_loop(adam)
    results = _optimizer_loop(precondition_by_rms)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class PreconditionByYogiTest(chex.TestCase):
  """Test correctness of precondition_by_yogi."""

  def test_with_0_momentum_yogi(self):
    optax_yogi = optax.yogi(learning_rate=1.0, b1=0.0, b2=0.9, eps=1e-8)
    precondition_by_yogi = transform_chain(['precondition_by_yogi'], [{
        'eps': 1e-8,
        'eps_root': 1e-6,
        'b2': 0.9,
        'debias': True
    }],
                                           learning_rate=1.0)
    targets = _optimizer_loop(optax_yogi)
    results = _optimizer_loop(precondition_by_yogi)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class TwistedAdamTest(chex.TestCase):
  """Test correctness of twisted_adam."""

  def test_correctness(self):
    """Testing correctness via independent implementation."""

    rms_decay = 0.9
    eps_root = 0.0
    eps = 1e-8
    moment_decay = 0.1

    class State(NamedTuple):
      nu: optax.Updates
      trace: optax.Params
      count: chex.Array

    def twisted_adam():

      def init_fn(params):
        return State(
            nu=jax.tree_map(jnp.zeros_like, params),
            trace=jax.tree_map(jnp.zeros_like, params),
            count=jnp.zeros([], jnp.int32))

      def update_fn(updates, state, params=None):
        del params
        count = state.count + jnp.array(1, jnp.int32)
        nu = {
            'w': (1 - rms_decay) * (updates['w']**2) + rms_decay * state.nu['w']
        }
        updates = {'w': updates['w'] / (jax.lax.sqrt(nu['w'] + eps_root) + eps)}

        updates = {'w': updates['w'] * jnp.sqrt((1 - rms_decay**count))}

        trace = {
            'w': (1 - moment_decay) * updates['w'] +
                 moment_decay * state.trace['w']
        }
        updates = {'w': trace['w']}

        updates = {'w': updates['w'] / (1 - moment_decay**count)}

        return updates, State(nu=nu, count=count, trace=trace)

      return optax.GradientTransformation(init_fn, update_fn)

    true_twisted_adam = twisted_adam()
    ks_twisted_adam = transform_chain(
        ['precondition_by_rms', 'first_moment_ema'], [
            {
                'decay': rms_decay,
                'eps': eps,
                'eps_root': eps_root,
                'debias': True
            },
            {
                'decay': moment_decay,
                'debias': True
            },
        ])

    targets = _optimizer_loop(true_twisted_adam)
    results = _optimizer_loop(ks_twisted_adam)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class AMSGradTest(chex.TestCase):
  """Test correctness of scale_by_amsgrad."""

  def test_correctness(self):
    """Testing correctness via optax.adam."""

    def amsgrad():
      adam = optax.scale_by_adam()

      def init_fn(params):
        return adam.init(params)

      def update_fn(updates, state, params=None):
        prev_nu = state.nu
        _, state = adam.update(updates, state, params)
        curr_nu = state.nu
        nu_hat = jax.tree_multimap(jnp.maximum, curr_nu, prev_nu)
        updates = jax.tree_multimap(lambda m, v: m / (jnp.sqrt(v + 0.0) + 1e-8),
                                    state.mu, nu_hat)

        return updates, optax.ScaleByAdamState(
            count=state.count, mu=state.mu, nu=nu_hat)

      return optax.GradientTransformation(init_fn, update_fn)

    true_amsgrad = amsgrad()
    ks_amsgrad = transform_chain(['scale_by_amsgrad'])

    targets = _optimizer_loop(true_amsgrad)
    results = _optimizer_loop(ks_amsgrad)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class EquivalenceTest(chex.TestCase):
  """Test equivalence of transform_chain and optax adagrad."""

  def test_adagrad(self):
    true_adagrad = optax.adagrad(0.7, initial_accumulator_value=0.3)
    ks_adagrad = transform_chain(['precondition_by_rss', 'first_moment_ema'], [{
        'initial_accumulator_value': 0.3
    }, {
        'decay': 0.0
    }],
                                 learning_rate=0.7)

    targets = _optimizer_loop(true_adagrad)
    results = _optimizer_loop(ks_adagrad)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class EqEMAHBTest(chex.TestCase):
  """Tests the equivalence of EMA vs HB. Both LR and WD need to scale."""

  def test_equivalence(self):
    hb = transform_chain(
        ['precondition_by_rms', 'polyak_hb', 'add_decayed_weights'], [{
            'decay': 0.3
        }, {
            'decay': 0.5
        }, {
            'weight_decay': 0.1
        }],
        learning_rate=1.0)
    ema = transform_chain(
        ['precondition_by_rms', 'first_moment_ema', 'add_decayed_weights'], [{
            'decay': 0.3
        }, {
            'decay': 0.5
        }, {
            'weight_decay': 0.05
        }],
        learning_rate=2.0)

    targets = _optimizer_loop(hb)
    results = _optimizer_loop(ema)

    for target, result in zip(targets, results):
      chex.assert_trees_all_close(target, result)


class LayeredBetaTest(chex.TestCase):
  """Tests precondition_by_layered_adaptive_rms."""

  def test_output_modality_1(self):
    decays = [0.19, 0.75, 1.0]
    scales = [0.9, 0.5, 1.0]
    decay_distribution = [0.34, 0.34, 0.32]
    ks_opt = transform_chain(['precondition_by_layered_adaptive_rms'], [{
        'decays': decays,
        'scales': scales,
        'decay_distribution': decay_distribution,
        'eps_root': 0.0
    }],
                             learning_rate=1.0)
    scales = jnp.array([0.9, 0.5, 1.0, 1.0])
    betas = jnp.array([0.19, 0.75, 1.0, 1.0])
    one_minus_betas = jnp.array([0.81, 0.25, 1.0, 1.0])
    params = {'w': jnp.ones((4,))}
    opt_state = ks_opt.init(params)
    # step 1
    grads = {'w': 2 * jnp.ones((4,))}
    true_nu = one_minus_betas * (grads['w']**2)
    true_updates = {
        'w': -1.0 * jnp.array(scales) * grads['w'] / jnp.sqrt(true_nu)
    }
    opt_updates, opt_state = ks_opt.update(grads, opt_state)
    chex.assert_trees_all_close(true_updates, opt_updates)
    params = optax.apply_updates(params, opt_updates)
    # step2
    grads = {'w': jnp.ones((4,))}
    true_nu = one_minus_betas * (grads['w']**2) + betas * true_nu
    true_updates = {
        'w': -1.0 * jnp.array(scales) * grads['w'] / jnp.sqrt(true_nu)
    }
    opt_updates, opt_state = ks_opt.update(grads, opt_state)
    chex.assert_trees_all_close(true_updates, opt_updates)


if __name__ == '__main__':
  absltest.main()
