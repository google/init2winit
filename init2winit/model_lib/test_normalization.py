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

"""Tests for normalization.py."""

import functools
from absl.testing import absltest

from flax import linen as nn
from init2winit.model_lib import model_utils
from init2winit.model_lib import normalization

import jax
import jax.numpy as jnp
import numpy as np

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def _init(flax_module, rng, input_shape):
  model_init_fn = jax.jit(
      functools.partial(flax_module.init, use_running_average=False))
  xs = np.zeros(input_shape)
  init_dict = model_init_fn({'params': rng}, xs)
  params = init_dict['params']
  batch_stats = init_dict['batch_stats']
  return params, batch_stats


def _run_sync_ema(
    vbn_flax_module,
    input_shape,
    x,
    virtual_means,
    sync_frequency,
    num_steps):
  """Run num_steps forward passes, syncing state every sync_frequency steps."""
  rng = jax.random.PRNGKey(0)
  vbn_params, vbn_state = _init(vbn_flax_module, rng, input_shape)
  for t in range(num_steps):
    # Note that in trainer.py, we synchronize after calling `model(x)`, but
    # the ordering below tests the same functionality and simplifies our
    # calculations.
    if t % sync_frequency == 0:
      vbn_state = model_utils.sync_local_batch_norm_stats(vbn_state)
    _, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']
  # Always do a final sync at the end of training.
  vbn_state = model_utils.sync_local_batch_norm_stats(vbn_state)

  # Given that we have the same mean/variance for each step (and assuming
  # `momentum=0.9`), we expect each EMA at step t to have value:
  #      0.9^t * x_0 + \sum_{j=0}^{t-1} (0.9^j * 0.1 * mean)
  #    = 0.9^t * x_0 + 0.1 * mean * \sum_{j=0}^{t-1} 0.9^j
  #    = 0.9^t * x_0 + 0.1 * mean * (1 - 0.9^t) / (1 - 0.9)
  #    = 0.9^t * x_0 + mean * (1 - 0.9^t)
  #
  # This is without synchronization. We synchronize the EMAs every
  # `sync_frequency` steps, and we can treat x_0 as the synchronized value.
  ema_means = jnp.zeros((3,))  # The EMAs for means are initialized to zero.
  for _ in range(num_steps // sync_frequency):
    x_0 = jnp.mean(ema_means) * jnp.ones((3,))
    ema_means = (
        0.9 ** sync_frequency * x_0 +
        virtual_means * (1 - 0.9 ** sync_frequency))
  feature_size = vbn_state['batch_norm_running_mean'].shape[-1]
  # Always do a final sync at the end of training.
  ema_means = jnp.mean(ema_means) * jnp.ones((3,))
  expected_synced_ema_means = jnp.stack(
      [m * jnp.ones((feature_size,)) for m in ema_means])

  np.testing.assert_allclose(
      vbn_state['batch_norm_running_mean'],
      expected_synced_ema_means,
      atol=1e-4)
  return vbn_state


class NormalizationTest(absltest.TestCase):
  """Tests for virtual batch norm."""

  def test_batch_norm(self):
    """Test virtual BN recovers BN when the virtual size equals batch size."""
    rng = jax.random.PRNGKey(0)
    batch_size = 10
    feature_size = 7
    input_shape = (batch_size, 3, 3, feature_size)
    half_input_shape = (batch_size // 2, 3, 3, feature_size)
    twos = 2.0 * jnp.ones(half_input_shape)
    nines = 9.0 * jnp.ones(half_input_shape)
    x = jnp.concatenate((twos, nines))

    bn_flax_module = nn.BatchNorm(momentum=0.9)
    bn_params, bn_state = _init(bn_flax_module, rng, input_shape)

    vbn_flax_module = normalization.VirtualBatchNorm(
        momentum=0.9, virtual_batch_size=batch_size, data_format='NHWC')
    vbn_params, vbn_state = _init(vbn_flax_module, rng, input_shape)

    _, bn_state = bn_flax_module.apply(
        {'params': bn_params, 'batch_stats': bn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    bn_state = bn_state['batch_stats']
    bn_y, bn_state = bn_flax_module.apply(
        {'params': bn_params, 'batch_stats': bn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    bn_state = bn_state['batch_stats']

    _, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']
    vbn_y, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']

    # Test that the layer forward passes are the same.
    np.testing.assert_allclose(bn_y, vbn_y, atol=1e-4)

    # Test that virtual and regular BN produce the same EMAs.
    np.testing.assert_allclose(
        bn_state['mean'],
        np.squeeze(vbn_state['batch_norm_running_mean'], 0),
        atol=1e-4)
    np.testing.assert_allclose(
        bn_state['var'],
        np.squeeze(vbn_state['batch_norm_running_var'], 0),
        atol=1e-4)

  def test_forward_pass(self):
    """Test that two calls are the same as one with twice the batch size."""
    rng = jax.random.PRNGKey(0)
    batch_size = 10
    feature_size = 7
    input_shape = (batch_size, 3, 3, feature_size)
    half_input_shape = (batch_size // 2, 3, 3, feature_size)
    twos = 2.0 * jnp.ones(half_input_shape)
    threes = 3.0 * jnp.ones(half_input_shape)
    fives = 5.0 * jnp.ones(half_input_shape)
    nines = 9.0 * jnp.ones(half_input_shape)
    # The mean(x1) = 2.5, stddev(x1) = 0.5 so we expect
    # `(x1 - mean(x1)) / stddev(x1)` to be half -1.0, then half 1.0.
    x1 = jnp.concatenate((twos, threes))
    # The mean(x2) = 7.0, stddev(x2) = 2.0 so we expect
    # `(x2 - mean(x2)) / stddev(x2)` to be half -1.0, then half 1.0.
    x2 = jnp.concatenate((fives, nines))
    x_both = jnp.concatenate((x1, x2))

    expected_bn_y = jnp.concatenate(
        (jnp.ones(half_input_shape) * -1.0, jnp.ones(half_input_shape)))

    bn_flax_module = nn.BatchNorm(momentum=0.9)
    bn_params, bn_state = _init(bn_flax_module, rng, input_shape)

    vbn_flax_module = normalization.VirtualBatchNorm(
        momentum=0.9, virtual_batch_size=batch_size, data_format='NHWC')
    vbn_params, vbn_state = _init(vbn_flax_module, rng, input_shape)

    bn_y1, _ = bn_flax_module.apply(
        {'params': bn_params, 'batch_stats': bn_state},
        x1,
        mutable=['batch_stats'],
        use_running_average=False)

    bn_y2, _ = bn_flax_module.apply(
        {'params': bn_params, 'batch_stats': bn_state},
        x2,
        mutable=['batch_stats'],
        use_running_average=False)

    bn_y_both, _ = bn_flax_module.apply(
        {'params': bn_params, 'batch_stats': bn_state},
        x_both,
        mutable=['batch_stats'],
        use_running_average=False)

    vbn_y_both, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x_both,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']

    # Test that the layer forward passes behave as expected.
    np.testing.assert_allclose(bn_y1, expected_bn_y, atol=1e-4)
    np.testing.assert_allclose(bn_y2, expected_bn_y, atol=1e-4)
    np.testing.assert_allclose(
        vbn_y_both, jnp.concatenate((bn_y1, bn_y2)), atol=1e-4)
    # Test that the virtual batch norm and nn.BatchNorm layers do not perform
    # the same calculation on the concatenated batch.
    # There is no negative of `np.testing.assert_allclose` so we test that the
    # diff is greater than zero.
    np.testing.assert_array_less(
        -jnp.abs(vbn_y_both - bn_y_both), jnp.zeros_like(vbn_y_both))

    _, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x_both,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']

    # The mean running average stats at 0.0, and the variance starts at 1.0. So
    # after two applications of the same batch we should expect the value to be
    # mean_ema = 0.9 * (0.9 * 0.0 + 0.1 * mean) + 0.1 * mean = 0.19 * mean
    # var_ema = 0.9 * (0.9 * 1.0 + 0.1 * var) + 0.1 * var = 0.19 * var + 0.81
    expected_mean_ema_x1 = 0.19 * jnp.mean(x1) * jnp.ones((feature_size,))
    expected_mean_ema_x2 = 0.19 * jnp.mean(x2) * jnp.ones((feature_size,))
    expected_mean_ema_both = jnp.stack(
        (expected_mean_ema_x1, expected_mean_ema_x2))
    expected_var_ema_x1 = (
        0.19 * jnp.std(x1) ** 2.0 + 0.81) * jnp.ones((feature_size,))
    expected_var_ema_x2 = (
        0.19 * jnp.std(x2) ** 2.0 + 0.81) * jnp.ones((feature_size,))
    expected_var_ema_both = jnp.stack(
        (expected_var_ema_x1, expected_var_ema_x2))
    np.testing.assert_allclose(
        vbn_state['batch_norm_running_mean'],
        expected_mean_ema_both,
        atol=1e-4)
    np.testing.assert_allclose(
        vbn_state['batch_norm_running_var'],
        expected_var_ema_both,
        atol=1e-4)

  def test_batch_norm_sync(self):
    """Test syncing the multiple per-device EMAs.

    In a multi-host setting, we have to sync the BN stats, and in addition to
    syncing across hosts, we also need to sync the multiple EMAs on each
    device, which is what we test here (we ignore the `lax.pmean` in
    `sync_batchnorm_stats`).
    """
    n = 10
    # Make 3 per-device EMAs, one for each pair of constant arrays.
    virtual_batch_size = n * 2
    batch_size = n * 6
    feature_size = 7
    input_shape = (batch_size, 3, 3, feature_size)
    shape = (n, 3, 3, feature_size)
    ones = jnp.ones(shape)
    twos = 2.0 * jnp.ones(shape)
    threes = 3.0 * jnp.ones(shape)
    fives = 5.0 * jnp.ones(shape)
    sevens = 7.0 * jnp.ones(shape)
    tens = 10.0 * jnp.ones(shape)
    x = jnp.concatenate((ones, twos, threes, fives, sevens, tens))
    # Calculate the means/variances of each virtual batch.
    virtual_means = jnp.array([1.5, 4.0, 8.5])

    vbn_flax_module = normalization.VirtualBatchNorm(
        momentum=0.9, virtual_batch_size=virtual_batch_size, data_format='NHWC')
    vbn_state_sync_5 = _run_sync_ema(
        vbn_flax_module,
        input_shape,
        x,
        virtual_means,
        sync_frequency=5,
        num_steps=20)
    vbn_state_sync_10 = _run_sync_ema(
        vbn_flax_module,
        input_shape,
        x,
        virtual_means,
        sync_frequency=10,
        num_steps=20)
    np.testing.assert_allclose(
        vbn_state_sync_5['batch_norm_running_mean'],
        vbn_state_sync_10['batch_norm_running_mean'],
        atol=1e-4)
    np.testing.assert_allclose(
        vbn_state_sync_5['batch_norm_running_var'],
        vbn_state_sync_10['batch_norm_running_var'],
        atol=1e-4)

  def test_different_eval_batch_size(self):
    """Test virtual BN can use a different batch size for evals."""
    rng = jax.random.PRNGKey(0)
    batch_size = 10
    feature_size = 7
    input_shape = (batch_size, 3, 3, feature_size)
    x = 2.0 * jnp.ones(input_shape)

    vbn_flax_module = normalization.VirtualBatchNorm(
        momentum=0.9, virtual_batch_size=batch_size, data_format='NHWC')
    vbn_params, vbn_state = _init(vbn_flax_module, rng, input_shape)

    _, vbn_state = vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        x,
        mutable=['batch_stats'],
        use_running_average=False)
    vbn_state = vbn_state['batch_stats']

    vbn_flax_module.apply(
        {'params': vbn_params, 'batch_stats': vbn_state},
        jnp.ones((13, 3, 3, feature_size)),
        use_running_average=True)


if __name__ == '__main__':
  absltest.main()
