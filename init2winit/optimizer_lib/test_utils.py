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

"""Tests for utils."""


from absl.testing import absltest
import chex
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib import utils
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict

# pylint:disable=duplicate-key


class ExtractFieldTest(chex.TestCase):
  """Test the extract_field() function."""

  def test_adam(self):
    init_fn, update_fn = optimizers.get_optimizer(
        ConfigDict({
            'optimizer': 'adam',
            'l2_decay_factor': None,
            'batch_size': 50,
            'total_accumulated_batch_size': 100,  # Use gradient accumulation.
            'opt_hparams': {
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-7,
                'weight_decay': 0.0,
            }
        }))
    del update_fn
    optimizer_state = init_fn({'foo': jnp.ones(10)})
    # Test that we can extract 'count'.
    chex.assert_type(utils.extract_field(optimizer_state, 'count'), int)
    # Test that we can extract 'nu'.
    chex.assert_shape(utils.extract_field(optimizer_state, 'nu')['foo'], (10,))
    # Test that we can extract 'mu'.
    chex.assert_shape(utils.extract_field(optimizer_state, 'mu')['foo'], (10,))
    # Test that attemptping to extract a nonexistent field "abc" returns None.
    chex.assert_equal(utils.extract_field(optimizer_state, 'abc'), None)


class GradientAggregationDecoratorTest(chex.TestCase):
  """Test the requires_gradient_aggregation() decorator."""

  def test_no_aggregation(self):
    """Tests behavior with the decorator."""
    @utils.no_cross_device_gradient_aggregation
    def dummy_update_fn(updates, state, params):
      del updates, state, params

    self.assertFalse(utils.requires_gradient_aggregation(dummy_update_fn))

  def test_with_aggregation(self):
    """Tests the default behavior."""
    def dummy_update_fn(updates, state, params):
      del updates, state, params

    self.assertTrue(utils.requires_gradient_aggregation(dummy_update_fn))


if __name__ == '__main__':
  absltest.main()
