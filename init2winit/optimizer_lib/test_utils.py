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

"""Tests for utils."""


from absl.testing import absltest
import chex
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib.utils import extract_field
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
    chex.assert_type(extract_field(optimizer_state, 'count'), int)
    # Test that we can extract 'nu'.
    chex.assert_shape(extract_field(optimizer_state, 'nu')['foo'], (10,))
    # Test that we can extract 'mu'.
    chex.assert_shape(extract_field(optimizer_state, 'mu')['foo'], (10,))
    # Test that attemptping to extract a nonexistent field "abc" returns None.
    chex.assert_equal(extract_field(optimizer_state, 'abc'), None)


if __name__ == '__main__':
  absltest.main()
