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

"""Unit tests for hyperparameters.py."""

from absl.testing import absltest
from init2winit import hyperparameters
import tensorflow.compat.v1 as tf


class HyperParameterTest(absltest.TestCase):
  """Tests hyperparameter overrides."""

  def test_override(self):
    """Test polynomial schedule works correctly."""
    hps_overrides = {
        'lr_hparams': {
            'schedule': 'polynomial',
            'power': 2.0,
            'base_lr': 0.1,
            'end_factor': 0.01,
            'decay_steps_factor': 0.5,
        },
    }

    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides,
    )

    self.assertEqual(merged_hps.lr_hparams['schedule'], 'polynomial')
    self.assertEqual(
        set(merged_hps.lr_hparams.keys()),
        set(
            ['schedule', 'power', 'base_lr', 'end_factor', 'decay_steps_factor']
        ),
    )

  def test_unrecognized_override(self):
    """Test overriding with unrecognized hparams."""
    # Sadly, if we try 'lr_hparams.base_lrTYPO' no exception will be raised.
    # We currently do not detect issues nested inside the lr_hparams or other
    # nested hparam overrides.
    hps_overrides = {'lr_hparamsTYPO.base_lr': 77.0}
    with self.assertRaises(KeyError):
      hyperparameters.build_hparams(
          model_name='transformer',
          initializer_name='noop',
          dataset_name='lm1b_v2',
          hparam_file=None,
          hparam_overrides=hps_overrides,
          allowed_unrecognized_hparams=[],
      )
    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides,
        allowed_unrecognized_hparams=['lr_hparamsTYPO'],
    )
    expected_added_field = {'base_lr': 77.0}
    self.assertEqual(merged_hps.lr_hparamsTYPO.to_dict(), expected_added_field)

  def test_dot_override(self):
    """Test overriding lr_hparams.base_lr works correctly."""
    hps_overrides = {'lr_hparams.base_lr': 77.0}

    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides,
    )

    self.assertEqual(
        merged_hps.lr_hparams['schedule'], 'rsqrt_normalized_decay_warmup'
    )
    expected_lr_hparams = {
        'base_lr': 77.0,
        'warmup_steps': 1000,
        'squash_steps': 1000,
        'schedule': 'rsqrt_normalized_decay_warmup',
    }
    self.assertEqual(
        set(merged_hps.lr_hparams.keys()),
        set(['schedule', 'warmup_steps', 'base_lr', 'squash_steps']),
    )
    self.assertEqual(merged_hps.lr_hparams.to_dict(), expected_lr_hparams)

  def test_optimizer_override(self):
    """Test polynomial schedule works correctly."""
    hps_overrides = {
        'optimizer': 'nesterov',
        'opt_hparams': {
            'momentum': 0.42,
        },
        'lr_hparams': {
            'schedule': 'polynomial',
            'power': 2.0,
            'base_lr': 0.1,
            'end_factor': 0.01,
            'decay_steps_factor': 0.5,
        },
    }

    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides,
    )

    self.assertEqual(merged_hps.lr_hparams['schedule'], 'polynomial')
    self.assertEqual(
        set(merged_hps.lr_hparams.keys()),
        set(
            ['schedule', 'power', 'base_lr', 'end_factor', 'decay_steps_factor']
        ),
    )

  def test_expand_dot_keys(self):
    """Test expanding keys with dots in them into sub-dicts."""
    d = {
        'a.b.c': 1,
        'a.b.d': 2,
        'e.a.b.c': 3,
        'e.a.b.d': 4,
        'e.b.d': 6,
        'j': 5,
    }
    expanded = hyperparameters.expand_dot_keys(d)
    print(expanded)
    expected = {
        'a': {
            'b': {
                'c': 1,
                'd': 2,
            },
        },
        'e': {
            'a': {
                'b': {
                    'c': 3,
                    'd': 4,
                },
            },
            'b': {
                'd': 6,
            },
        },
        'j': 5,
    }
    self.assertEqual(expanded, expected)

  def test_unsafe_expand_dot_keys(self):
    """Test that we will raise an error when unsafe keys are present."""
    d = {
        'a': 0,
        'a.b.c': 1,
        'a.b': 2,
        'e.f.g': 3,
        'e.f.g.h': 4,
        'k': 5,
    }
    expected_msg = (
        'Aborting dotted key expansion as prefix of dotted key is not a dict:'
        ' prefix = a, prefix_value = 0'
    )
    with self.assertRaisesWithLiteralMatch(ValueError, expected_msg):
      hyperparameters.expand_dot_keys(d)

  def test_safe_dict_merge_expand_dot_keys(self):
    """Test dotted keys can be safely expanded when there's common prefix."""
    d = {'a.b': {'c': 8}, 'a': {'d': 4}}

    expected = {'a': {'d': 4, 'b': {'c': 8}}}
    expanded = hyperparameters.expand_dot_keys(d)

    self.assertEqual(expanded, expected)

  def test_unsafe_dict_merge_expand_dot_keys(self):
    """Test that we will raise an error when unsafe keys are present."""
    d = {'a.b': {'c': 8}, 'a.b.c': 5}

    expected_msg = 'prefix = a.b.c already exists with value = 8'
    with self.assertRaisesWithLiteralMatch(ValueError, expected_msg):
      hyperparameters.expand_dot_keys(d)

  def test_safe_expand_dot_keys(self):
    """Test that dotted key expansion works when prefix is dict."""
    d = {
        'lr_hparams': {
            'warmup_steps': 1000,
            'schedule': 'cosine_warmup',
        },
        'lr_hparams.base_lr': 1.0,
    }

    expanded = hyperparameters.expand_dot_keys(d)

    expected = {
        'lr_hparams': {
            'base_lr': 1.0,
            'warmup_steps': 1000,
            'schedule': 'cosine_warmup',
        },
    }

    self.assertEqual(expanded, expected)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
