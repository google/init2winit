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
            'base_lr': .1,
            'end_factor': .01,
            'decay_steps_factor': 0.5,
        },
    }

    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides)

    self.assertEqual(merged_hps.lr_hparams['schedule'],
                     'polynomial')
    self.assertEqual(
        set(merged_hps.lr_hparams.keys()),
        set([
            'schedule', 'power', 'base_lr',
            'end_factor', 'decay_steps_factor'
        ]))

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
            'base_lr': .1,
            'end_factor': .01,
            'decay_steps_factor': 0.5,
        },
    }

    merged_hps = hyperparameters.build_hparams(
        model_name='transformer',
        initializer_name='noop',
        dataset_name='lm1b_v2',
        hparam_file=None,
        hparam_overrides=hps_overrides)

    self.assertEqual(merged_hps.lr_hparams['schedule'],
                     'polynomial')
    self.assertEqual(
        set(merged_hps.lr_hparams.keys()),
        set([
            'schedule', 'power', 'base_lr',
            'end_factor', 'decay_steps_factor'
        ]))

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
        'Key a has dot children that would be overriden: b\nKey b has dot '
        'children that would be overriden: c\nKey g has dot children that '
        'would be overriden: h')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_msg):
      hyperparameters.expand_dot_keys(d)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
