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

"""Unit tests for schedules.py."""

from absl.testing import absltest
from init2winit import schedules
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v1 as tf


class LearningRateTest(absltest.TestCase):
  """Tests learning rate decay schedules."""

  def test_polynomial_decay(self):
    """Test polynomial schedule works correctly with decay_steps_factor."""
    hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'polynomial',
            'power': 2.0,
            'base_lr': .1,
            'end_factor': .01,
            'decay_steps_factor': 0.5,
        }
    ))
    max_training_steps = 400
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    hps = hps.lr_hparams
    decay_steps = max_training_steps * hps['decay_steps_factor']
    for step in range(max_training_steps):
      expected_learning_rate = tf.train.polynomial_decay(
          hps['base_lr'],
          step,
          decay_steps,
          hps['end_factor'] * hps['base_lr'],
          power=hps['power'])().numpy()
      self.assertAlmostEqual(lr_fn(step), expected_learning_rate)

  def test_polynomial_decay_decay_steps(self):
    """Test polynomial schedule works correctly with decay_steps."""
    hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'polynomial',
            'power': 2.0,
            'base_lr': .1,
            'end_factor': .01,
            'decay_steps': 200,
        }
    ))
    max_training_steps = 400
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    hps = hps.lr_hparams
    decay_steps = hps['decay_steps']
    for step in range(max_training_steps):
      expected_learning_rate = tf.train.polynomial_decay(
          hps['base_lr'],
          step,
          decay_steps,
          hps['end_factor'] * hps['base_lr'],
          power=hps['power'])().numpy()
      self.assertAlmostEqual(lr_fn(step), expected_learning_rate)

  def test_compound_schedule_cosine(self):
    """Test that compound schedule correctly handles the cosine schedule."""
    expected_lrs = [
        2.0,
        1 + np.cos(0.1 * np.pi),
        1 + np.cos(0.2 * np.pi),
        1 + np.cos(0.3 * np.pi),
        1 + np.cos(0.4 * np.pi),
        1 + np.cos(0.5 * np.pi),
        1 + np.cos(0.6 * np.pi),
        1 + np.cos(0.7 * np.pi),
        1 + np.cos(0.8 * np.pi),
        1 + np.cos(0.9 * np.pi),
        0.0,
    ]
    hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'compound',
            'factors': 'constant * cosine',
            'base_lr': 2,
        }
    ))
    max_training_steps = 11
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    for step in range(max_training_steps):
      self.assertAlmostEqual(
          lr_fn(step), expected_lrs[step], places=6, msg=f'{step=}'
      )

  def test_warmup_steps_fraction(self):
    """Test that the _warmup suffix works with warmup_steps_fraction."""
    lr_hparams_a = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 10.0,
        'warmup_steps': 2,
    })
    lr_hparams_b = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 10.0,
        'warmup_steps_fraction': 0.2,
    })
    lf_fn_a = schedules.get_schedule_fn(lr_hparams_a, 10)
    lf_fn_b = schedules.get_schedule_fn(lr_hparams_b, 10)
    lrs_a = [lf_fn_a(t) for t in range(10)]
    lrs_b = [lf_fn_b(t) for t in range(10)]
    self.assertEqual(lrs_a, lrs_b)

    lr_hparams_malformed = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 10.0,
        'warmup_steps_fraction': 0.2,
        'warmup_steps': 2,
    })
    with self.assertRaises(ValueError):
      schedules.get_schedule_fn(lr_hparams_malformed, 10)

  def test_compound_schedule_cosine_with_warmup(self):
    """Test that compound schedule correctly handles the cosine schedule."""
    expected_lrs = [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        4.0 * (1 + np.cos(0.1 * np.pi)),
        4.0 * (1 + np.cos(0.2 * np.pi)),
        4.0 * (1 + np.cos(0.3 * np.pi)),
        4.0 * (1 + np.cos(0.4 * np.pi)),
        4.0 * (1 + np.cos(0.5 * np.pi)),
        4.0 * (1 + np.cos(0.6 * np.pi)),
        4.0 * (1 + np.cos(0.7 * np.pi)),
        4.0 * (1 + np.cos(0.8 * np.pi)),
        4.0 * (1 + np.cos(0.9 * np.pi)),
        0.0,
    ]
    hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'compound',
            'factors': 'constant * cosine * linear_warmup',
            'base_lr': 8,
            'warmup_steps': 4,
        }
    ))
    max_training_steps = 4 + 11
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    for step in range(max_training_steps):
      self.assertAlmostEqual(
          lr_fn(step), expected_lrs[step], places=6, msg=f'{step=}'
      )

  def test_concatenate_out_of_bounds(self):
    """Test concatenating multiple schedules and going beyond the end."""
    lengths = [3, 5]
    lr_hparams_a = config_dict.ConfigDict({
        'schedule': 'constant',
        'base_lr': 10.0,
    })
    lr_hparams_b = config_dict.ConfigDict({
        'schedule': 'constant',
        'base_lr': 7.0,
    })
    lr_fn = schedules.concatenate(lengths, lr_hparams_a, lr_hparams_b)
    # Although the schedule is only defined for sum(lengths), out of bounds
    # steps should be blithely passed into the final sub-schedule piece so we
    # test an extra 3 steps.
    lrs = [lr_fn(t) for t in range(sum(lengths)+3)]
    expected_lrs = [10.0] * 3 + [7.0] * (5+3)
    self.assertEqual(lrs, expected_lrs)

  def test_concatenate(self):
    """Test concatenating multiple schedules."""
    lengths = [3, 5, 7]
    lr_hparams_a = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 10.0,
        'warmup_steps': 2,
    })
    lr_hparams_b = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 7.0,
        'warmup_steps': 2,
    })
    lr_hparams_c = config_dict.ConfigDict({
        'schedule': 'constant_warmup',
        'base_lr': 7.0,
        'warmup_steps': 5,
    })
    lr_fn = schedules.concatenate(
        lengths, lr_hparams_a, lr_hparams_b, lr_hparams_c)
    lrs = [lr_fn(t) for t in range(sum(lengths))]
    lf_fn_a = schedules.get_schedule_fn(lr_hparams_a, lengths[0])
    lf_fn_b = schedules.get_schedule_fn(lr_hparams_b, lengths[1])
    lf_fn_c = schedules.get_schedule_fn(lr_hparams_c, lengths[2])
    expected_lrs_a = [lf_fn_a(t) for t in range(lengths[0])]
    expected_lrs_b = [lf_fn_b(t) for t in range(lengths[1])]
    expected_lrs_c = [lf_fn_c(t) for t in range(lengths[2])]
    expected_lrs = expected_lrs_a + expected_lrs_b + expected_lrs_c
    self.assertEqual(lrs, expected_lrs)

  def test_schedule_stretching(self):
    """Test that schedules can be properly stretched."""
    max_training_steps = 100
    lr_hparams = config_dict.ConfigDict({
        'schedule': 'mlperf_polynomial',
        'base_lr': 10.0,
        'warmup_steps': 10,
        'decay_end': -1,
        'end_lr': 1e-4,
        'power': 2.0,
        'start_lr': 0.0,
        'warmup_power': 1.0,
    })
    lr_fn = schedules.get_schedule_fn(lr_hparams, max_training_steps)
    stretch_factor = 3
    stretched_lr_fn = schedules.get_schedule_fn(
        lr_hparams, max_training_steps, stretch_factor=stretch_factor)
    lrs = [lr_fn(t) for t in range(max_training_steps)]
    stretched_lrs = [
        stretched_lr_fn(t) for t in range(stretch_factor * max_training_steps)
    ]
    self.assertEqual(lrs, stretched_lrs[::stretch_factor])
    self.assertEqual(lrs, stretched_lrs[1::stretch_factor])
    self.assertEqual(lrs, stretched_lrs[2::stretch_factor])
    # Assert that the stretched schedule has proper staircase behavior.
    for update_step in range(max_training_steps):
      start = update_step * stretch_factor
      end = (update_step + 1) * stretch_factor
      expected = [lrs[update_step]] * stretch_factor
      self.assertEqual(stretched_lrs[start:end], expected)

  def test_mlperf_schedule(self):
    """Test there are no changes to the MLPerf polynomial decay schedule."""
    expected_lrs = [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,
        2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4,
        5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2,
        8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 9.802962, 9.607885,
        9.414769, 9.223614, 9.034419, 8.847184, 8.661909, 8.478596, 8.297242,
        8.117851, 7.940418, 7.764947, 7.5914364, 7.419886, 7.2502966, 7.082668,
        6.917, 6.7532916, 6.591545, 6.4317584, 6.273932, 6.1180663, 5.964162,
        5.812217, 5.662234, 5.5142093, 5.368148, 5.2240453, 5.0819044, 4.941723,
        4.803503, 4.6672425, 4.532944, 4.4006047, 4.2702274, 4.1418095,
        4.0153522, 3.8908558, 3.7683203, 3.647745, 3.5291305, 3.4124763,
        3.297783, 3.18505, 3.0742776, 2.965466, 2.858614, 2.753724, 2.6507936,
        2.5498245, 2.4508152, 2.353767, 2.2586792, 2.1655521, 2.0743854,
        1.9851794, 1.8979341, 1.8126491, 1.7293249, 1.6479613, 1.568558,
        1.4911155, 1.4156334, 1.342112, 1.2705511, 1.2009507, 1.133311,
        1.0676318, 1.0039133, 0.94215524, 0.8823574, 0.8245205, 0.7686443,
        0.71472853, 0.66277343, 0.61277884, 0.56474483, 0.5186714, 0.4745585,
        0.43240622, 0.39221448, 0.35398334, 0.31771275, 0.28340274, 0.2510533,
        0.22066444, 0.19223614, 0.16576843, 0.14126128, 0.11871469, 0.098128565,
        0.079503134, 0.06283828, 0.048134, 0.035390284, 0.02460714, 0.01578457,
        0.00892257, 0.004021142,
    ]
    hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'mlperf_polynomial',
            'base_lr': 10.0,
            'warmup_steps': 50,
            'decay_end': -1,
            'end_lr': 1e-4,
            'power': 2.0,
            'start_lr': 0.0,
            'warmup_power': 1.0,
        }
    ))
    max_training_steps = 50
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    for step in range(max_training_steps):
      self.assertAlmostEqual(lr_fn(step), expected_lrs[step])

  def test_t2t_rsqrt_normalized_decay(self):
    """Test t2t_rsqrt_normalized_decay schedule works correctly."""
    hps = config_dict.ConfigDict(
        dict(
            lr_hparams={
                'schedule': 't2t_rsqrt_normalized_decay',
                'base_lr': 0.01,
                'defer_steps': 10,
            }))
    expected_lrs = [
        0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
        0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
        0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
        0.009999999776482582, 0.009999999776482582, 0.009534625336527824,
        0.009128709323704243, 0.008770580403506756, 0.008451541885733604,
        0.00816496554762125, 0.007905693724751472, 0.0076696500182151794,
        0.007453559897840023, 0.007254762575030327
    ]

    max_training_steps = 20
    lr_fn = schedules.get_schedule_fn(hps.lr_hparams, max_training_steps)
    for step, expected_lr in zip(range(max_training_steps), expected_lrs):
      self.assertAlmostEqual(lr_fn(step), expected_lr)

  def test_raises(self):
    """Test that an exception is raised with extra hparams."""
    good_hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'mlperf_polynomial',
            'base_lr': .1,
            'warmup_steps': 200,
            'decay_end': -1,
            'end_lr': 1e-4,
            'power': 2.0,
            'start_lr': 0.0,
            'warmup_power': 1.0,
        }
    ))
    bad_hps = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'mlperf_polynomial',
            'warmup_steps': 200,
            'base_lr': .1,
            'decay_end': -1,
            'end_lr': 1e-4,
            'power': 2.0,
            'start_lr': 0.0,
            'initial_value': .1,
        }
    ))
    bad_hps2 = config_dict.ConfigDict(dict(
        lr_hparams={
            'schedule': 'polynomial',
            'power': 2.0,
            'base_lr': .1,
            'end_factor': .01,
            'decay_steps': 200,
            'decay_steps_factor': 0.5
        }
    ))
    # This should pass.
    schedules.get_schedule_fn(good_hps.lr_hparams, 1)

    # This should raise an exception due to the extra hparam.
    with self.assertRaises(ValueError):
      schedules.get_schedule_fn(bad_hps.lr_hparams, 1)

    # This should raise an exception due to the mutually exclusive hparams.
    with self.assertRaises(ValueError):
      schedules.get_schedule_fn(bad_hps2.lr_hparams, 1)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
