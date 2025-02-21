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

r"""Unit tests for the scheduler family class.

These tests verify the generation of learning rate schedules.
For PiecewiseScheduleFamily, schedule ID and different interpolation methods
(spline, pchip, polynomial) identity lr schedule.
For TwoPointSplineSchedyleFamily, lr schedule is identified by two control
points and warmup point.
For CosineScheduleFamily, lr schedule is identified by warmup steps, alpha, and
initial value.
"""

from absl.testing import absltest
from init2winit.projects.optlrschedule.scheduler import cosine_schedule_family
from init2winit.projects.optlrschedule.scheduler import piecewise_schedule_family
from init2winit.projects.optlrschedule.scheduler import rex_schedule_family
from init2winit.projects.optlrschedule.scheduler import smooth_nonmonotonic_schedule_family
from init2winit.projects.optlrschedule.scheduler import sqrt_schedule_family
from init2winit.projects.optlrschedule.scheduler import twopointslinear_schedule_family
from init2winit.projects.optlrschedule.scheduler import twopointsspline_schedule_family
import numpy as np


class TestPiecewiseScheduleFamily(absltest.TestCase):
  """Setup common test parameters."""

  schedule_config = {
      'x_num_bin': 4,
      'y_num_bin': 5,
      'y_min': 0.01,
      'y_max': 1.0,
      'y_distribution': 'uniform',
      'total_steps': 100,
      'scheduler_type': 'pchip',
  }

  schedule_param = {
      'p.schedule_id': 2,
  }
  base_lr = 0.1

  def test_get_schedule_pchip(self):
    """Test PCHIP interpolation of learning rate."""
    schedule_family = piecewise_schedule_family.PiecewiseScheduleFamily(
        self.schedule_config
    )
    self.schedule_config['scheduler_type'] = 'pchip'
    lr_fn = schedule_family.get_schedule(
        self.schedule_param,
        self.base_lr,
    )
    t = 50  # Midpoint of training updates
    interpolated_lr = lr_fn[t]
    self.assertBetween(
        interpolated_lr,
        self.schedule_config['y_min'] * self.base_lr,
        self.schedule_config['y_max'] * self.base_lr,
    )

  def test_get_schedule_polynomial(self):
    """Test polynomial interpolation of learning rate."""
    schedule_family = piecewise_schedule_family.PiecewiseScheduleFamily(
        self.schedule_config
    )
    self.schedule_config['scheduler_type'] = 'polynomial'
    lr_fn = schedule_family.get_schedule(
        self.schedule_param,
        self.base_lr,
    )
    t = 50  # Midpoint of training updates
    interpolated_lr = lr_fn[t]
    self.assertBetween(
        interpolated_lr,
        self.schedule_config['y_min'] * self.base_lr,
        self.schedule_config['y_max'] * self.base_lr,
    )

  def test_invalid_schedule_id(self):
    """Test that invalid schedule_id raises an error."""
    schedule_family = piecewise_schedule_family.PiecewiseScheduleFamily(
        self.schedule_config
    )
    with self.assertRaises(ValueError):

      schedule_param = {
          'p.schedule_id': -1,
      }
      base_lr = 0.1

      schedule_family.get_schedule(
          schedule_param,
          base_lr,
      )

    with self.assertRaises(ValueError):
      schedule_param = {
          'p.schedule_id': 123456789,
      }
      base_lr = 0.1

      schedule_family.get_schedule(
          schedule_param,
          base_lr,
      )


class TestTwoPointSplineDecayWithWarmup(absltest.TestCase):
  """Test cases for TwoPointSplineDecayWithWarmup scheduler."""

  # Setup common test parameters.
  scheduler_family_config = {
      'y_min': 0.0,
      'y_max': 1.0,
      'total_steps': 100,
      'is_monotonic_decay': True,
      'warmup_type': 'exponential',
      'warmup_config': {
          'exp_growth_rate': 5.0,
      },
  }

  schedule_param = {
      'p.x0': 0.1,  # Initial warmup phase (10% of total steps)
      'p.y1': 0.7,  # First control point multiplier
      'p.delta_x1': 0.3,  # Offset factor for x1 position
      'p.delta_x2': 1.0,  # Offset factor for x2 position
      'p.delta_y2': 0.1,  # Coefficient for y2 position
  }

  base_lr = 0.1
  eps = 1e-4  # Global numerical tolerance

  def test_decay_phase(self):
    """Test the decay phase of the schedule."""
    scheduler = twopointsspline_schedule_family.TwoPointSplineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    warmup_steps = int(
        self.schedule_param['p.x0']
        * self.scheduler_family_config['total_steps']
    )
    decay_schedule = schedule[warmup_steps:]

    # Basic range validation
    self.assertTrue(
        np.all(decay_schedule >= 0), 'Found negative values in decay schedule'
    )

    self.assertTrue(
        np.all(decay_schedule <= self.base_lr + self.eps),
        'Values exceed base learning rate',
    )

    # Check overall decreasing trend
    # Allow small local increases
    if len(decay_schedule) > 1:
      # Verify start and end points
      self.assertGreater(
          decay_schedule[0],
          decay_schedule[-1],
          'Decay schedule does not show overall decrease',
      )

      # Check for absence of sharp increases
      diffs = np.diff(decay_schedule)
      max_increase = np.max(diffs)
      self.assertLessEqual(
          max_increase,
          self.eps,
          f'Large increase detected in decay schedule: {max_increase}',
      )

  def test_full_schedule(self):
    """Test the complete schedule including warmup and decay."""
    scheduler = twopointsspline_schedule_family.TwoPointSplineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    # Basic validations
    self.assertLen(schedule, self.scheduler_family_config['total_steps'])

    self.assertTrue(np.all(schedule >= 0), 'Found negative values in schedule')

    self.assertTrue(
        np.all(schedule <= self.base_lr + self.eps),
        'Values exceed base learning rate',
    )

  def test_extreme_values(self):
    """Test with extreme but valid configurations."""
    test_config = {
        **self.schedule_param,
        'p.x0': 0.02,  # Short warmup period
        'p.y1': 0.9,  # Gradual decay rate
        'p.delta_x1': 0.4,
        'p.delta_x2': 0.9,
        'p.delta_y2': 0.2,
    }

    scheduler = twopointsspline_schedule_family.TwoPointSplineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(test_config, self.base_lr)

    self.assertLen(schedule, self.scheduler_family_config['total_steps'])
    self.assertTrue(np.all(schedule >= 0))
    self.assertTrue(np.all(schedule <= self.base_lr + self.eps))


class TestTwoPointLinearDecayWithWarmup(absltest.TestCase):
  """Test cases for TwoPointLinearDecayWithWarmup scheduler."""

  # Setup common test parameters.
  scheduler_family_config = {
      'y_min': 0.0,
      'y_max': 1.0,
      'total_steps': 100,
      'is_monotonic_decay': True,
      'warmup_type': 'exponential',
      'warmup_config': {
          'exp_growth_rate': 5.0,
      },
  }

  schedule_param = {
      'p.x0': 0.1,  # Initial warmup phase (10% of total steps)
      'p.y1': 0.7,  # First control point multiplier
      'p.delta_x1': 0.3,  # Offset factor for x1 position
      'p.delta_x2': 1.0,  # Offset factor for x2 position
      'p.delta_y2': 0.1,  # Coefficient for y2 position
  }

  base_lr = 0.1
  eps = 1e-4  # Global numerical tolerance

  def test_decay_phase(self):
    """Test the decay phase of the schedule."""
    scheduler = twopointslinear_schedule_family.TwoPointLinearScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    warmup_steps = int(
        self.schedule_param['p.x0']
        * self.scheduler_family_config['total_steps']
    )
    decay_schedule = schedule[warmup_steps:]

    # Basic range validation
    self.assertTrue(
        np.all(decay_schedule >= 0), 'Found negative values in decay schedule'
    )

    self.assertTrue(
        np.all(decay_schedule <= self.base_lr + self.eps),
        'Values exceed base learning rate',
    )

    # Check overall decreasing trend
    # Allow small local increases
    if len(decay_schedule) > 1:
      # Verify start and end points
      self.assertGreater(
          decay_schedule[0],
          decay_schedule[-1],
          'Decay schedule does not show overall decrease',
      )

      # Check for absence of sharp increases
      diffs = np.diff(decay_schedule)
      max_increase = np.max(diffs)
      self.assertLessEqual(
          max_increase,
          self.eps,
          f'Large increase detected in decay schedule: {max_increase}',
      )

  def test_full_schedule(self):
    """Test the complete schedule including warmup and decay."""
    scheduler = twopointslinear_schedule_family.TwoPointLinearScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    # Basic validations
    self.assertLen(schedule, self.scheduler_family_config['total_steps'])

    self.assertTrue(np.all(schedule >= 0), 'Found negative values in schedule')

    self.assertTrue(
        np.all(schedule <= self.base_lr + self.eps),
        'Values exceed base learning rate',
    )

  def test_extreme_values(self):
    """Test with extreme but valid configurations."""
    test_config = {
        **self.schedule_param,
        'p.x0': 0.02,  # Short warmup period
        'p.y1': 0.9,  # Gradual decay rate
        'p.delta_x1': 0.4,
        'p.delta_x2': 0.9,
        'p.delta_y2': 0.2,
    }

    scheduler = twopointslinear_schedule_family.TwoPointLinearScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(test_config, self.base_lr)

    self.assertLen(schedule, self.scheduler_family_config['total_steps'])
    self.assertTrue(np.all(schedule >= 0))
    self.assertTrue(np.all(schedule <= self.base_lr + self.eps))


class TestCosineScheduleFamily(absltest.TestCase):
  """Test cases for CosineScheduleFamily scheduler."""

  # Setup common test parameters
  scheduler_family_config = {
      'total_steps': 100,
      'alpha': 0,
      'warmup_type': 'linear',
  }

  schedule_param = {
      'p.warmup_steps': 10,
      'p.exponent': 1.0,
  }

  base_lr = 0.1
  eps = 1e-3  # Global numerical tolerance

  def test_decay_phase(self):
    """Test the decay phase of the schedule."""
    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    warmup_steps = self.schedule_param['p.warmup_steps']
    decay_schedule = schedule[warmup_steps:]

    # Basic range validation
    self.assertTrue(
        np.all(decay_schedule >= 0), 'Found negative values in decay schedule'
    )

    max_lr = self.base_lr
    self.assertTrue(
        np.all(decay_schedule <= max_lr + self.eps),
        'Values exceed maximum learning rate',
    )

    # Check overall decreasing trend
    if len(decay_schedule) > 1:
      # Verify start and end points
      self.assertGreater(
          decay_schedule[0],
          decay_schedule[-1],
          'Decay schedule does not show overall decrease',
      )

      # Check for absence of sharp increases
      diffs = np.diff(decay_schedule)
      max_increase = np.max(diffs)
      self.assertLessEqual(
          max_increase,
          self.eps,
          f'Large increase detected in decay schedule: {max_increase}',
      )

    # Verify final value matches alpha
    expected_final_value = self.base_lr * self.scheduler_family_config['alpha']
    self.assertAlmostEqual(
        decay_schedule[-1],
        expected_final_value,
        delta=self.eps,
        msg='Final value does not match alpha scaling',
    )

  def test_warmup_phase(self):
    """Test the warmup phase of the schedule."""
    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)

    warmup_steps = self.schedule_param['p.warmup_steps']
    warmup_schedule = schedule[: warmup_steps + 1]

    if warmup_steps > 0:
      # Check if warmup starts from near zero
      self.assertAlmostEqual(
          warmup_schedule[0],
          0.0,
          delta=self.eps,
          msg='Warmup does not start from zero',
      )

      # Check if warmup reaches the initial value
      max_lr = self.base_lr
      self.assertAlmostEqual(
          warmup_schedule[-1],
          max_lr,
          delta=self.eps,
          msg='Warmup does not reach target value',
      )

      # Check monotonic increase during warmup
      self.assertTrue(
          np.all(np.diff(warmup_schedule) >= -self.eps),
          'Warmup phase is not monotonically increasing',
      )

  def test_full_schedule(self):
    """Test the complete schedule including warmup and decay."""
    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    total_steps = self.scheduler_family_config['total_steps']

    # Basic validations
    self.assertLen(schedule, total_steps)

    self.assertTrue(np.all(schedule >= 0), 'Found negative values in schedule')

    max_lr = self.base_lr
    self.assertTrue(
        np.all(schedule <= max_lr + self.eps),
        'Values exceed maximum learning rate',
    )

  def test_no_warmup(self):
    """Test schedule without warmup."""
    schedule_param = {
        'p.exponent': 1,
        'p.warmup_steps': 0,
    }

    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(schedule_param, self.base_lr)

    # Check if schedule starts from initial value
    self.assertAlmostEqual(
        schedule[0],
        self.base_lr,
        delta=self.eps,
        msg='Schedule does not start from initial value',
    )

  def test_extreme_values(self):
    """Test with extreme but valid configurations."""
    schedule_param = {
        'p.warmup_steps': 2,
        'p.exponent': 1,
    }
    total_steps = self.scheduler_family_config['total_steps']

    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(schedule_param, self.base_lr)

    self.assertLen(schedule, total_steps)
    self.assertTrue(np.all(schedule >= 0))
    max_lr = self.base_lr
    self.assertTrue(np.all(schedule <= max_lr + self.eps))

  def test_validation(self):
    """Test parameter validation."""
    scheduler = cosine_schedule_family.CosineScheduleFamily(
        self.scheduler_family_config
    )

    # Test invalid warmup steps
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.warmup_steps': -1}
      scheduler.validate_param(param)

    # Test invalid exponent
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.exponent': -1}
      scheduler.validate_param(param)


class TestNonmonotonicSmoothScheduleFamily(absltest.TestCase):
  """Test cases for NonmonotonicSmoothScheduleFamily scheduler."""

  # Setup common test parameters.
  scheduler_family_config = {
      'total_steps': 100,
  }

  schedule_param = {
      'p.x_peak': 0.5,
      'p.y_start': 0.0,
      'p.y_end': 1.0,
      'p.y1': 0.2,
      'p.delta_x1': 0.1,
      'p.y2': 0.7,
      'p.delta_x2': 0.8,
  }

  base_lr = 0.1
  eps = 1e-4  # Global numerical tolerance

  def test_invalid_params(self):
    """Test invalid parameters."""
    scheduler = smooth_nonmonotonic_schedule_family.TwoPointSplineSmoothNonmonoticScheduleFamily(
        self.scheduler_family_config
    )
    # Test invalid x_peak
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.x_peak': 1.1}
      scheduler.validate_param(param)

    # Test invalid y_start
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.y_start': -1}
      scheduler.validate_param(param)

    # Test invalid y_end
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.y_end': -0.1}
      scheduler.validate_param(param)

    # Test invalid y1
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.y1': -0.1}
      scheduler.validate_param(param)

    # Test invalid delta_x1
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.delta_x1': 1.1}
      scheduler.validate_param(param)

    # Test invalid y2
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.y2': -0.1}
      scheduler.validate_param(param)

    # Test invalid delta_x2
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.delta_x2': 10}
      scheduler.validate_param(param)

  def test_full_schedule(self):
    """Test the complete schedule."""
    scheduler = smooth_nonmonotonic_schedule_family.TwoPointSplineSmoothNonmonoticScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    total_steps = self.scheduler_family_config['total_steps']

    # Basic validations
    self.assertLen(schedule, total_steps)

    self.assertTrue(np.all(schedule >= 0), 'Found negative values in schedule')

    max_lr = self.base_lr
    self.assertTrue(
        np.all(schedule <= max_lr + self.eps),
        'Values exceed maximum learning rate',
    )

  def test_extreme_values(self):
    """Test with extreme but valid configurations."""
    schedule_param = {
        'p.x_peak': 0.01,
        'p.y_start': 0.0,
        'p.y_end': 0.0,
        'p.y1': 0.0,
        'p.delta_x1': 0.01,
        'p.y2': 0.0,
        'p.delta_x2': 0.01,
    }
    total_steps = self.scheduler_family_config['total_steps']

    scheduler = smooth_nonmonotonic_schedule_family.TwoPointSplineSmoothNonmonoticScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(schedule_param, self.base_lr)

    self.assertLen(schedule, total_steps)
    self.assertTrue(np.all(schedule >= 0))
    max_lr = self.base_lr
    self.assertTrue(np.all(schedule <= max_lr + self.eps))


class TestSqrtScheduleFamily(absltest.TestCase):
  """Test cases for SqrtScheduleFamily scheduler."""

  scheduler_family_config = {
      'total_steps': 100,
      'warmup_type': 'linear',
  }

  schedule_param = {
      'p.warmup_steps': 10,
      'p.alpha': 0.5,
  }

  base_lr = 0.1
  eps = 1e-2  # Global numerical tolerance

  def test_invalid_params(self):
    """Test invalid parameters."""
    scheduler = sqrt_schedule_family.SqrtScheduleFamily(
        self.scheduler_family_config
    )
    # Test invalid warmup steps
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.warmup_steps': -1}
      scheduler.validate_param(param)

  def test_full_schedule(self):
    """Test the complete schedule including warmup and decay."""
    scheduler = sqrt_schedule_family.SqrtScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    total_steps = self.scheduler_family_config['total_steps']

    # Basic validations
    self.assertLen(schedule, total_steps)

  def test_warmup_phase(self):
    """Test the warmup phase of the schedule."""
    scheduler = sqrt_schedule_family.SqrtScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    warmup_steps = self.schedule_param['p.warmup_steps']

    # Check the warmup phase
    self.assertEqual(
        schedule[warmup_steps],
        max(schedule),
        msg='Warmup phase does not reach target value',
    )


class TestRexScheduleFamily(absltest.TestCase):
  """Test cases for RexScheduleFamily scheduler."""

  scheduler_family_config = {
      'total_steps': 100,
      'warmup_type': 'linear',
  }

  schedule_param = {
      'p.warmup_steps': 10,
      'p.beta': 0.9,
  }

  base_lr = 0.1
  eps = 1e-2  # Global numerical tolerance

  def test_invalid_params(self):
    """Test invalid parameters."""
    scheduler = rex_schedule_family.RexScheduleFamily(
        self.scheduler_family_config
    )
    # Test invalid warmup steps
    with self.assertRaises(ValueError):
      param = {**self.schedule_param, 'p.warmup_steps': -1}
      scheduler.validate_param(param)

  def test_full_schedule(self):
    """Test the complete schedule including warmup and decay."""
    scheduler = rex_schedule_family.RexScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    total_steps = self.scheduler_family_config['total_steps']

    self.assertLen(schedule, total_steps)

  def test_warmup_phase(self):
    """Test the warmup phase of the schedule."""
    scheduler = rex_schedule_family.RexScheduleFamily(
        self.scheduler_family_config
    )
    schedule = scheduler.get_schedule(self.schedule_param, self.base_lr)
    warmup_steps = self.schedule_param['p.warmup_steps']

    # Check the warmup phase
    self.assertEqual(
        schedule[warmup_steps],
        max(schedule),
        msg='Warmup phase does not reach target value',
    )


if __name__ == '__main__':
  absltest.main()
