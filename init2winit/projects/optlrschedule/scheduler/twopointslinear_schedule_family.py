# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

r"""Implementation of TwoPointLinearInterpolationWithWarmup.

schedule_family_config: Dictionary containing configuration such as:
  y_min: Minimum learning rate factor (defalut is 0.0)
  y_max: Maximum learning rate factor (default is 1.0)
  total_steps: Total number of training steps
  warmup_type: Type of warmup schedule to use (linear or cosine)
  is_monotonic_decay: Whether the schedule should be monotonically decreasing

schedule_params:
  x0: Fraction of steps before peak is reached [0, 1]
  y0: Peak learning rate factor (should be 1.0)
  y1: First control point y value (0 < y1 <= 1)
  delta_x1: Factor to determine x1 position
  delta_x2: Factor to determine x2 position
  delta_y2: Factor to determine y2 position
"""

from typing import Any, Dict

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np


class TwoPointLinearScheduleFamily(base_schedule_family.WarmupScheduleFamily):
  """Learning rate scheduler with configurable warmup and linear decay using two control points."""

  def validate_config(self, config: Dict[str, Any]) -> None:
    super().validate_config(config)  # Execute parent class validation

    # Validate linear-specific configurations
    if 'y_min' not in config:
      config['y_min'] = 0.0
    if 'y_max' not in config:
      config['y_max'] = 1.0
    if 'is_monotonic_decay' not in config:
      config['is_monotonic_decay'] = True

    if config['y_min'] > config['y_max']:
      raise ValueError('y_min must be less than or equal to y_max')

  def validate_param(
      self, schedule_param: base_schedule_family.ScheduleParams
  ) -> bool:
    super().validate_param(schedule_param)  # Execute parent class validation

    required_params = {'p.x0', 'p.y1', 'p.delta_x1', 'p.delta_x2', 'p.delta_y2'}
    missing_params = required_params - set(schedule_param.keys())
    if missing_params:
      raise ValueError(f'Missing required parameters: {missing_params}')

    # Validate parameter ranges
    if not 0 <= schedule_param['p.x0'] <= 1:
      raise ValueError(f'x0 must be in [0, 1], got {schedule_param["p.x0"]}')

    if not 0 <= schedule_param['p.delta_x1'] <= 1:
      raise ValueError(
          f'delta_x1 must be in [0, 1], got {schedule_param["p.delta_x1"]}'
      )

    if not 0 <= schedule_param['p.delta_x2'] <= 1:
      raise ValueError(
          f'delta_x2 must be in [0, 1], got {schedule_param["p.delta_x2"]}'
      )

    if not 0 <= schedule_param['p.y1'] <= 1:
      raise ValueError(f'y1 must be in [0, 1], got {schedule_param["p.y1"]}')

    if not 0 <= schedule_param['p.delta_y2'] <= 1:
      raise ValueError(
          f'delta_y2 must be in [0, 1], got {schedule_param["p.delta_y2"]}'
      )

    return True

  def _compute_linear_interpolation(
      self, schedule_param: Dict[str, float]
  ) -> np.ndarray:
    """Compute linear interpolation for the decay phase.

    Args:
        schedule_param: Dictionary containing parameters for the linear
          interpolation

    Returns:
        np.ndarray: Array of learning rates for the decay phase
    """
    x0 = schedule_param['p.x0']
    y1 = schedule_param['p.y1']
    total_steps = self.total_steps

    if self.schedule_family_config['is_monotonic_decay']:
      y0 = self.schedule_family_config['y_max']

      # Calculate control points for the linear interpolation
      x1 = x0 + schedule_param['p.delta_x1'] * (1 - x0)
      x2 = x1 + schedule_param['p.delta_x2'] * (1 - x1)
      y2 = schedule_param['p.delta_y2'] * y1

      # Set end points of the decay
      x3 = 1
      y3 = self.schedule_family_config['y_min']
    else:
      raise ValueError(
          'Non monotonic decay is not supported for two point linear'
      )

    # Define control points for interpolation
    x_points = np.array([x0, x1, x2, x3])
    y_points = np.array([y0, y1, y2, y3])

    # Convert relative positions to actual step numbers
    x_steps = (x_points * total_steps).astype(int)

    # Create a piecewise linear interpolation
    lr_values = np.interp(np.arange(total_steps), x_steps, y_points)

    return lr_values

  def list_schedule_parameter_keys(self) -> list[str]:
    """List the keys of the schedule parameters."""
    return ['p.x0', 'p.y1', 'p.delta_x1', 'p.delta_x2', 'p.delta_y2']

  def get_schedule(
      self,
      schedule_param: Dict[str, float],
      base_lr: float,
  ) -> np.ndarray:
    """Generate learning rate schedule based on parameters.

    Args:
        schedule_param: Dictionary containing schedule parameters
        base_lr: Base learning rate

    Returns:
        np.ndarray: Array of learning rates for each training step
    """
    self.validate_param(schedule_param)

    total_steps = self.total_steps
    warmup_steps = int(schedule_param['p.x0'] * total_steps)
    warmup_config = self.schedule_family_config.get('warmup_config', {})
    lr_array = np.zeros(total_steps)

    # Apply warmup phase if specified
    if warmup_steps > 0:
      warmup_fn = self.get_warmup_fn(self.warmup_type)
      for step in range(warmup_steps):
        lr_array[step] = warmup_fn(step, warmup_steps, base_lr, **warmup_config)

    # Generate decay phase using linear interpolation
    decay_lr_values = self._compute_linear_interpolation(schedule_param)
    lr_array[warmup_steps:] = decay_lr_values[warmup_steps:] * base_lr

    # Ensure learning rates stay within specified bounds
    return np.clip(
        lr_array,
        self.schedule_family_config['y_min'] * base_lr,
        self.schedule_family_config['y_max'] * base_lr,
    )
