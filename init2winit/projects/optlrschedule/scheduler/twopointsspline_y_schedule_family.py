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

r"""Jax implementation of TwoPointSplineDecayWithWarmup, with nonzero final y.

schedule_family_config: Dictionary containing configuration such as:
  y_min: Minimum learning rate factor (defalut is 0.0)
  total_steps: Total number of training steps
  warmup_type: Type of warmup schedule to use (linear or cosine)
  is_monotonic_decay: Whether the schedule should be monotonically decreasing

schedule_params:
  x0: Fraction of steps before peak is reached [0, 1]
  y0: Peak learning rate factor (should be 1.0)
  y1: First control point y value (0 < y1 <= 1)
  y_end: Final learning rate, as fraction of y2.
  delta_x1: Factor to determine x1 position
  delta_x2: Factor to determine x2 position
  delta_y2: Factor to determine y2 position
"""

from typing import Any, Dict

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np
from scipy import interpolate


class TwoPointSplineYScheduleFamily(base_schedule_family.WarmupScheduleFamily):
  """Learning rate scheduler with configurable warmup and spline decay using two control points."""

  def validate_config(self, config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    super().validate_config(config)  # Execute parent class validation

    # Validate spline-specific configurations
    if 'y_max' not in config:
      config['y_max'] = 1.0
    if 'is_monotonic_decay' not in config:
      config['is_monotonic_decay'] = True

  def validate_param(
      self, schedule_param: base_schedule_family.ScheduleParams
  ) -> bool:
    """Validate schedule parameters."""
    super().validate_param(schedule_param)  # Execute parent class validation

    required_params = {
        'p.x0',
        'p.y1',
        'p.delta_x1',
        'p.delta_x2',
        'p.delta_y2',
        'p.y_end',
    }
    missing_params = required_params - set(schedule_param.keys())
    if missing_params:
      raise ValueError(f'Missing required parameters: {missing_params}')

    # Validate parameter ranges
    for param in required_params:
      value = schedule_param[param]
      if not 0 <= value <= 1:
        raise ValueError(f'{param} must be in [0, 1], got {value}')

    return True

  def _compute_spline(
      self, schedule_param: Dict[str, float]
  ) -> interpolate.PchipInterpolator:
    """Compute the spline for the decay phase.

    Args:
        schedule_param: Dictionary containing parameters for the spline

    Returns:
        interpolate.PchipInterpolator: Cubic spline object for the decay phase
    """
    x0 = schedule_param['p.x0']
    y1 = schedule_param['p.y1']
    total_steps = self.total_steps

    if self.schedule_family_config['is_monotonic_decay']:
      y0 = self.schedule_family_config['y_max']

      # Calculate control points for the spline
      x1 = x0 + schedule_param['p.delta_x1'] * (1 - x0)
      x2 = x1 + schedule_param['p.delta_x2'] * (1 - x1)
      y2 = schedule_param['p.delta_y2'] * y1

      # Set end points of the decay
      x3 = 1
      y3 = schedule_param['p.y_end'] * y2
    else:
      raise ValueError(
          'Non monotonic decay is not supported for two point spline'
      )

    # Define control points for interpolation
    x_points = np.array([x0, x1, x2, x3])
    y_points = np.array([y0, y1, y2, y3])

    # Convert relative positions to actual step numbers
    x_steps = (x_points * total_steps).astype(int)

    # Handle potential duplicate x values
    unique_indices = np.unique(x_steps, return_index=True)[1]
    if len(unique_indices) < len(x_steps):
      # If duplicate x values exist, adjust them
      x_steps = np.array(sorted(list(set(x_steps.tolist()))))
      y_points = y_points[unique_indices]

      # Fallback to linear interpolation if insufficient points
      if len(x_steps) < 2:
        x_steps = np.array([0, total_steps - 1])
        y_points = np.array([y0, y2])

    return interpolate.PchipInterpolator(x_steps, y_points)

  def list_schedule_parameter_keys(self) -> list[str]:
    """List the keys of the schedule parameters."""
    return ['p.x0', 'p.y1', 'p.delta_x1', 'p.delta_x2', 'p.delta_y2', 'p.y_end']

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
        lr_array[step] = warmup_fn(
            step, warmup_steps, base_lr, **warmup_config
        )

    # Generate decay phase using pchip spline
    spline = self._compute_spline(schedule_param)
    decay_steps = np.arange(warmup_steps, total_steps)
    lr_array[warmup_steps:] = spline(decay_steps) * base_lr

    # Ensure learning rates stay within specified bounds
    return np.clip(
        lr_array,
        0,
        self.schedule_family_config['y_max'] * base_lr,
    )
