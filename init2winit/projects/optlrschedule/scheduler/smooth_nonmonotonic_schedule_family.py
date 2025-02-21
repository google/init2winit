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

r"""Jax implementation of Smooth Non-monotonic Schedule family.

schedule_family_config = {
    total_steps: Total number of training steps
}

schedule_params = {
    'y_start': Boundary value of initial learning rate
    'y_end': Boundary value of final learning rate
    'x_peak': Peak position of learning rate in horizontal axis
    'y1': Learning rate at first control point
    'delta_x1': Related distance of first control point from start
    'y2': Learning rate at second control point
    'delta_x2': Related distance of second control point from first control
    point
}
"""

from typing import Dict, Tuple
from absl import logging
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np
from scipy import interpolate


class TwoPointSplineSmoothNonmonoticScheduleFamily(
    base_schedule_family.BaseScheduleFamily
):
  """Non-monotonic learning rate scheduler with arbitrary peak placement."""

  def validate_param(self, params: Dict[str, float]) -> None:
    """Validate schedule parameters."""
    required_params = {
        'p.y_start',
        'p.y_end',  # Boundary values
        'p.x_peak',  # Peak position
        'p.y1',
        'p.delta_x1',  # First normal point
        'p.y2',
        'p.delta_x2',  # Second normal point
    }
    if not all(k in params for k in required_params):
      raise ValueError(f'Missing parameters. Required: {required_params}')

    # Validate ranges
    for param, value in params.items():
      if param.startswith('p.delta_x') and not (0 < value < 1):
        raise ValueError(f'{param} must be in (0, 1), got {value}')
      if param.startswith('p.y') and not (0 <= value <= 1):
        raise ValueError(f'{param} must be in [0, 1], got {value}')
      if param == 'p.x_peak' and not (0 < value < 1):
        raise ValueError(f'x_peak must be in (0, 1), got {value}')

  def _compute_control_points(
      self, params: Dict[str, float]
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute control points using stick-breaking procedure.

    Points can be placed in any order, with peak at any position.

    Args:
      params: Dictionary of schedule parameters.

    Returns:
      Tuple of control points (x, y).
    """
    # First point using delta_x1
    x1 = params['p.delta_x1']

    # Second point using delta_x2 of remaining space
    remaining_space = 1.0 - x1
    x2 = x1 + remaining_space * params['p.delta_x2']

    x_points = np.array([0.0, x1, x2, params['p.x_peak'], 1.0])
    y_points = np.array(
        [params['p.y_start'],
         params['p.y1'],
         params['p.y2'],
         1.0,
         params['p.y_end']]
    )
    order = np.argsort(x_points)
    x_points = x_points[order]
    y_points = y_points[order]

    # Ensure uniqueness of x coordinates
    unique_indices = np.unique(x_points, return_index=True)[1]
    if len(unique_indices) < len(x_points):
      logging.warning(
          'Found duplicates in x_points. Reducing from %d to %d unique points.',
          len(x_points),
          len(unique_indices),
      )
    x_points = x_points[unique_indices]
    y_points = y_points[unique_indices]

    return x_points, y_points

  def list_schedule_parameter_keys(self) -> list[str]:
    return [
        'p.y_start',
        'p.y_end',
        'p.x_peak',
        'p.y1',
        'p.delta_x1',
        'p.y2',
        'p.delta_x2',
    ]

  def get_schedule(
      self, params: Dict[str, float], base_lr: float
  ) -> np.ndarray:
    """Generate learning rate schedule."""
    self.validate_param(params)

    # Compute control points
    x_points, y_points = self._compute_control_points(params)
    x_steps = x_points * self.total_steps
    spline = interpolate.PchipInterpolator(x_steps, y_points)
    steps = np.arange(self.total_steps)
    lr_array = spline(steps) * base_lr

    return lr_array
