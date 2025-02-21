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

r"""Absolute learning rate schedule families based on a schedule ID.

A family of learning rate schedules is a function that maps
`schedule_id` to an array of learning rates.
The array itself is the factor of learning rate schedule.
As a default configuration (8-digits schedule ID),
`schedule_id` could be interpreted as follows.
- ID=22222222: constant learning rate schedule
- ID=12487531: schedule with a peak in the middle
- ID=98765432: schedule with monotonic decrease

get_schedule function requires the following hyperparameters:
  schedule_param: Dictionary containing configuration parameters.
  base_lr: Base learning rate.

As for the schedule_family_config, the following configurations are required:
schedule_family_config:
  x_num_bin: Number of chunks to divide the array into (default 8).
  y_num_bin: Number of possible values (default 10).
  y_min: Minimum value of the lr factor range (default 0).
  y_max: Maximum value of the lr factor range (default 1).
  y_distribution: Type of distribution ('uniform' or 'log_uniform').
  total_steps: Maximum number of training updates.
  scheduler_type: Type of scheduler ('spline', 'pchip', 'polynomial').

Learning rate factor arrays determine the shape of the learning rate schedule.
Each digit of `schedule_id` is mapped to a learning rate factor.
Then, the learning rate factor array is interpolated to generate a function that
computes the interpolated learning rate factor at time step t.
Finally, the interpolated learning rate factor is multiplied by the base
learning rate togenerate the final learning rate schedule.
"""

from typing import Any, Dict

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np
from scipy import interpolate


class PiecewiseScheduleFamily(base_schedule_family.BaseScheduleFamily):
  """PiecewiseScheduleFamily based on schedule ID and interpolation type."""

  def validate_config(self, config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    if 'total_steps' not in config:
      raise ValueError('total_steps must be specified in config')
    if config['total_steps'] <= 0:
      raise ValueError('total_steps must be positive')

    required_params = {
        'x_num_bin',  # number of chunks to divide the array into
        'y_num_bin',  # number of possible values (bins)
        'y_min',  # minimum value of the lr factor range
        'y_max',  # maximum value of the lr factor range
        'y_distribution',  # distribution type（'uniform' or 'log_uniform'）
        'scheduler_type',  # scheduler type（'pchip' or 'polynomial'）
    }

    missing_params = required_params - set(config.keys())
    if missing_params:
      raise ValueError(f'Missing required parameters: {missing_params}')

    if config['x_num_bin'] <= 0:
      raise ValueError('x_num_bin must be positive')
    if config['y_num_bin'] <= 0:
      raise ValueError('y_num_bin must be positive')
    if config['y_min'] > config['y_max']:
      raise ValueError('y_min must be less than or equal to y_max')
    if config['y_distribution'] not in ['uniform', 'log_uniform']:
      raise ValueError('y_distribution must be either uniform or log_uniform')
    if config['scheduler_type'] not in ['pchip', 'polynomial']:
      raise ValueError('scheduler_type must be either pchip or polynomial')

  def validate_param(
      self, schedule_param: base_schedule_family.ScheduleParams
  ) -> bool:
    """Validate schedule parameters."""

    if 'p.schedule_id' not in schedule_param:
      raise ValueError('schedule_id must be specified')

    schedule_id = int(schedule_param['p.schedule_id'])
    max_id = (
        self.schedule_family_config['y_num_bin']
        ** self.schedule_family_config['x_num_bin']
    )

    if not isinstance(schedule_id, (int, float)):
      raise ValueError('schedule_id must be a number')
    if schedule_id < 0:
      raise ValueError('schedule_id must be non-negative')
    if schedule_id >= max_id:
      raise ValueError(f'schedule_id must be less than {max_id}')

    return True

  def _generate_y_map(
      self, y_num_bin: int, min_val: float, max_val: float, distribution: str
  ) -> np.ndarray:
    """Generate y_num_bin values (log)uniformly spaced on [min_val, max_val].

    Args:
        y_num_bin (int): Number of possible values (bins).
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.
        distribution (str): Type of distribution ('uniform' or 'log_uniform').

    Returns:
        np.ndarray: Array of possible values based on the distribution.
    """
    if distribution == 'uniform':
      return np.linspace(min_val, max_val, y_num_bin)
    elif distribution == 'log_uniform':
      if min_val <= 0 or max_val <= 0:
        raise ValueError(
            'min_val and max_val must be greater than 0 for log_uniform'
            ' distribution'
        )
      log_min = np.log(min_val)
      log_max = np.log(max_val)
      return np.exp(np.linspace(log_min, log_max, y_num_bin))
    else:
      raise ValueError(f'Distribution {distribution} is not supported')

  def _id_to_lr_factor_array(self, schedule_id: int) -> list[float]:
    """Convert a schedule_id to an array of x_num_bin values.

    Args:
        schedule_id (int): The schedule identifier [0, y_num_bin^x_num_bin).

    Returns:
        list (float): List of x_num_bin values generated from the id.
    """
    config = self.schedule_family_config
    if schedule_id >= config['y_num_bin'] ** config['x_num_bin']:
      raise ValueError(
          f'Set id less than {config["y_num_bin"]**config["x_num_bin"] - 1}'
      )
    if schedule_id < 0:
      raise ValueError('Set id greater than or equal to 0')

    y_mapped_values = self._generate_y_map(
        config['y_num_bin'],
        config['y_min'],
        config['y_max'],
        config['y_distribution'],
    )
    reversed_lr_factor_array = []

    # Convert id to base-y_num_bin and map to possible values
    for _ in range(config['x_num_bin']):
      reversed_lr_factor_array.append(
          y_mapped_values[schedule_id % config['y_num_bin']]
      )
      schedule_id //= config['y_num_bin']

    return reversed_lr_factor_array[::-1]

  def list_schedule_parameter_keys(self) -> list[str]:
    return ['p.schedule_id']

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

    schedule_id = int(schedule_param['p.schedule_id'])
    lr_factor_array = self._id_to_lr_factor_array(schedule_id)
    chunk_size = (
        self.schedule_family_config['total_steps']
        // self.schedule_family_config['x_num_bin']
    )

    # Define base points for interpolation
    interpolation_base_points = np.linspace(
        chunk_size // 2,
        self.schedule_family_config['total_steps'] - chunk_size // 2 - 1,
        self.schedule_family_config['x_num_bin'],
        dtype=int,
    )

    # Select interpolation method
    if self.schedule_family_config['scheduler_type'] == 'pchip':
      f = interpolate.PchipInterpolator(
          interpolation_base_points, lr_factor_array
      )
    elif self.schedule_family_config['scheduler_type'] == 'polynomial':
      coefficients = np.polyfit(interpolation_base_points, lr_factor_array, 2)
      f = np.poly1d(coefficients)
    else:
      raise ValueError(
          f"Scheduler type {self.schedule_family_config['scheduler_type']} is"
          ' not supported'
      )

    # Generate an array of learning rates for all time steps
    lr_array = np.array([
        np.clip(
            f(t) * base_lr,
            self.schedule_family_config['y_min'] * base_lr,
            self.schedule_family_config['y_max'] * base_lr,
        )
        for t in range(self.schedule_family_config['total_steps'])
    ])

    return lr_array
