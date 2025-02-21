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

"""Cosine schedule family implementation using optax.

Our code is based on the optax implementation of cosine decay schedule as listed
here:
https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.cosine_decay_schedule

schedule_family_config: Dictionary containing configuration such as:
  total_steps: Maximum number of training updates.

schedule_params:
  warmup_steps: Number of warmup steps.
  alpha: Decay factor.
"""

from typing import Any, Dict

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np


class CosineScheduleFamily(base_schedule_family.WarmupScheduleFamily):
  """Cosine schedule with configurable warmup methods."""

  def validate_config(self, config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""

    if 'alpha' not in config:
      raise ValueError('alpha must be specified in config')
    if config['alpha'] < 0.0:
      raise ValueError('alpha must be non-negative')

    if (
        'warmup_type' in config
        and config['warmup_type'] not in base_schedule_family.WARMUP_TYPES
    ):
      raise ValueError(
          'warmup_type must be one of linear, cosine, exponential, or'
          ' polynomial'
      )

  def validate_param(
      self, schedule_param: base_schedule_family.ScheduleParams
  ) -> bool:
    """Validate schedule parameters."""
    super().validate_param(schedule_param)

    required_params = {'p.exponent'}
    missing_params = required_params - set(schedule_param.keys())
    if missing_params:
      raise ValueError(f'Missing required parameters: {missing_params}')

    if not isinstance(schedule_param['p.exponent'], (int, float)):
      raise ValueError('exponent must be a number')
    if not (0.0 <= schedule_param['p.exponent']):
      raise ValueError('exponent must be larger than 0.0')

    return True

  def cosine_decay(
      self,
      step: int,
      decay_steps: int,
      base_lr: float,
      alpha: float,
      exponent: float,
  ) -> float:
    """Cosine decay from base_lr to alpha * base_lr.

    Args:
        step: Current training step
        decay_steps: Number of decay steps
        base_lr: Base learning rate
        alpha: Decay factor
        exponent: Power to raise the cosine decay to

    Returns:
        float: Learning rate at the current step
    """
    progress = step / decay_steps
    cosine = (1 + np.cos(np.pi * progress)) / 2
    decayed = (1 - alpha) * (cosine**exponent) + alpha
    return base_lr * decayed

  def list_schedule_parameter_keys(self) -> list[str]:
    return ['p.warmup_steps', 'p.exponent']

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
    # self.validate_param(schedule_param)

    alpha = self.schedule_family_config['alpha']
    warmup_steps = int(schedule_param.get('p.warmup_steps', 0))
    exponent = schedule_param.get('p.exponent', 1.0)

    schedule = np.zeros(self.total_steps)
    warmup_fn = self.get_warmup_fn(self.warmup_type)
    warmup_config = self.schedule_family_config.get('warmup_config', {})

    # Warmup phase
    for step in range(warmup_steps):
      schedule[step] = warmup_fn(step, warmup_steps, base_lr, **warmup_config)

    # Decay phase
    decay_steps = self.total_steps - warmup_steps
    for step in range(warmup_steps, self.total_steps):
      decay_step = step - warmup_steps
      schedule[step] = self.cosine_decay(
          decay_step, decay_steps, base_lr, alpha, exponent
      )

    return schedule
