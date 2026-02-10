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

"""Cosine schedule family implementation with fixed exponent of 1.

The only difference with cosine_schedule_family.py is that the exponent is
fixed to 1.0.

schedule_family_config: Dictionary containing configuration such as:
  total_steps: Maximum number of training updates.

schedule_params:
  warmup_steps: Number of warmup steps.
  alpha: Decay factor.
"""

from typing import Dict

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
from init2winit.projects.optlrschedule.scheduler import cosine_schedule_family
import numpy as np


class CosineStandardScheduleFamily(cosine_schedule_family.CosineScheduleFamily):
  """Cosine schedule with configurable warmup methods and exponent of 1.0."""

  def validate_param(
      self, schedule_param: base_schedule_family.ScheduleParams
  ) -> bool:
    """Validate schedule parameters."""
    return base_schedule_family.WarmupScheduleFamily.validate_param(
        self, schedule_param
    )

  def list_schedule_parameter_keys(self) -> list[str]:
    return ['p.warmup_steps']

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

    alpha = self.schedule_family_config['alpha']
    warmup_steps = int(schedule_param.get('p.warmup_steps', 0))

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
          decay_step, decay_steps, base_lr, alpha, 1.0
      )

    return schedule
