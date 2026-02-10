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

"""Constant schedule family implementation using optax."""

from typing import Dict
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np


class ConstantScheduleFamily(base_schedule_family.WarmupScheduleFamily):
  """Constant learning rate schedule with configurable warmup."""

  def list_schedule_parameter_keys(self) -> list[str]:
    return ['p.warmup_steps']

  def get_schedule(
      self,
      schedule_param: Dict[str, float],
      base_lr: float,
  ) -> np.ndarray:
    """Generate constant learning rate schedule with warmup.

    Args:
        schedule_param: Dictionary containing schedule parameters.
        base_lr: Base learning rate.

    Returns:
        np.ndarray: Array of learning rates for each training step.
    """
    self.validate_param(schedule_param)
    schedule = np.zeros(self.total_steps)
    warmup_config = self.schedule_family_config.get('warmup_config', {})

    warmup_steps = int(schedule_param.get('p.warmup_steps', 0))
    if warmup_steps > 0:
      # Warmup phase
      warmup_fn = self.get_warmup_fn(self.warmup_type)
      for step in range(warmup_steps):
        schedule[step] = warmup_fn(
            step, warmup_steps, base_lr, **warmup_config
        )

      # Constant phase
      schedule[warmup_steps:] = base_lr
    else:
      # No warmup, constant throughout
      schedule[:] = base_lr

    return schedule
