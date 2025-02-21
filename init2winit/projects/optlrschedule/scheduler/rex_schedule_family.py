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

"""REX schedule family implementation.

W generalize REX schedule by introducing a beta parameter and remove coefficient
of 1/2 in the denominator.

arxiv: https://arxiv.org/pdf/2107.04197 (MLSys2022)
github: https://github.com/IvanVassi/REX_LR/blob/main/lr_scheduler.py

The difference between this implementation and the original REX schedule is
that the beta parameter is provided as a schedule parameter.

Original REX schedule:
  progress = t/T
  beta = 0.9 (in code) or 1 (in paper)
  return (1 - progress) / ((1 - progress * beta)/2 + 1/2)

Our Generalized REX schedule:
  progress = t/T
  beta = [0, inf]
  alpha = 1 - beta
  return (1 - progress) / (1 - progress * alpha)
"""

from typing import Any
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np


class RexScheduleFamily(base_schedule_family.WarmupScheduleFamily):
  """REX schedule implementation with configurable warmup using NumPy.

  The default values for max_val and min_val in schedule_config are 1 and 0,
  respectively,
  and the learning rate is scaled by base_lr. The beta parameter (default 0.9)
  is provided as a schedule parameter.
  """

  def validate_param(self, schedule_param: dict[str, Any]) -> bool:
    """Validate schedule parameters."""
    super().validate_param(schedule_param)
    required_params = {
        'p.warmup_steps',
        'p.beta',
    }  # p.max_val, p.min_val, and p.beta are optional with defaults.
    missing_params = required_params - set(schedule_param.keys())
    if missing_params:
      raise ValueError(
          f'Missing required schedule parameters: {missing_params}'
      )

    return True

  def list_schedule_parameter_keys(self) -> list[str]:
    return ['p.warmup_steps', 'p.beta']

  def rex_decay(self, progress: np.ndarray, beta: float = 0.9) -> np.ndarray:
    """Compute the REX decay multiplier.

    Args:
        progress: The progress in the decay phase (range from 0 to 1). Measured
          from the beginning of the decay phase.
        beta: The beta parameter controlling the shape of the decay.

    Returns:
        np.ndarray: The REX decay multiplier for each progress value.
    """
    alpha = 1 - beta
    return (1 - progress) / (1 - progress * alpha)

  def get_schedule(
      self, schedule_param: dict[str, Any], base_lr: float
  ) -> np.ndarray:
    """Generate the learning rate schedule for all training steps.

    Args:
        schedule_param: Dictionary of schedule parameters (e.g.,
          {'p.warmup_steps': 100, 'p.beta': 0.9}).
        base_lr: Base learning rate, which is used to scale the schedule.

    Returns:
        np.ndarray: An array of learning rates for each training step.
    """
    # Validate parameters (optional if called externally).
    self.validate_param(schedule_param)

    warmup_steps = int(schedule_param['p.warmup_steps'])
    beta = schedule_param.get('p.beta', 0.9)

    schedule = np.zeros(self.total_steps)
    warmup_fn = self.get_warmup_fn(self.warmup_type)
    warmup_config = self.schedule_family_config.get('warmup_config', {})

    # Warmup phase: compute learning rate values during warmup.
    # Use base_lr as the target learning rate during warmup.
    warmup_lr = []
    for step in range(warmup_steps):
      lr = warmup_fn(step, warmup_steps, base_lr, **warmup_config)
      warmup_lr.append(lr)
    warmup_lr = np.array(warmup_lr)
    schedule[:warmup_steps] = warmup_lr

    # Decay phase:
    decay_steps = self.total_steps - warmup_steps

    # Compute normalized progress (0 to 1) for the decay phase.
    steps = np.arange(decay_steps)
    progress = steps / decay_steps  # progress increases from 0 to 1
    # Apply the REX decay function with the specified beta.
    decay_multiplier = self.rex_decay(progress, beta=beta)
    # Compute learning rate for the decay phase:
    # scale by base_lr.
    decay_lr = base_lr * decay_multiplier

    schedule[warmup_steps:] = decay_lr
    return schedule
