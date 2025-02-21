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

r"""Base schedule families for learning rate schedules."""

import abc
from typing import Any, Callable, Dict, List, TypeAlias

import numpy as np


ScheduleParams: TypeAlias = Dict[str, float]
WARMUP_TYPES = ['linear', 'cosine', 'exponential', 'polynomial']
_IGNORE_PREFIX_KEYS = ['score', 'group']
_SCHEDULE_PARAM_KEY_PREFIX = 'p.'


def is_schedule_param(key_name: str) -> bool:
  """Check if key_name is one of the schedule parameters."""
  return key_name.startswith(_SCHEDULE_PARAM_KEY_PREFIX)


def add_prefix_to_schedule_param_dict(
    schedule_param_config: Dict[str, Any],
) -> Dict[str, Any]:
  """Add prefix to schedule parameter config.

  Args:
    schedule_param_config: Dictionary containing schedule parameter config.

  Returns:
    Dictionary containing schedule parameter config with prefix added.
  """
  d = {}
  for key, value in schedule_param_config.items():
    if any(sub in key for sub in _IGNORE_PREFIX_KEYS):
      d[key] = value
    elif is_schedule_param(key):
      d[key] = value
    else:
      d[_SCHEDULE_PARAM_KEY_PREFIX + key] = value
  return d


def add_prefix_to_schedule_param_key_list(
    schedule_param_key_list: List[str],
) -> List[str]:
  """Add prefix to schedule parameter config list.

  Args:
    schedule_param_key_list: List of schedule parameter keys.

  Returns:
    List of schedule parameter keys with prefix added.
  """
  return [
      key if is_schedule_param(key) else _SCHEDULE_PARAM_KEY_PREFIX + key
      for key in schedule_param_key_list
  ]


def linear_warmup(
    step: int,
    warmup_steps: int,
    base_lr: float,
) -> float:
  """Linear warmup from 0 to base_lr.

  Args:
      step: Current training step (default: = warmup_steps).
      warmup_steps: Number of warmup steps.
      base_lr: Base learning rate.

  Returns:
      float: Learning rate at the current step.
  """
  return base_lr * (step / warmup_steps)


def cosine_warmup(
    step: int,
    warmup_steps: int,
    base_lr: float,
) -> float:
  """Cosine warmup from 0 to base_lr.

  Args:
      step: Current training step (default: = warmup_steps).
      warmup_steps: Number of warmup steps.
      base_lr: Base learning rate.

  Returns:
      float: Learning rate at the current step.
  """
  progress = step / warmup_steps
  return base_lr * (1 - np.cos(progress * np.pi)) / 2


def exponential_warmup(
    step: int,
    warmup_steps: int,
    base_lr: float,
    exp_growth_rate: float,
) -> float:
  """Exponential warmup from 0 to base_lr.

  Args:
      step: Current training step (default: = warmup_steps).
      warmup_steps: Number of warmup steps.
      base_lr: Base learning rate.
      exp_growth_rate: Growth rate of the exponential warmup.

  Returns:
      float: Learning rate at the current step.
  """
  progress = step / warmup_steps
  return base_lr * (1 - np.exp(-exp_growth_rate * progress))


def polynomial_warmup(
    step: int, warmup_steps: int, base_lr: float, poly_power: float
) -> float:
  """Polynomial (quadratic) warmup from 0 to base_lr.

  Args:
      step: Current training step (default: = warmup_steps).
      warmup_steps: Number of warmup steps.
      base_lr: Base learning rate.
      poly_power: Power to raise the polynomial warmup to.

  Returns:
      float: Learning rate at the current step.
  """
  progress = step / warmup_steps
  return base_lr * (progress**poly_power)


class BaseScheduleFamily(abc.ABC):
  """Base class for learning rate schedules."""

  def __init__(self, schedule_family_config: Dict[str, Any]):
    self.schedule_family_config = schedule_family_config
    self.total_steps = schedule_family_config['total_steps']

  @abc.abstractmethod
  def list_schedule_parameter_keys(self) -> List[str]:
    """List all valid schedule parameter keys."""
    pass

  def validate_config(self, config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    if 'total_steps' not in config:
      raise ValueError('total_steps must be specified in config')
    if config['total_steps'] <= 0:
      raise ValueError('total_steps must be positive')

  @abc.abstractmethod
  def get_schedule(
      self,
      schedule_param: Dict[str, float],
      base_lr: float,
  ) -> np.ndarray:
    """Generate learning rate schedule based on parameters.

    Args:
        schedule_param: Dictionary containing schedule parameters.
        base_lr: Base learning rate.

    Returns:
        np.ndarray: Array of learning rates for each training step.
    """
    pass

  def get_schedules(
      self,
      schedule_params: List[Dict[str, float]],
      base_lr_list: List[float],
  ) -> np.ndarray:
    """Generate learning rate schedules for all training steps.

    Args:
        schedule_params: List of dictionary containing schedule parameters.
        base_lr_list: List of base learning rates.

    Returns:
        np.ndarray: Array of learning rates for each training step.
    """
    if len(schedule_params) != len(base_lr_list):
      raise ValueError(
          'Length of schedule_params and base_lr_list must be the same'
      )

    num_schedules = len(schedule_params)
    schedules = np.zeros((num_schedules, self.total_steps))
    for idx, (schedule_param, base_lr) in enumerate(
        zip(schedule_params, base_lr_list)
    ):
      schedules[idx] = self.get_schedule(schedule_param, base_lr)
    return schedules


class WarmupScheduleFamily(BaseScheduleFamily):
  """Base class for learning rate schedules with warmup."""

  def __init__(self, schedule_family_config: Dict[str, Any]):
    super().__init__(schedule_family_config)
    self.validate_config(schedule_family_config)
    self.warmup_type = schedule_family_config.get('warmup_type', 'linear')

  def validate_config(self, config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    super().validate_config(config)  # Execute parent class validation

    warmup_config = config.get('warmup_config', {})
    # Validate warmup configuration
    if config['warmup_type'] == 'exponential':
      if not isinstance(warmup_config['exp_growth_rate'], float):
        raise ValueError('exp_growth_rate must be a float')
      if warmup_config['exp_growth_rate'] <= 0.0:
        raise ValueError('exp_growth_rate must be positive')
    elif config['warmup_type'] == 'polynomial':
      if not isinstance(warmup_config['poly_power'], float):
        raise ValueError('poly_power must be a float')
      if warmup_config['poly_power'] <= 0.0:
        raise ValueError('poly_power must be positive')
    elif config['warmup_type'] == 'linear':
      pass
    elif config['warmup_type'] == 'cosine':
      pass
    else:
      raise ValueError(
          'warmup_type must be one of linear, cosine, exponential, or'
          ' polynomial'
      )

  def validate_param(self, schedule_param: ScheduleParams) -> bool:
    """Validate schedule parameters."""
    if 'p.warmup_steps' in schedule_param:
      warmup_steps = schedule_param['p.warmup_steps']
      if not isinstance(warmup_steps, (int, float)):
        raise ValueError('warmup_steps must be a number')
      if not (0 <= warmup_steps <= self.total_steps):
        raise ValueError(
            f'warmup_steps must be between 0 and {self.total_steps}'
        )
    return True

  def get_warmup_fn(self, warmup_type: str) -> Callable[..., float]:
    """Get warmup function based on type.

    Args:
        warmup_type: Type of warmup function.

    Returns:
        Callable: Function for warmup.

    Usage of callable:
        warmup_fn = get_warmup_fn(warmup_type)
        warmup_schedule = [warmup_fn(step, warmup_steps, base_lr, **config) for
        step in
        range(warmup_steps)]
    """
    warmup_fns = {
        'linear': linear_warmup,
        'cosine': cosine_warmup,
        'exponential': exponential_warmup,
        'polynomial': polynomial_warmup,
    }
    if warmup_type not in warmup_fns:
      raise ValueError(f'Unsupported warmup type: {warmup_type}')
    return warmup_fns[warmup_type]
