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

r"""Schedule families for learning rate schedules."""

from typing import Any

from init2winit.projects.optlrschedule.scheduler import constant_schedule_family
from init2winit.projects.optlrschedule.scheduler import cosine_schedule_family
from init2winit.projects.optlrschedule.scheduler import rex_schedule_family
from init2winit.projects.optlrschedule.scheduler import smooth_nonmonotonic_schedule_family
from init2winit.projects.optlrschedule.scheduler import sqrt_schedule_family
from init2winit.projects.optlrschedule.scheduler import twopointslinear_schedule_family
from init2winit.projects.optlrschedule.scheduler import twopointsspline_schedule_family

SCHEDULE_FAMILIES = {
    'cosine': cosine_schedule_family.CosineScheduleFamily,
    'constant': constant_schedule_family.ConstantScheduleFamily,
    'twopointsspline': (
        twopointsspline_schedule_family.TwoPointSplineScheduleFamily
    ),
    'twopointslinear': (
        twopointslinear_schedule_family.TwoPointLinearScheduleFamily
    ),
    'sqrt': sqrt_schedule_family.SqrtScheduleFamily,
    'smoothnonmonotonic': (
        smooth_nonmonotonic_schedule_family.TwoPointSplineSmoothNonmonoticScheduleFamily
    ),
    'rex': rex_schedule_family.RexScheduleFamily,
}


def get_schedule_family_class(
    schedule_type: str,
) -> type[Any]:
  """Get schedule family class for a given schedule type."""
  try:
    return SCHEDULE_FAMILIES[schedule_type]
  except KeyError as e:
    raise ValueError(f'Unsupported schedule type: {schedule_type}') from e
