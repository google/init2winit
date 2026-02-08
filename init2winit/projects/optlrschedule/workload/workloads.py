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

r"""Workload classes for different tasks."""

from typing import Any

from init2winit.projects.optlrschedule.workload import cifar10_cnn
from init2winit.projects.optlrschedule.workload import linear_regression
from init2winit.projects.optlrschedule.workload import wikitext103_transformer


WORKLOADS = {
    'cifar10_cnn': cifar10_cnn.Cifar10Training,
    'wikitext103': wikitext103_transformer.Wikitext103Transformer,
    'linear_regression': linear_regression.LinearRegression,
}


def get_workload_class(workload_name: str) -> type[Any]:
  """Get workload class for a given workload name.

  Args:
    workload_name: The name of the workload to get.

  Returns:
    The class of the workload.

  Raises:
    ValueError: If the workload name is not found.
  """
  if workload_name not in WORKLOADS:
    raise ValueError(f'Unsupported workload: {workload_name}')
  return WORKLOADS[workload_name]
