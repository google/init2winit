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

"""Init2winit workload."""

from init2winit.experiments.mlcommons.workloads import mlcommons_targets
from init2winit.experiments.mlcommons.workloads import mlcommons_workload_info
from init2winit.trainer_lib import spec


class Init2winitWorkload(spec.Workload):
  """Init2winit workload."""

  def initialize(self, model, hps):
    self._model = model
    self._hps = hps

  @property
  def workload_name(self):
    if not self._hps.workload_name:
      self._workload_name = self._hps.dataset + '_' + self._hps.model
    else:
      self._workload_name = self._hps.workload_name

    return self._workload_name

  @property
  def param_shapes(self):
    return self._model.param_shapes

  @property
  def model_params_types(self):
    return self._model.param_types

  @property
  def step_hint(self):
    if self.workload_name not in mlcommons_workload_info.num_train_steps:
      raise ValueError(
          f'Workload {self.workload_name} not found in num_train_steps.')
    return mlcommons_workload_info.num_train_steps[self.workload_name]

  @property
  def target_metric_name(self):
    if self.workload_name not in mlcommons_targets.validation_targets:
      raise ValueError(
          f'Workload {self.workload_name} not found in validation targets.')

    return mlcommons_targets.validation_targets[self.workload_name]['metric']
