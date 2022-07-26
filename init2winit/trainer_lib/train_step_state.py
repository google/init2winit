# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""TODO(gdahl): DO NOT SUBMIT without one-line documentation for train_state.

TODO(gdahl): DO NOT SUBMIT without a detailed description of train_state.
"""
import dataclasses
from typing import Any


@dataclasses.dataclass()
class TrainStepState:
  """Dataclass to group together *replicated* model/optimizer/metrics state."""
  # PLEASE do NOT use this for unreplicated state, it will confuse us.
  optimizer_state: Any
  params: Any
  batch_stats: Any
  metrics_state: Any

  def astuple(self):
    # dataclasses.astuple will perform an undesirable deepcopy.
    return (self.optimizer_state, self.params, self.batch_stats,
            self.metrics_state)
