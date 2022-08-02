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

"""Trainers for init2winit."""

from init2winit.trainer_lib import quantization_trainer
from init2winit.trainer_lib import self_tuning_trainer
from init2winit.trainer_lib import trainer


_ALL_TRAINERS = {
    'standard': trainer,
    'self_tuning': self_tuning_trainer,
    'quantization': quantization_trainer,
}


def get_train_fn(trainer_name):
  """Maps trainer name to a train function."""
  try:
    return _ALL_TRAINERS[trainer_name].train
  except KeyError:
    raise ValueError('Unrecognized trainer: {}'.format(trainer_name)) from None

