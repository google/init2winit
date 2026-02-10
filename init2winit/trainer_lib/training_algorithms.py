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

"""Module for registering and retrieving training algorithms."""

from init2winit.trainer_lib import training_algorithm
# pylint: disable=g-bad-import-order


_ALGORITHMS = {
    'optax_training_algorithm': training_algorithm.OptaxTrainingAlgorithm,
}


def get_training_algorithm(name):
  if name not in _ALGORITHMS:
    raise ValueError(f'Training algorithm {name} not found.')
  return _ALGORITHMS[name]


def register_training_algorithm(name):
  def decorator(cls):
    _ALGORITHMS[name] = cls
    return cls
  return decorator
