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

"""init2winit."""

import importlib
from absl import logging

# By lazily importing, we do not need to import the entire library even if we
# are only using a few models/datasets/optimizers, which substantially cuts down
# on import times.
_IMPORTS = [
    # Directories.
    'colab',
    'dataset_lib',
    'hessian',
    'init_lib',
    'model_lib',
    'mt_eval',
    'optimizer_lib',
    'tests',
    # Files.
    'base_callback',
    'callbacks',
    'checkpoint',
    'hyperparameters',
    'import_utils',
    'main',
    'schedules',
    'setup',
    'test_checkpoint',
    'test_hyperparameters',
    'test_schedules',
    'test_trainer',
    'test_training_metrics_grabber',
    'test_utils',
    'trainer',
    'utils',
]


def _lazy_import(name):
  """Load a submodule named `name`."""
  if name not in _IMPORTS:
    raise AttributeError(
        'module init2winit has no attribute {}'.format(name))
  module = importlib.import_module(__name__)
  try:
    imported = importlib.import_module('.' + name, 'init2winit')
  except AttributeError:
    logging.warning(
        'Submodule %s was found, but will not be loaded due to AttributeError '
        'within it while importing.', name)
    return
  setattr(module, name, imported)
  return imported


# Lazily load any top level modules when accessed. Requires Python 3.7.
__getattr__ = _lazy_import
