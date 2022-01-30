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

"""Import utilities."""

import importlib
from absl import logging


def lazy_import_fn(module_subname, valid_imports):
  """Create a function that can be used to lazily import modules.

  This function returned from calling this is intended to be used as the
  __getattr__ method on a module. For example, for the optimizer_lib submodule,
  this would be used like the following in optimizer_lib/__init__.py:

  __getattr__ = lazy_import_fn(
      'optimizer_lib', ['hessian_free', 'optimizers', ...])

  Args:
    module_subname: the name of the submodule under init2winit whose modules
      will be lazily loaded.
    valid_imports: a list of strings of valid import names, typically just the
      list of .py files in the submodule.

  Returns:
    A function that takes a module name and returns the imported module using
    importlib.import_module.
  """
  if module_subname:
    module_subname = '.' + module_subname

  def _lazy_import(name):
    """Load a submodule named `name`."""
    if name not in valid_imports:
      raise AttributeError(
          'module init2winit{} has no attribute {}'.format(
              module_subname, name))
    module = importlib.import_module(__name__)
    try:
      imported = importlib.import_module(
          '.' + name, 'init2winit' + module_subname)
    except AttributeError:
      logging.warning(
          'Submodule %s was found, but will not be loaded due to AttributeError '
          'within it while importing.', name)
      return
    setattr(module, name, imported)
    return imported

  return _lazy_import
