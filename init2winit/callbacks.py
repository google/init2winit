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

"""Registry for the available callbacks."""

from init2winit import gradient_statistics_callback
from init2winit.hessian import hessian_callback
from init2winit.hessian import model_debugger_callback
from init2winit.mt_eval import mt_callback


_ALL_CALLBACKS = {
    'hessian': hessian_callback.HessianCallback,
    'mt': mt_callback.MTEvaluationCallback,
    'model_debugger': model_debugger_callback.ModelDebugCallback,
    'gradient_statistics': (
        gradient_statistics_callback.GradientStatisticsCallback
    ),
}


def get_callback(callback_name):
  """Get the corresponding callback builder based on the callback_name.

  Args:
    callback_name: (str) e.g. mt.

  Returns:
    Callback builder class.
  Raises:
    ValueError if callback_name is unrecognized.
  """
  try:
    return _ALL_CALLBACKS[callback_name]
  except KeyError:
    raise ValueError('Unrecognized callback name: {}'.format(callback_name))
