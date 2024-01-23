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

"""Infrastructure for arbitrary code to be run during training.

Callbacks can be stateful, and in trainer are meant to be called as follows:


  callback_builder = callbacks.get_callback(config['callback_name'])
  callback = callback_builder(model, params, batch_stats, optimizer_state,
                              dataset, hps, config, train_dir, rng)

  callback_metrics = callback.run_eval(params, batch_stats,
                                       optimizer_state, global_step).

We require that the config has a field 'callback_name', which the trainer
uses to determine which callbacks to run. The dictionary, callback_metrics
should be scalar valued, and will be automatically added to the existing trainer
scalar metrics.
"""

# TODO(gilmer) Add serialization so that we can checkpoint callback state.


class BaseCallBack:
  """Base callback to specify the required API."""

  def __init__(self, model, params, batch_stats, optimizer_state,
               optimizer_update_fn, dataset, hps, callback_config, train_dir,
               rng):
    """Defines the API for callback construction."""
    pass

  def run_eval(self, params, batch_stats, optimizer_state, global_step):
    """Define the API for running the callback during eval.

    Args:
      params: Replicated params from the trainer.
      batch_stats: Replicated batch_stats from the trainer.
      optimizer_state: Replicated optimizer state from the trainer.
      global_step: Current training step.

    Returns:
      A dictionary of scalar metrics. Note, any existing metric returned by
      trainer.evaluate are forbidden, e.g. including 'train/ce_loss' will
      resort in trainer throwing an exception.
    """
    raise NotImplementedError('Subclasses must implement run_eval().')
