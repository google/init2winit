# coding=utf-8
# Copyright 2021 The init2winit Authors.
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


  callback_builder = callbacks.get_callback(config['name'])
  callback = callback_builder(optimizer, dataset, hps, config, train_dir)

  callback_metrics = callback.run_eval(optimizer, batch_stats, global_step).

We require that the callback config has a field 'name', which the trainer
uses to determine which callbacks to run. The dictionary, callback_metrics
should be scalar valued, and will be automatically added to the existing trainer
scalar metrics.
"""

# TODO(gilmer) Add serialization so that we can checkpoint callback state.


class TestCallBack:
  """Example callback to specify the required API."""

  def __init__(self, optimizer, dataset, hps, config, train_dir):
    """Define the API for callback construction."""
    del optimizer
    del dataset
    del hps
    del config
    del train_dir

  def run_eval(self, optimizer, batch_stats, global_step):
    """Define the API for running the callback during eval.

    Args:
      optimizer: Replicated optimizer the trainer has (this also has the
        model parameters).
      batch_stats: Replicated batch_stats from the trainer.
      global_step: Current training step.

    Returns:
      A dictionary of scalar metrics. Note, any existing metric returned by
      trainer.evaluate are forbidden, e.g. including 'train/ce_loss' will
      resort in trainer throwing an exception.
    """
    del optimizer
    del batch_stats
    del global_step

    return {'train/fake_metric': 1.0}

_ALL_CALLBACKS = {
    'test': TestCallBack
}


def get_callback(callback_name):
  return _ALL_CALLBACKS[callback_name]
