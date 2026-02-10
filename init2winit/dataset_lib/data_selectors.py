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

"""Definitions for supported data selection functions."""


def noop(
    dataset_iterator,
    optimizer_state,
    params,
    batch_stats,
    hps,
    global_step,
    constant_base_rng):
  """An example no-op data selector that just yields the next batch.

  Args:
    dataset_iterator: the (preprocessed, batched, prefetched) dataset iterator.
    optimizer_state: the current optimizer state.
    params: the model parameters.
    batch_stats: the model batch statistics.
    hps: the experiment hyperparameters.
    global_step: the current global step.
    constant_base_rng: the RNG used for the experiment. IMPORTANT NOTE: this
      will be constant for all calls to this function, in order to get a unique
      RNG each time we need to do
      `rng = jax.random.fold_in(constant_base_rng, global_step)`.

  Yields:
    A batch of data.
  """
  del optimizer_state
  del params
  del batch_stats
  del hps
  del global_step
  del constant_base_rng
  yield from dataset_iterator


def data_echoing(
    dataset_iterator,
    optimizer_state,
    params,
    batch_stats,
    hps,
    global_step,
    constant_base_rng):
  """An example data echoing selector.

  Args:
    dataset_iterator: the (preprocessed, batched, prefetched) dataset iterator.
    optimizer_state: the current optimizer state.
    params: the model parameters.
    batch_stats: the model batch statistics.
    hps: the experiment hyperparameters.
    global_step: the current global step.
    constant_base_rng: the RNG used for the experiment. IMPORTANT NOTE: this
      will be constant for all calls to this function, in order to get a unique
      RNG each time we need to do
      `rng = jax.random.fold_in(constant_base_rng, global_step)`.

  Yields:
    A batch of data.
  """
  del optimizer_state
  del params
  del batch_stats
  del global_step
  del constant_base_rng
  for x in dataset_iterator:
    for _ in range(hps.num_data_echoes):
      yield x


ALL_SELECTORS = {
    'noop': noop,
    'data_echoing': data_echoing,
}
