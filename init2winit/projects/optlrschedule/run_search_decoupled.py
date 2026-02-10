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

r"""execute random search for optlrschedule."""

import collections
from collections.abc import Sequence
import datetime
import logging
import math
import os

from absl import app
from absl import flags
from init2winit import utils
from init2winit.projects.optlrschedule import log_utils
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
from init2winit.projects.optlrschedule.scheduler import schedule_families
from init2winit.projects.optlrschedule.search_algorithm import search_algorithms
from init2winit.projects.optlrschedule.workload import workloads
import jax
from jax.experimental import multihost_utils
from ml_collections import config_flags
import numpy as np



FLAGS = flags.FLAGS

os.environ['FLAX_PROFILE'] = 'true'

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

# By making the workd ids use 1-based indexing, they will match XM WorkUnits.
flags.DEFINE_integer(
    'worker_id', 1,
    'Worker id to fold into top level rng key. Note that to exactly reproduce a'
    ' previous experiment, you must use the same worker_id.')


def get_schedules_and_augmented_params(
    schedule_family: base_schedule_family.BaseScheduleFamily,
    schedule_params: list[dict[str, float]],
    base_lr_list: list[float],
    num_training_steps: int,
    num_param_rngs: int,
) -> tuple[np.ndarray, list[tuple[dict[str, float], float]], np.ndarray]:
  """Get schedules and augmented schedule parameters.

  Args:
    schedule_family: configuration for schedule family.
    schedule_params: list of schedule parameters.
    base_lr_list: list of base learning rates.
    num_training_steps: number of training steps.
    num_param_rngs: number of parameter random keys per pair of base_lr and
      schedule shape.
  Returns:
  The total number schedules returned is num_param_rngs * len(schedule_params) *
  len(base_lr_list), and is the length of all of the return values. Every block
  of num_param_rngs adjacent schedules should be identical replicas with
  identical group keys to make it easier to associate scores that go together.
    schedules: array of schedules to score.
    augmented_schedule_params: list of augmented schedule parameters.
    schedule_group_keys: array of schedule group keys.
  """
  num_schedule_per_gen = (
      len(schedule_params) * len(base_lr_list) * num_param_rngs
  )
  schedules = np.zeros((
      num_schedule_per_gen,
      num_training_steps,
  ))
  augmented_schedule_params = []
  # All replicas in a group will have the same key. This way we can group
  # together all the different seeds for a given pair of schedule parameters
  # and base_lr.
  schedule_group_keys = []
  group_idx = 0
  current_idx = 0
  for base_lr in base_lr_list:
    base_lrs = [base_lr] * len(schedule_params)
    current_schedules = schedule_family.get_schedules(schedule_params, base_lrs)
    for schedule_idx in range(len(schedule_params)):
      for _ in range(num_param_rngs):
        schedules[current_idx] = current_schedules[schedule_idx]
        augmented_schedule_params.append(
            (schedule_params[schedule_idx].copy(), base_lr)
        )
        schedule_group_keys.append(group_idx)
        current_idx += 1
      group_idx += 1
  return schedules, augmented_schedule_params, np.array(schedule_group_keys)


# Because this function returns numpy arrays and not jax array futures, it is
# safe to use the timed decorator.
@utils.timed
def score_schedules(
    workload,
    num_parallel_schedules: int,
    data_rng: jax.Array,
    schedules: np.ndarray,  # shape (num_schedules, num_training_steps)
    params_rngs: jax.Array,  # shape (num_schedules,)
    reuse_data_rng_across_chunks: bool,
    pbar=None,
    ) -> dict[str, np.ndarray]:
  """Score schedules in chunks of num_parallel_schedules schedules at a time.

  Args:
    workload: workload to use for training and evaluation.
    num_parallel_schedules: number of schedules to score at a time.
    data_rng: random key to use for training data sampling.
    schedules: array of schedules to score.
    params_rngs: array of random keys to use for model initialization.
    reuse_data_rng_across_chunks: if True, use the same data rng for all chunks.
      If False, use a different data rng for each chunk. Results will only be
      exactly equivalent to the un-chunked version if the data_rng is reused
      across chunks. However, we would otherwise prefer to change the data_rng
      as frequently as possible.
    pbar: Optional tqdm-style progress bar to update.

  Returns:
    dict[str, np.ndarray]: dictionary of results, keyed by metric name. Each
    result is an array of shape (num_schedules,).
  """
  num_schedules = len(schedules)
  assert (schedules.shape[0] == params_rngs.shape[0])
  full_results = collections.defaultdict(lambda: np.zeros((num_schedules,)))
  num_chunks = math.ceil(num_schedules / num_parallel_schedules)
  for c in range(num_chunks):
    chunk_start = c * num_parallel_schedules
    chunk_end = (c + 1) * num_parallel_schedules
    logging.info('Starting chunk %d of %d', c, num_chunks)
    if reuse_data_rng_across_chunks:
      chunk_data_rng = data_rng
    else:
      chunk_data_rng = jax.random.fold_in(data_rng, c)
    _, chunk_results = workload.train_and_evaluate_models(
        schedules[chunk_start:chunk_end],
        params_rngs[chunk_start:chunk_end],
        chunk_data_rng,
    )
    for k, v in chunk_results.items():
      full_results[k][chunk_start:chunk_end] = np.asarray(v)
    if pbar is not None:
      chunk_size = len(params_rngs[chunk_start:chunk_end])
      pbar.update(chunk_size)
  return dict(full_results)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Setup config
  config = FLAGS.config
  logging.info('config: %s', config)
  if jax.process_index() == 0:
    logging.info(
        'Unless there is a preemption, this log line should appear only once.'
    )
  worker_id = FLAGS.worker_id
  logging.info('worker_id: %d', worker_id)
  workload_config = config.workload_config
  schedule_family_config = config.schedule_family_config
  schedule_family_class = schedule_families.get_schedule_family_class(
      schedule_family_config['schedule_type']
  )
  schedule_family = schedule_family_class(schedule_family_config)
  schedule_param_range = (
      base_schedule_family.add_prefix_to_schedule_param_dict(
          config.schedule_param_range
      )
  )
  allowed_schedule_param_keys = set(
      schedule_family.list_schedule_parameter_keys()
  )
  if any(
      k not in allowed_schedule_param_keys for k in schedule_param_range
  ):
    raise ValueError(
        'Some schedule parameters in the schedule_param_range are not valid for'
        ' the schedule family. Please double check the config.'
        f'search space: {schedule_param_range}, '
        f'allowed keys: {allowed_schedule_param_keys}'
    )
  search_config = config.search_config
  scoring_metric = search_config['scoring_metric']
  # TODO(gdahl): remove this check once any configs we care about are updated.
  if 'reuse_data_rng_across_chunks' not in search_config:
    raise ValueError(
        'reuse_data_rng_across_chunks must be specified. Add'
        ' `config.search_config.reuse_data_rng_across_chunks = True` to the'
        ' config file to reproduce the old behavior.'
    )
  reuse_data_rng_across_chunks = search_config['reuse_data_rng_across_chunks']

  # Set writer to None for OSS
  # pylint: disable=unused-variable
  writer = None

  # Setup workload, search algorithm, and schedule family
  workload_class = workloads.get_workload_class(
      workload_config.workload_name
  )
  workload = workload_class(workload_config)
  search_algorithm_class = search_algorithms.get_search_algorithm_class(
      search_config['type']
  )
  search_algorithm = search_algorithm_class(
      search_config,
      schedule_param_range,
  )

  # Setup top level key
  rng = jax.random.key(search_config['seed'])
  rng = jax.random.fold_in(rng, worker_id)
  num_param_rngs = search_config['num_param_rngs']
  # num_parallel_schedules controls how many schedules are trained and evaluated
  # in parallel. This is limited by the available memory on the device. A higher
  # value will result in faster throughput, but may lead to out-of-memory
  # errors.
  num_parallel_schedules = search_config['num_parallel_schedules']
  num_schedule_per_gen = (search_config['num_schedule_shapes_per_gen'] *
                          len(config.base_lr_list) *
                          num_param_rngs)
  logging.info('%d schedules/generation in chunks of %d schedules at a time',
               num_schedule_per_gen,
               num_parallel_schedules)
  num_generations = search_config['num_generation']
  pbar = None
  for gen_idx in range(num_generations):
    logging.info('generation: %d', gen_idx)

    # Fold in generation index to rng to generate new keys for each generation
    gen_rng = jax.random.fold_in(rng, gen_idx)
    data_rng, param_rng, schedules_rng = jax.random.split(gen_rng, 3)

    params_rngs = jax.random.split(param_rng, num_schedule_per_gen)

    # Generate new schedule parameters for this generation
    schedule_params = search_algorithm.get_schedule_params(
        schedules_rng,
        search_config['num_schedule_shapes_per_gen'],
    )

    # Augment schedule parameters with base learning rate
    (schedules, augmented_schedule_params, _) = (
        get_schedules_and_augmented_params(
            schedule_family,
            schedule_params,
            config.base_lr_list,
            config.total_steps,
            num_param_rngs,
        )
    )

    # Score schedules in chunks
    results, execution_time = score_schedules(
        workload,
        num_parallel_schedules,
        data_rng,
        schedules,
        params_rngs,
        reuse_data_rng_across_chunks,
        pbar
    )
    logging.info('execution_time: %f', execution_time)
    scores = results[scoring_metric]


    scores_in_gen = np.median(scores.reshape(-1, num_param_rngs), axis=1)
    best_score_in_gen = float(np.min(scores_in_gen))
    unique_augmented_schedule_params = [
        augmented_schedule_params[i]
        for i in range(0, len(augmented_schedule_params), num_param_rngs)
    ]
    search_algorithm.update(
        gen_idx, scores_in_gen, unique_augmented_schedule_params
    )

    # extract and log best score
    _, best_score = search_algorithm.get_best_solution()
    logging.info('best_score_in_gen: %f', best_score)

    # calc num_schedules per second
    num_schedules_per_sec = num_schedule_per_gen / execution_time
    if jax.process_index() == 0 and writer is not None:
      writer.write({
          'gen_idx': gen_idx,
          'execution_time': execution_time,
          'num_schedules_per_sec': num_schedules_per_sec,
          'best_score': best_score,
          'best_score_in_gen': best_score_in_gen,
      })

    logging.info('num_schedules_per_sec: %f', num_schedules_per_sec)

    del results, schedules, augmented_schedule_params
    del data_rng, param_rng, schedules_rng, params_rngs

  if pbar is not None:
    pbar.close()


  multihost_utils.sync_global_devices('end of program')


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)
