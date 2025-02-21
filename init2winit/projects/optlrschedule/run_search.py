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

r"""execute random search for optlrschedule."""

from collections.abc import Sequence
import datetime
import logging
import os
import time

from absl import app
from absl import flags
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
    'worker_id',
    1,
    'Worker id to fold into top level rng key. Note that to exactly reproduce a'
    ' previous experiment, you must use the same worker_id.',
)


def get_schedules_and_augmented_params(
    schedule_family_config: dict[str, float],
    schedule_params: list[dict[str, float]],
    base_lr_list: list[float],
    num_training_steps: int,
) -> tuple[np.ndarray, list[tuple[dict[str, float], float]]]:
  """Get schedules and augmented schedule parameters."""

  schedule_family_class = schedule_families.get_schedule_family_class(
      str(schedule_family_config['schedule_type'])
  )
  schedule_family = schedule_family_class(schedule_family_config)

  num_schedule_per_gen = len(schedule_params) * len(base_lr_list)
  schedules = np.zeros((
      num_schedule_per_gen,
      num_training_steps,
  ))
  augmented_schedule_params = []
  current_idx = 0
  for base_lr in base_lr_list:
    base_lrs = [base_lr] * len(schedule_params)
    current_schedules = schedule_family.get_schedules(schedule_params, base_lrs)

    for schedule_idx in range(len(schedule_params)):
      schedules[current_idx] = current_schedules[schedule_idx]
      augmented_schedule_params.append(
          (schedule_params[schedule_idx].copy(), base_lr)
      )
      current_idx += 1
  return schedules, augmented_schedule_params


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Setup config
  config = FLAGS.config
  logging.info('config: %s', config)
  worker_id = FLAGS.worker_id
  if jax.process_index() == 0:
    logging.info(
        'Unless there is a preemption, this log line should appear only once.'
    )
  logging.info('worker_id: %d', worker_id)
  workload_config = config.workload_config
  schedule_family_config = config.schedule_family_config
  schedule_param_range = (
      base_schedule_family.add_prefix_to_schedule_param_dict(
          config.schedule_param_range
      )
  )
  search_config = config.search_config
  scoring_metric = search_config['scoring_metric']

  # Set writer to None for OSS
  # pylint: disable=unused-variable
  writer = None

  # Setup workload, search algorithm, and schedule family
  workload_class = workloads.get_workload_class(workload_config.workload_name)
  workload = workload_class(workload_config)
  search_algorithm_class = search_algorithms.get_search_algorithm_class(
      search_config['type']
  )
  search_algorithm = search_algorithm_class(
      search_config,
      schedule_param_range,
  )
  num_schedule_per_gen = search_config['num_schedule_shapes_per_gen'] * len(
      config.base_lr_list
  )

  # Setup top level key
  rng = jax.random.key(search_config['seed'])
  rng = jax.random.fold_in(rng, worker_id)
  num_param_rngs = search_config['num_param_rngs']

  for gen_idx in range(search_config['num_generation']):

    logging.info('generation: %d', gen_idx)

    # Fold in generation index to rng to generate new keys for each generation
    gen_rng = jax.random.fold_in(rng, gen_idx)
    data_rng, param_rng, schedules_rng = jax.random.split(gen_rng, 3)
    replica_rngs = jax.random.split(param_rng, num_param_rngs)

    # params_rngs = jax.random.split(param_rng, num_schedule_per_gen)
    params_rngs = [
        jax.random.split(r, num_schedule_per_gen) for r in replica_rngs
    ]

    # Generate new schedule parameters for this generation
    schedule_params = search_algorithm.get_schedule_params(
        schedules_rng,
        search_config['num_schedule_shapes_per_gen'],
    )

    # Augment schedule parameters with base learning rate
    schedules, augmented_schedule_params = get_schedules_and_augmented_params(
        schedule_family_config,
        schedule_params,
        config.base_lr_list,
        config.total_steps,
    )

    start_time = time.time()

    all_results = []
    for current_params_rngs in params_rngs:
      results = workload.train_and_evaluate_models(
          schedules,
          current_params_rngs,
          data_rng,
      )[1]
      scores = np.asarray(results[scoring_metric].block_until_ready())

      all_results.append(results)

    scores_in_gen = np.median(
        np.concatenate(
            [r[scoring_metric][:, None] for r in all_results], axis=1
        ),
        axis=1,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info('execution_time: %f', execution_time)

    best_score_in_gen = float(np.min(scores_in_gen))

    # feedback score in current generation to search_algorithm
    search_algorithm.update(gen_idx, scores_in_gen, augmented_schedule_params)

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

    del all_results, schedules, augmented_schedule_params
    del data_rng, param_rng, schedules_rng, params_rngs


  multihost_utils.sync_global_devices('end of program')


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)
