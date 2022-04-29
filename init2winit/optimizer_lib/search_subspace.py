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

"""Algorithms for narrowing hyperparameter search spaces.

TODO(dsuo): suport discrete hparams.

"""
import itertools
import re

from absl import logging
import numpy as np
import pandas as pd


def find_best_cube(trials,
                   objective,
                   search_space,
                   k,
                   cube_sizes,
                   cube_strides,
                   min_objective=True,
                   **kwargs):
  """Find best cube in original search space."""
  del kwargs

  # Get the top k trials as ordered by objective
  top_k_obj = trials[objective].apply(lambda x: x[-1]).sort_values(
      ascending=min_objective).head(n=k)
  top_k_idx = top_k_obj.index
  hp_keys = [f'hps.{key}' for key in search_space.keys()]
  top_k_df = pd.concat((trials.loc[top_k_idx][hp_keys], top_k_obj), axis=1)

  # Compute starting points of hyperparam cubes.
  cube_start_points = {}
  cube_end_points = {}
  for key, hp in search_space.items():
    if hp['scale_type'] == 'UNIT_LOG_SCALE':
      hp['mapped_range'] = [
          np.log10(float(hp['min_value'])),
          np.log10(float(hp['max_value']))
      ]
    elif hp['scale_type'] == 'UNIT_LINEAR_SCALE':
      hp['mapped_range'] = [hp['min_value'], hp['max_value']]
    cube_start_points[key] = np.arange(hp['mapped_range'][0],
                                       hp['mapped_range'][1], cube_strides[key])
    cube_end_points[key] = cube_start_points[key] + cube_sizes[key]

    if hp['scale_type'] == 'UNIT_LOG_SCALE':
      cube_start_points[key] = np.power(10, cube_start_points[key])
      cube_end_points[key] = np.power(10, cube_end_points[key])

  cube_start_points = list(itertools.product(*cube_start_points.values()))
  cube_end_points = list(itertools.product(*cube_end_points.values()))

  best_cube = None
  best_cube_mean = float('inf') if min_objective else -1.0 * float('inf')
  best_cube_trials = None
  best_cube_top_trial_included = False

  # Find trials from top k in each cube.
  for cube_start_point, cube_end_point in zip(cube_start_points,
                                              cube_end_points):
    points_df = top_k_df
    top_trial_included = True
    for i, (start, end) in enumerate(zip(cube_start_point, cube_end_point)):
      series = top_k_df.iloc[:, i]
      selectors = (series >= start) & (series <= end)
      if top_trial_included:
        top_trial_included = selectors.iloc[0] or False
      points_df = points_df[selectors]

    # Check if we have found a better cube.
    change_flag = points_df[objective].mean(
    ) < best_cube_mean if min_objective else points_df[objective].mean(
    ) > best_cube_mean

    # Record best cube.
    if change_flag:
      best_cube = cube_start_point
      best_cube_mean = points_df[objective].mean()
      best_cube_trials = points_df
      best_cube_top_trial_included = top_trial_included

  new_search_space = {}
  for val, (key, hp) in zip(best_cube, search_space.items()):
    if hp['scale_type'] == 'UNIT_LOG_SCALE':
      new_search_space[key] = (np.power(10., np.log10(val)),
                               np.power(10.,
                                        np.log10(val) + cube_sizes[key]))
    elif hp['scale_type'] == 'UNIT_LINEAR_SCALE':
      new_search_space[key] = (val, val + cube_sizes[key])

  num_trials = len(best_cube_trials)
  logging.info('Total number of trials included in the reported cube is %d',
               num_trials)
  if not best_cube_top_trial_included:
    logging.info('Warning the best trial was not included in the cube')

  return dict(
      search_space=new_search_space,
      trials=best_cube_trials,
      mean_trial_objective=best_cube_mean,
      contains_best_trial=best_cube_top_trial_included,
  )
