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

r"""Tests for random_search.py."""

# import logging
from absl.testing import absltest
from init2winit.projects.optlrschedule.search_algorithm.grid_search import (
    GridSearch,
)
from init2winit.projects.optlrschedule.search_algorithm.random_search import (
    RandomSearch,
)
import jax
import numpy as np


class TestRandomSearch(absltest.TestCase):
  """Basic test cases for the RandomSearch class."""

  def _init(self):
    """Initialize the random search algorithm with fixed parameters."""

    # Define simple parameter ranges for testing
    self.param_ranges = {
        'x0': (0.01, 0.3),
        'y1': (0.1, 1.0),
    }

    self.search_config = {
        'num_generation': 1,
        'num_schedule_shapes_per_gen': 1,
        'seed': 0,
    }

    # Initialize algorithm
    self.algorithm = RandomSearch(
        search_config=self.search_config,
        schedule_param_range=self.param_ranges,
    )

  def test_sampling_within_bounds(self):
    """Test that sampled parameters are within specified bounds."""
    self._init()

    rng = jax.random.key(self.search_config['seed'])
    param = self.algorithm.get_schedule_param(rng)

    # Check if all parameters exist and are within bounds
    for param_name, (min_val, max_val) in self.param_ranges.items():
      self.assertIn(param_name, param)
      self.assertGreaterEqual(param[param_name], min_val)
      self.assertLessEqual(param[param_name], max_val)

  def test_update_and_get_best(self):
    """Test updating best solution and retrieving it."""
    self._init()

    # Update with first solution
    params1 = {'x0': 0.1, 'y1': 0.5}
    self.algorithm.update(1, 0.5, params1)

    # Update with better solution
    params2 = {'x0': 0.2, 'y1': 0.6}
    self.algorithm.update(2, 0.3, params2)

    # Get best solution and verify it's the better one
    best_param, best_score = self.algorithm.get_best_solution()

    self.assertEqual(best_score, 0.3)
    self.assertEqual(best_param, params2)

  def test_update_with_multiple_scores(self):
    """Test updating with multiple scores and parameters simultaneously."""
    self._init()

    # Multiple scores and corresponding parameters
    scores = np.array([0.5, 0.3, 0.8, 0.1])
    params_list = [
        {'x0': 0.1, 'y1': 0.5},  # score: 0.5
        {'x0': 0.2, 'y1': 0.6},  # score: 0.3
        {'x0': 0.3, 'y1': 0.7},  # score: 0.8
        {'x0': 0.4, 'y1': 0.8},  # score: 0.1 (best)
    ]

    # Update with multiple solutions at once
    self.algorithm.update(1, scores, params_list)

    # Get best solution and verify it corresponds to the lowest score
    best_param, best_score = self.algorithm.get_best_solution()

    # Best score should be 0.1 (minimum of scores)
    self.assertEqual(best_score, 0.1)
    # Best parameters should be the fourth set
    self.assertEqual(best_param, params_list[3])

  def test_update_multiple_generations(self):
    """Test updating with multiple scores across different generations."""
    self._init()

    # First generation
    scores1 = np.array([0.5, 0.3, 0.4])
    params_list1 = [
        {'x0': 0.1, 'y1': 0.5},
        {'x0': 0.2, 'y1': 0.6},  # best in generation 1
        {'x0': 0.3, 'y1': 0.7},
    ]
    self.algorithm.update(1, scores1, params_list1)

    # Verify best solution after first generation
    best_param, best_score = self.algorithm.get_best_solution()
    self.assertEqual(best_score, 0.3)
    self.assertEqual(best_param, params_list1[1])

    # Second generation with better solutions
    scores2 = np.array([0.2, 0.1, 0.4])  # 0.1 is better than previous best
    params_list2 = [
        {'x0': 0.4, 'y1': 0.8},
        {'x0': 0.5, 'y1': 0.9},  # new best
        {'x0': 0.6, 'y1': 1.0},
    ]
    self.algorithm.update(2, scores2, params_list2)

  def test_update_with_invalid_params_scores(self):
    """Test handling of mismatched parameters and scores."""
    self._init()

    scores = np.array([0.5, 0.3])
    params_list = [
        {'x0': 0.1, 'y1': 0.5},
    ]  # Less parameters than scores

    # Should raise ValueError due to index out of bounds
    with self.assertRaises(ValueError):
      self.algorithm.update(1, scores, params_list)


class TestGridSearch(absltest.TestCase):
  """Basic test cases for the GridSearch class."""

  def _init(self):
    """Initialize the grid search algorithm with fixed parameters."""
    self.param_ranges = {'steps_budget': (100, 1000), 'base_lr': (0.001, 0.1)}
    self.num_grid_points = {'steps_budget': 2, 'base_lr': 3}

    self.search_config = {
        'num_generation': 1,
        'num_schedule_shapes_per_gen': 1,
        'seed': 0,
        'num_grid_points': self.num_grid_points,
        'log_scale_params': [],
    }

    self.algorithm = GridSearch(
        search_config=self.search_config,
        schedule_param_range=self.param_ranges,
    )

  def test_init_grid_points(self):
    """Test if grid points are correctly generated during initialization."""
    self._init()

    # Check if grid points are correctly generated for each parameter
    self.assertLen(self.algorithm.grid_points['steps_budget'], 2)
    self.assertLen(self.algorithm.grid_points['base_lr'], 3)

    # Check if the values are within the specified ranges
    np.testing.assert_array_almost_equal(
        self.algorithm.grid_points['steps_budget'], np.array([100, 1000])
    )
    np.testing.assert_array_almost_equal(
        self.algorithm.grid_points['base_lr'], np.array([0.001, 0.0505, 0.1])
    )

  def test_total_combinations(self):
    """Test if total number of combinations is correct."""
    self._init()

    expected_combinations = (
        self.num_grid_points['steps_budget'] * self.num_grid_points['base_lr']
    )
    self.assertEqual(self.algorithm.total_combinations, expected_combinations)

  def test_get_schedule_params(self):
    """Test if get_schedule_params returns correct number of parameters."""
    self._init()

    rng = jax.random.key(self.search_config['seed'])
    num_schedule = 3
    params_batch = self.algorithm.get_schedule_params(
        rng=rng, num_schedule_shapes_per_gen=num_schedule
    )

    # Check if correct number of parameters are returned
    self.assertLen(params_batch, num_schedule)

    # Check if parameters are within ranges
    for params in params_batch:
      self.assertBetween(
          params['steps_budget'],
          self.param_ranges['steps_budget'][0],
          self.param_ranges['steps_budget'][1],
      )
      self.assertBetween(
          params['base_lr'],
          self.param_ranges['base_lr'][0],
          self.param_ranges['base_lr'][1],
      )

  def test_update_best_solution(self):
    """Test if update method correctly updates the best solution."""
    self._init()

    # Get initial parameters
    rng = jax.random.key(self.search_config['seed'])
    params_batch = self.algorithm.get_schedule_params(
        rng=rng, num_schedule_shapes_per_gen=2
    )

    # Update with some scores
    scores = np.array([0.5, 0.3])
    self.algorithm.update(
        gen_idx=0, scores=scores, augmented_schedule_params_in_gen=params_batch
    )

    # Check if best score is updated correctly
    best_params, best_score = self.algorithm.get_best_solution()
    self.assertEqual(best_score, 0.3)
    self.assertEqual(best_params, params_batch[1])

  def test_exhaustive_search(self):
    """Test if algorithm explores all combinations."""
    self._init()

    total_combinations = self.algorithm.total_combinations
    all_params = []

    while len(all_params) < total_combinations:
      rng = jax.random.key(self.search_config['seed'])
      params = self.algorithm.get_schedule_params(
          rng=rng, num_schedule_shapes_per_gen=1
      )
      if not params:
        break
      all_params.extend(params)

    # Check if all combinations were explored
    self.assertLen(all_params, total_combinations)

    # Check if all parameters are unique
    unique_params = {tuple(sorted(p.items())) for p in all_params}
    self.assertLen(unique_params, total_combinations)


if __name__ == '__main__':
  absltest.main()
