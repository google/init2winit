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

r"""Grid search algorithm for schedule parameter optimization."""

import itertools
import logging
from typing import Any, Dict, List, Tuple

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import jax
import numpy as np


class GridSearch:
  """Implementation of a grid search algorithm for optimization problems.

  Designed for schedule parameter optimization with support for different
  schedule types. Systematically explores all combinations of parameters
  within specified ranges using a grid-based approach.
  """

  def __init__(
      self,
      search_config: dict[str, Any],
      schedule_param_range: Dict[str, Tuple[float, float]],
  ):
    """Initialize the grid search algorithm.

    Args:
        search_config (dict[str, Any]): Configuration for the search algorithm
          including seed, number of generations, and number of schedule shapes
          per generation
        schedule_param_range (Dict[str, Tuple[float, float]]): Dictionary of
          parameter ranges where each key is a parameter name and value is a
          tuple of (min, max)
    """
    self.search_config = search_config
    self.schedule_param_range = schedule_param_range
    self.num_grid_points = search_config["num_grid_points"]
    self.log_scale_params = (
        base_schedule_family.add_prefix_to_schedule_param_key_list(
            search_config["log_scale_params"]
        )
    )

    # Generate grid points for each parameter
    self.grid_points = self._generate_grid_points()

    # Generate all possible combinations
    self.param_combinations = list(
        itertools.product(*self.grid_points.values())
    )
    self.total_combinations = len(self.param_combinations)
    self.current_combo_idx = 0

    # Initialize internal state
    self.internal_state = {
        "best_augumented_param": {},
        "best_score": np.inf,
        "generation": 0,
    }

  def _generate_grid_points(self) -> Dict[str, np.ndarray]:
    """Generate grid points for each parameter within its range.

    Returns:
        Dict[str, np.ndarray]: Dictionary of arrays containing grid points
            for each parameter
    """
    grid_points = {}
    for param_key, (min_val, max_val) in self.schedule_param_range.items():
      num_points_per_param = self.num_grid_points[param_key]
      if param_key in self.log_scale_params:
        grid_points[param_key] = np.logspace(
            np.log10(min_val), np.log10(max_val), num_points_per_param
        )
      else:
        grid_points[param_key] = np.linspace(
            min_val, max_val, num_points_per_param
        )
    return grid_points

  def get_schedule_params(
      self,
      rng: jax.Array,
      num_schedule_shapes_per_gen: int,
  ) -> List[Dict[str, float]]:
    """Generate next batch of schedule parameters from the grid.

    Args:
        rng (jax.Array): JAX key (just for keeping consistency with random
          search)
        num_schedule_shapes_per_gen (int): Number of schedules to generate in
          this generation

    Returns:
        List[Dict[str, float]]: List of dictionary of parameter values
    """
    del rng
    schedule_params = []
    param_keys = list(self.schedule_param_range.keys())

    for _ in range(num_schedule_shapes_per_gen):
      if self.current_combo_idx >= self.total_combinations:
        break

      combo = self.param_combinations[self.current_combo_idx]
      param_dict = {key: float(value) for key, value in zip(param_keys, combo)}
      schedule_params.append(param_dict)
      self.current_combo_idx += 1

    return schedule_params

  def update(
      self,
      gen_idx: int,
      scores: np.ndarray,
      augmented_schedule_params_in_gen: list[Tuple[Dict[str, float], float]],
  ) -> None:
    """Update internal state with results from the current generation.

    Args:
        gen_idx (int): Current generation index
        scores (np.ndarray): Array of scores for each schedule in the current
          generation
        augmented_schedule_params_in_gen (list[Tuple[Dict[str, float], float]]):
          List of tuples, where each tuple contains the schedule parameters used
          in a trial and the corresponding score.
    """
    self.internal_state["generation"] = gen_idx

    # Convert single score to numpy array
    if isinstance(scores, (float, int)):
      scores = np.array([scores])

    # Handle single parameter dict
    if isinstance(augmented_schedule_params_in_gen, dict):
      augmented_schedule_params_in_gen = [augmented_schedule_params_in_gen]

    # Find best score in current generation
    best_score_in_gen = float(np.min(scores))
    best_idx = int(np.argmin(scores))

    current_best_augmented_param = augmented_schedule_params_in_gen[best_idx]

    # Update best solution if current generation found a better one
    if best_score_in_gen < self.internal_state["best_score"]:
      self.internal_state["best_score"] = best_score_in_gen
      self.internal_state["best_augumented_param"] = (
          current_best_augmented_param
      )
      self.internal_state["generation"] = gen_idx

    logging.info("generation: %d", self.internal_state["generation"])
    logging.info("best_score: %f", self.internal_state["best_score"])
    logging.info(
        "grid search progress: %d/%d combinations",
        self.current_combo_idx,
        self.total_combinations,
    )

  def get_best_solution(self) -> Tuple[Dict[str, float], float]:
    """Get the best solution found so far.

    Returns:
        Tuple[Dict[str, float], float]: Tuple of (best parameters, best score)
    """
    return (
        self.internal_state["best_augumented_param"],
        self.internal_state["best_score"],
    )
