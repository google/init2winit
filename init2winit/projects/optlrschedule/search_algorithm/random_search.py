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

r"""Random search algorithm for schedule parameter optimization."""

from typing import Any, Dict, Tuple

from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import jax
import jax.numpy as jnp
import numpy as np


def sample_log_uniform(
    key: jax.Array, min_val: float, max_val: float
) -> jax.Array:
  """Sample from a log uniform distribution.

  Args:
      key (jax.Array): JAX key
      min_val (float): Minimum value of the distribution
      max_val (float): Maximum value of the distribution

  Returns:
      jax.Array: Sampled values from the log uniform distribution
  """

  assert min_val > 0
  assert max_val > 0
  assert min_val < max_val

  log_min = jnp.log(min_val)
  log_max = jnp.log(max_val)
  log_sample = jax.random.uniform(key, minval=log_min, maxval=log_max)
  return jnp.exp(log_sample)


class RandomSearch:
  """Implementation of a simple random search algorithm for optimization problems.

  Specifically designed for schedule parameter optimization with support for
  different schedule types.
  """

  def __init__(
      self,
      search_config: dict[str, Any],
      schedule_param_range: Dict[str, Tuple[float, float]],
  ):
    """Initialize the random search algorithm.

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
    self.log_scale_params = (
        base_schedule_family.add_prefix_to_schedule_param_key_list(
            search_config["log_scale_params"]
            if search_config.get("log_scale_params") is not None
            else []
        )
    )

    self._best_augmented_param = {}  # include base_lr
    self._best_score = np.inf
    self._generation = 0

  def get_schedule_params(
      self,
      rng: jax.Array,
      num_schedule_shapes_per_gen: int,
  ) -> list[Dict[str, float]]:
    """Generate new schedule parameters using random sampling.

    Args:
        rng (jax.Array): JAX key
        num_schedule_shapes_per_gen (int): Number of schedules to generate in
          this generation

    Returns:
        list[Dict[str, float]]: List of dictionary of sampled parameter values
    """

    schedule_params = []
    for idx in range(num_schedule_shapes_per_gen):
      subkey = jax.random.fold_in(rng, idx)
      schedule_param = self.get_schedule_param(subkey)
      schedule_params.append(schedule_param)
    return schedule_params

  def get_schedule_param(self, rng: jax.Array) -> Dict[str, float]:
    """Generate new schedule parameters using random sampling.

    Args:
        rng (jax.Array): JAX key

    Returns:
        Dict[str, float]: Dictionary of sampled parameter values
    """
    return self._sampling(rng)

  def _sampling(self, rng: jax.Array) -> Dict[str, float]:
    """Perform uniform random sampling within the specified parameter ranges.

    Args:
        rng (jax.Array): JAX key

    Returns:
        Dict[str, float]: Dictionary of sampled parameter values
    """
    schedule_param = {}
    counter = 0
    for param_key, (min_val, max_val) in self.schedule_param_range.items():
      subkey = jax.random.fold_in(rng, counter)
      if param_key in self.log_scale_params:
        val = sample_log_uniform(subkey, min_val, max_val)
      else:
        val = jax.random.uniform(subkey, minval=min_val, maxval=max_val)
      schedule_param[param_key] = float(val)
      counter += 1
    return schedule_param

  def update(
      self,
      gen_idx: int,
      scores: np.ndarray,
      augmented_schedule_params_in_gen: list[Tuple[Dict[str, float], float]],
  ) -> None:
    """Update internal state with results from the current generation.

    For random search, this only tracks the best solution found so far.

    Args:
        gen_idx (int): Current generation index
        scores (np.ndarray): Array of scores for each schedule in the current
          generation
        augmented_schedule_params_in_gen (list[Tuple[Dict[str, float], float]]):
          List of tuples, where each tuple contains the schedule parameters used
          in a trial and the corresponding base learning rate.
    """

    if isinstance(augmented_schedule_params_in_gen, dict):
      augmented_schedule_params_in_gen = [augmented_schedule_params_in_gen]
    if isinstance(scores, (float, int)):
      scores = np.array([scores])

    # assert same length for scores and params_in_gen
    if len(augmented_schedule_params_in_gen) != len(scores):
      raise ValueError(
          "Length of augmented_schedule_params_in_gen"
          f" {len(augmented_schedule_params_in_gen)} does not match length of"
          f" scores {len(scores)}"
      )

    self._generation = gen_idx

    # find best score in current generation
    best_score_in_gen = float(np.min(scores))
    best_idx = int(np.argmin(scores))

    current_best_augmented_param = augmented_schedule_params_in_gen[best_idx]

    # Update best solution if current generation found a better one
    if best_score_in_gen < self._best_score:
      self._best_score = best_score_in_gen
      self._best_augmented_param = current_best_augmented_param
      self._generation = gen_idx

  def get_best_solution(self) -> Tuple[Dict[str, float], float]:
    """Get the best solution found so far.

    Returns:
        Tuple[Dict[str, float], float]: Tuple of (best parameters, best score)
    """
    return (
        self._best_augmented_param,
        self._best_score,
    )
