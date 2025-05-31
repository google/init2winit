# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

r"""Coordinate descent search algorithm for schedule parameter optimization."""

import itertools
from typing import Any, Dict, Generator, List, Tuple

from init2winit.projects.optlrschedule import log_utils
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import jax
import numpy as np
import pandas as pd


# TODO(hnaganuma): Refactor this function to a common util file
def batch_unique_schedules_base_lrs_and_metadata(
    df: pd.DataFrame, unique_schedule_batch_size: int
) -> Generator[
    Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]]], None, None
]:
  """Batch schedules into groups of a specified size.

  This function takes a DataFrame containing schedule data, splits it into
  batches of size `unique_schedule_batch_size`, and processes each batch
  to extract schedule parameters, base learning rates, and metadata.

  Args:
      df (pd.DataFrame): DataFrame containing schedule data with scores,
        parameters, and metadata.
      unique_schedule_batch_size (int): Number of schedules to include in each
        batch.

  Yields:
      Generator[Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]]]]:
      - A list of schedule parameters for each batch.
      - A list of base learning rates for each batch.
      - A list of metadata corresponding to the schedules.
  """

  total_rows = len(df)
  current_idx = 0

  while current_idx < total_rows:
    end_idx = min(current_idx + unique_schedule_batch_size, total_rows)
    batch_dicts = df.iloc[current_idx:end_idx].to_dict("records")

    results = [
        log_utils.split_row_dict_into_schedule_params_base_lr_and_metadata(
            d, rank=i
        )
        for i, d in enumerate(batch_dicts, start=current_idx)
    ]

    schedule_params, base_lrs, metadata = zip(*results)

    yield (list(schedule_params), list(base_lrs), list(metadata))

    current_idx = end_idx


class CoordinateDescentSearch:
  """A coordinate descent search algorithm for schedule parameter optimization.

  This class implements an optimization algorithm that searches for the
  best schedule parameters by perturbing and testing different configurations
  of hyperparameters.

  Attributes:
      search_config (dict[str, Any]): Configuration parameters for the search
        algorithm, including: - `cns_path` (str): Path to the file containing
        the schedule data. - `top_k` (int): Number of top schedules to use for
        the search. - `num_schedule_shapes_per_gen` (int): Number of schedules
        to evaluate per generation. - `num_sweep_points` (int, optional): Number
        of points to generate for sweeping along each parameter axis. Default is
        4. - `per_axis_search_factor` (float, optional): Factor by which the
        search range is divided to perturb parameters. Default is 2.0. -
        `log_scale_params` (list[str], optional): List of parameter names to
        perturb on a logarithmic scale. Default is an empty list. -
        `num_generation` (int, optional): Number of generations for the search.
        If not provided, it will be calculated automatically based on other
        parameters.
      schedule_param_range (dict[str, Tuple[float, float]]): A dictionary
        mapping parameter names to their valid ranges (min, max). Each parameter
        must have a corresponding range specified.
      num_total_schedule (int): Total number of schedules that will be generated
        during the search.
      num_sweep_points (int): Number of points to generate for sweeping along
        each parameter axis.
      per_axis_search_factor (float): Factor by which the search range is
        divided to perturb parameters.
      log_scale_params (list[str]): List of parameter names to perturb on a
        logarithmic scale.
      schedules_df (pd.DataFrame): DataFrame containing the top k schedules with
        their base learning rates and scores.
  """

  def __init__(
      self,
      search_config: dict[str, Any],
      schedule_param_range: Dict[str, Tuple[float, float]],
  ):
    """Initialize the perturbed search algorithm.

    Args:
        search_config (dict[str, Any]): Configuration for the search algorithm
        schedule_param_range (dict[str, Tuple[float, float]]): param ->
          (min_val, max_val)
    """
    self.search_config = search_config
    self.schedule_param_range = schedule_param_range

    # Initialize internal state variables
    self._best_augmented_param = {}
    self._best_score = np.inf
    self._generation = 0

    self.num_sweep_points = search_config.get("num_sweep_points", 4)
    self.per_axis_search_factor = search_config.get(
        "per_axis_search_factor", 2.0
    )
    self.log_scale_params = search_config.get("log_scale_params", [])

    # Load schedule params, base_lr and metadata to do coordinate descent search
    top_reduced_schedules_and_metadata_df = (
        log_utils.load_schedules_to_evaluate(
            xid=search_config["xid"],
            top_k=search_config["num_top_schedule_shapes"],
            metric="score_median",
            num_schedule_shapes_for_sampling=search_config.get(
                "num_schedule_shapes_for_sampling"
            ),
            rename_columns=True,
        )
    )

    self.schedules_df = top_reduced_schedules_and_metadata_df

    # Get number of elements (keys) in base_param from the first record
    first_record = self.schedules_df.iloc[0].to_dict()
    # Extract only keys that exist in schedule_param_range
    filtered_first_record = {
        k: first_record[k]
        for k in self.schedule_param_range.keys()
        if k in first_record
    }
    num_base_param_keys = len(filtered_first_record)
    num_rows = len(self.schedules_df)

    # num_total_schedule = number of DataFrame rows × number of base_param keys
    # × num_sweep_points
    self.num_total_schedule = (
        num_rows * num_base_param_keys * (self.num_sweep_points + 1)
    )

    # Automatically determine number of generations
    self.search_config["num_generation"] = int(
        np.ceil(
            self.num_total_schedule
            / search_config["num_schedule_shapes_per_gen"]
        )
    )

    # Use generator to create all schedules on demand
    self._schedule_generator = self._create_schedule_generator()
    # Counter for number of generated items
    self._generated_count = 0

  def _generate_dummy_schedule(self) -> Dict[str, float]:
    """Generate a dummy schedule with random parameters within specified ranges.

    Returns:
        Dict containing randomly generated schedule parameters
    """
    dummy_schedule = {}
    counter = 0
    for param_key, (min_val, max_val) in self.schedule_param_range.items():
      dummy_schedule[param_key] = float((min_val + max_val) / 2)
      counter += 1

    return dummy_schedule

  def _create_schedule_generator(
      self,
  ) -> Generator[Dict[str, float], None, None]:
    """Generator that reads DataFrame in batches and yields one schedule at a time while perturbing parameters."""
    base_params_gen = batch_unique_schedules_base_lrs_and_metadata(
        self.schedules_df, self.search_config["num_schedule_shapes_per_gen"]
    )
    for schedule_params_batch, base_lrs, metadata in base_params_gen:
      del base_lrs, metadata  # Unused

      # Perturb parameters for each and yield
      for base_param in schedule_params_batch:
        # Extract only keys that exist in schedule_param_range
        # (base_param is expected to have same structure but different values)
        filtered_params = {
            k: base_param[k]
            for k in self.schedule_param_range.keys()
            if k in base_param
        }
        # Generate perturbations for each param_key
        for param_key, base_value in filtered_params.items():
          perturbed_values = self._generate_perturbed_values(
              base_value, param_key
          )
          for val in perturbed_values:
            new_params = dict(filtered_params)
            new_params[param_key] = val
            yield new_params

  def _generate_perturbed_values(
      self, base_value: float, param_key: str
  ) -> List[float]:
    """Generate a set of perturbed values for a parameter, centered at a base value.

    This method generates a range of values around a given base_value for a
    specific parameter (param_key). The perturbation can be done either on
    a logarithmic scale or a linear scale, depending on whether the parameter
    is included in the log_scale_params attribute.

    The perturbed values are clipped to ensure they remain within the valid
    range specified in schedule_param_range.

    Args:
        base_value (float): The base value of the parameter around which the
          perturbation is performed.
        param_key (str): The name of the parameter being perturbed.

    Returns:
        List[float]: A sorted list of unique perturbed values, including the
        base value. The list includes values within the range specified for
        the parameter in schedule_param_range.

    Key Details:
        - If the parameter is in log_scale_params, perturbation is performed
          using logarithmic scaling, and values are generated as multiplicative
          factors around the base_value.
        - If the parameter is not in log_scale_params, perturbation is
          performed linearly by adding/subtracting values within a specified
          sweep range.
        - The number of perturbed values is determined by num_sweep_points.
        - All generated values are clipped to the valid range
          (min_val, max_val) of the parameter to ensure they are feasible.
        - Duplicates are removed from the resulting list, and values are sorted
          in ascending order.
    """
    min_val, max_val = self.schedule_param_range[param_key]
    values = []

    if param_key in self.log_scale_params:
      # log scale
      multipliers = np.logspace(
          -np.log10(self.per_axis_search_factor),
          np.log10(self.per_axis_search_factor),
          self.num_sweep_points,
      )
      for m in multipliers:
        candidate = base_value * m
        values.append(max(min_val, min(candidate, max_val)))
    else:
      # linear scale
      sweep_range = (max_val - min_val) / self.per_axis_search_factor
      sweep = np.linspace(
          -sweep_range / 2,
          sweep_range / 2,
          self.num_sweep_points,
      )
      for p in sweep:
        candidate = base_value + p
        values.append(max(min_val, min(candidate, max_val)))

    # Include base_value
    values.append(base_value)

    # Remove duplicates & sort
    return sorted(list(set(values)))

  def get_schedule_params(
      self,
      rng: jax.Array,
      num_schedule_shapes_per_gen: int,
  ) -> List[Dict[str, float]]:
    """Return num_schedule_shapes_per_gen number of schedules."""
    del rng  # Unused

    batch = list(
        itertools.islice(self._schedule_generator, num_schedule_shapes_per_gen)
    )
    self._generated_count += len(batch)

    # Fill remaining slots with dummy schedules if needed
    remaining_slots = num_schedule_shapes_per_gen - len(batch)
    if remaining_slots > 0:
      for _ in range(remaining_slots):
        dummy_schedule = self._generate_dummy_schedule()
        batch.append(dummy_schedule)

    # TODO(hnaganuma): temporary solution to make consistent param prefix
    # Eventual solution will be to assume loaded schedule params
    # have already been prefixed

    # convert to list of schedule params with prefix
    batch = [
        base_schedule_family.add_prefix_to_schedule_param_dict(param)
        for param in batch
    ]

    return batch

  def update(
      self,
      gen_idx: int,
      scores: np.ndarray,
      augmented_schedule_params_in_gen: List[Dict[str, float]],
  ) -> None:
    """Update internal state variables with results from the current generation."""
    self._generation = gen_idx

    # Consider the case where there's only one score
    if isinstance(scores, (float, int)):
      scores = np.array([scores])

    best_score_in_gen = float(np.min(scores))
    best_idx = int(np.argmin(scores))

    if best_score_in_gen < self._best_score:
      self._best_score = best_score_in_gen
      self._best_augmented_param = augmented_schedule_params_in_gen[best_idx]

  def get_best_solution(self) -> Tuple[Dict[str, float], float]:
    """Get the best solution found so far."""
    return (
        self._best_augmented_param,
        self._best_score,
    )
