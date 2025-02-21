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

r"""Search algorithms for schedule parameters search."""

from typing import Any

from init2winit.projects.optlrschedule.search_algorithm import (
    coordinate_descent_search,
)
from init2winit.projects.optlrschedule.search_algorithm import grid_search
from init2winit.projects.optlrschedule.search_algorithm import random_search


SEARCH_ALGORITHMS = {
    'random': random_search.RandomSearch,
    'grid': grid_search.GridSearch,
    'coordinate_descent': coordinate_descent_search.CoordinateDescentSearch,
}


def get_search_algorithm_class(search_type: str) -> type[Any]:
  """Get search algorithm class for a given search type.

  Args:
    search_type: The type of search algorithm to get.

  Returns:
    The class of the search algorithm.

  Raises:
    ValueError: If the search type is not found.
  """
  try:
    return SEARCH_ALGORITHMS[search_type]
  except KeyError as e:
    raise ValueError(
        f'Search type {search_type} not found in {SEARCH_ALGORITHMS.keys()}'
    ) from e
