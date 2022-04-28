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

"""Tests for search_subspace."""

from absl.testing import absltest
from init2winit.optimizer_lib import search_subspace
from numpy import testing
import pandas as pd

trials = pd.DataFrame.from_dict({
    'hps.beta1': [2e-5, 4e-2, 6e-3, 1e-7, 3e-4],
    'hps.beta2': [3e-1, 5e-4, 6e-7, 7e-1, 1e-2],
    'hps.int': [1, 4, 5, 10, 2],
    'hps.float': [1., 4., 5., 10., 2.],
    'objective': [[0.12451], [2.2312], [0.123123], [0.5325], [0.6423]],
})

k = 10
objective = 'objective'


class SearchSubspaceTest(absltest.TestCase):
  """Test search_subspace."""

  def test_find_best_cube_log(self):
    """Test find best cube of log params."""

    search_space = {
        'beta1': {
            'min_value': 1e-8,
            'max_value': 1e-1,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LOG_SCALE'
        }
    }
    cube_sizes = {'beta1': 2}
    cube_strides = {'beta1': 1}

    result = search_subspace.find_best_cube(
        trials,
        objective,
        search_space,
        k=k,
        cube_sizes=cube_sizes,
        cube_strides=cube_strides)

    testing.assert_equal(result['contains_best_trial'], False)
    testing.assert_equal(result['mean_trial_objective'], 0.12451)
    testing.assert_array_equal(result['search_space']['beta1'], [1e-6, 1e-4])
    testing.assert_equal(len(result['trials']), 1)

  def test_find_best_cube_int(self):
    """Test find best cube of int params."""
    search_space = {
        'int': {
            'min_value': 1,
            'max_value': 12,
            'type': 'INTEGER',
            'scale_type': 'UNIT_LINEAR_SCALE'
        }
    }
    cube_sizes = {'int': 2}
    cube_strides = {'int': 1}

    result = search_subspace.find_best_cube(
        trials,
        objective,
        search_space,
        k=k,
        cube_sizes=cube_sizes,
        cube_strides=cube_strides)

    testing.assert_equal(result['contains_best_trial'], True)
    testing.assert_equal(result['mean_trial_objective'], 0.123123)
    testing.assert_array_equal(result['search_space']['int'], [5, 7])
    testing.assert_equal(len(result['trials']), 1)

  def test_find_best_cube_float(self):
    """Test find best cube of float params."""
    search_space = {
        'float': {
            'min_value': 1.,
            'max_value': 12.,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LINEAR_SCALE'
        }
    }
    cube_sizes = {'float': 2}
    cube_strides = {'float': 1}

    result = search_subspace.find_best_cube(
        trials,
        objective,
        search_space,
        k=k,
        cube_sizes=cube_sizes,
        cube_strides=cube_strides)

    testing.assert_equal(result['contains_best_trial'], True)
    testing.assert_equal(result['mean_trial_objective'], 0.123123)
    testing.assert_array_equal(result['search_space']['float'], [5.0, 7.0])
    testing.assert_equal(len(result['trials']), 1)

  def test_find_best_cube_2d(self):
    """Test find best cube of 2 params."""
    search_space = {
        'beta1': {
            'min_value': 1e-8,
            'max_value': 1e-1,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LOG_SCALE'
        },
        'float': {
            'min_value': 0.,
            'max_value': 12.,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LINEAR_SCALE'
        }
    }
    cube_sizes = {'beta1': 2, 'float': 2}
    cube_strides = {'beta1': 2, 'float': 2}

    result = search_subspace.find_best_cube(
        trials,
        objective,
        search_space,
        k=k,
        cube_sizes=cube_sizes,
        cube_strides=cube_strides)

    testing.assert_equal(result['contains_best_trial'], True)
    testing.assert_equal(result['mean_trial_objective'], 0.123123)
    testing.assert_array_equal(result['search_space']['beta1'], [1e-4, 1e-2])
    testing.assert_array_equal(result['search_space']['float'], [4.0, 6.0])
    testing.assert_equal(len(result['trials']), 1)

  def test_find_best_cube_max_objective(self):
    """Test find best cube with max objective."""
    search_space = {
        'beta1': {
            'min_value': 1e-8,
            'max_value': 1e-1,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LOG_SCALE'
        },
        'float': {
            'min_value': 0.,
            'max_value': 12.,
            'type': 'DOUBLE',
            'scale_type': 'UNIT_LINEAR_SCALE'
        }
    }
    cube_sizes = {'beta1': 2, 'float': 2}
    cube_strides = {'beta1': 2, 'float': 2}

    result = search_subspace.find_best_cube(
        trials,
        objective,
        search_space,
        k=k,
        cube_sizes=cube_sizes,
        cube_strides=cube_strides,
        min_objective=False)

    testing.assert_equal(result['contains_best_trial'], True)
    testing.assert_equal(result['mean_trial_objective'], 2.2312)
    testing.assert_array_equal(result['search_space']['beta1'], [1e-2, 1.0])
    testing.assert_array_equal(result['search_space']['float'], [2.0, 4.0])
    testing.assert_equal(len(result['trials']), 1)


if __name__ == '__main__':
  absltest.main()
