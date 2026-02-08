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

"""Contains functions to partition a parameter pytree."""


def outer_key(x):
  return x[0]


def create_partition_flat_params_fn(key_map):
  """Partitions a flattened pytree according to the provided key_map.

  Subsets are determined by the kep_map which hashes the flattened model
  parameter keys into disjount groups. For example, if the flattened param
  tree is

  {('a', 'b'): 1.0,
  ('a', 'c'): 1.0,
  ('d', 'b'): 2.0}

  And we partition on the out_key then the output is
  {'a': {('a', 'b'): 1.0, ('a', 'c'): 1.0}
  'd': {('d', 'b'): 2.0}}.

  Args:
    key_map: Maps a tuple of strings to a hashable value.

  Returns:
    partition_flat_params, a function which returns a partitioned param
      dictionary.
  """
  def partition_flat_params(flat_params):
    subparam_groups = {}
    for tup in flat_params:
      mapped_key = key_map(tup)
      if mapped_key not in subparam_groups:
        subparam_groups[mapped_key] = {}
      subparam_groups[mapped_key][tup] = flat_params[tup]

    return subparam_groups
  return partition_flat_params


registry = {
    'outer_key': create_partition_flat_params_fn(outer_key),
}


def get_param_partition_fn(name):
  return registry[name]


# Used in test_model_debugger.py
def get_test_group(params):
  del params
  return ['B_0/C_0', 'C_0']


skip_analysis_registry = {
    'test_group': get_test_group,
}


def get_skip_analysis_fn(name):
  return skip_analysis_registry[name]

