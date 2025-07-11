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

"""Shared utilities for unit tests."""

import functools

import jax
import jax.numpy as jnp


def pytree_equal(tree1, tree2):
  try:
    equal_tree = jax.tree_util.tree_map(jnp.array_equal, tree1, tree2)
    return jax.tree_util.tree_reduce(lambda x, y: x and y, equal_tree)
  # The tree_utils will raise TypeErrors if structures don't match.
  except TypeError:
    return False


def pytree_allclose(tree1, tree2, rtol=1e-5):
  try:
    allclose = functools.partial(jnp.allclose, rtol=rtol)
    equal_tree = jax.tree_util.tree_map(allclose, tree1, tree2)
    return jax.tree_util.tree_reduce(lambda x, y: x and y, equal_tree)
  # The tree_utils will raise TypeErrors if structures don't match.
  except TypeError:
    return False
