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

"""Mask utilities."""
import flax


def create_mask(fn):
  """Creates a mask that maps fn over the leaves of a dict.

  Args:
    fn: function to apply taking - k: Tuple containing nodes (strings) in path
      to the leaf - v: The leaf

  Returns:
    mask: function that takes dict and returns mapped dict
  """

  def mask(data):
    flattened_dict = flax.traverse_util.flatten_dict(data)
    return flax.traverse_util.unflatten_dict(
        {k: fn(k, v) for k, v in flattened_dict.items()})

  return mask


def create_weight_decay_mask():
  return create_mask(
      lambda p, _: 'bias' not in p and not p[-2].startswith('BatchNorm'))

mask_registry = {
    'bias_bn': create_weight_decay_mask(),
}
