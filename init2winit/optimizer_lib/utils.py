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

"""Optimizer utilities."""

import flax


def create_mask(fn):
  """Creates a mask that maps fn over the leaves of a dict.

  Args:
    fn: function to apply taking
      - k: Tuple containing nodes (strings) in path to the leaf
      - v: The leaf

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


def extract_field(state, field_name):
  """Extract a field from a nested tuple (especially an optax optimizer state).

  Suppose that we'd like to extract Adam's "nu" pytree from an optax optimizer
  state that consists of a ScaleByAdam namedtuple wrapped inside of a
  combine.chain() tuple wrapped inside of an InjectHyperparamsState namedtuple
  wrapped inside of a GradientAccumulatorState namedtuple.  This function
  can do that.

  Args:
    state: An optax optimizer state.  This should be a nested tuple, meaning a
      tuple (potentially a namedtuple) that contains other tuples (or
      potentially namedtuples) in its slots.  Note that the state can contain
      non-tuple values as well (e.g. one of the slots in InjectHyperparamsState
      is a dict), but these will be ignored.
    field_name: (str) The name of the field we'd like to extract from the nested
      tuple.  For example, "nu" to extract Adam's second-moment accumulator.

  Returns:
    The value of a field with the given field name.  If there is more than
      one field with this name in the nested tuple, the behavior of this
      function is undefined.  Returns None if there is no field with the
      given name in "state".
  """
  assert isinstance(state, tuple)

  # If "state" is a namedtuple containing a field with the right name, return
  # the value in that field.
  if hasattr(state, '_fields') and field_name in state._fields:
    return getattr(state, field_name)

  # Else, recursively call this function on the slots of the tuple "state".
  for element in state:
    if isinstance(element, tuple):
      field = extract_field(element, field_name)
      if field is not None:
        return field

  # If we didn't find anything, return None.
  return None
