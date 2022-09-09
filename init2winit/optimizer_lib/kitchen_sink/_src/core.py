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

"""Modularizing optimization ideas.

This project seeks to take ideas in optimization (e.g., scale decay,
momentum) and understand when, how, and some insight into why they are
effective.
"""
from typing import Any
from typing import Dict

from init2winit.optimizer_lib.kitchen_sink._src import utils
from init2winit.optimizer_lib.kitchen_sink._src.combine import join
from init2winit.optimizer_lib.kitchen_sink._src.mask import mask_registry
from init2winit.optimizer_lib.kitchen_sink._src.transform import transformation_registry
import optax
# TODO(dsuo): document config syntax.


def _get_mask(x):
  """Find a mask in a given element."""
  if 'mask' in x:
    mask = x['mask']
  elif 'mask' in x.get('hps', {}):
    mask = x['hps']['mask']
  else:
    mask = None

  if mask in mask_registry:
    mask = mask_registry[mask]

  return mask


def _kitchen_sink_helper(config):
  """Recursively chain and join `optax.GradientTransformation`s."""

  if utils.is_leaf(config):
    if 'by' in config:
      raise KeyError(f'Leaf {config} should not have key `by`.')

    el = config['element']
    if el not in transformation_registry:
      raise ValueError(f'Transformation {el} not found.')
    hps = config.get('hps', {})
    tx = transformation_registry[el](**hps)

  else:
    if 'hps' in config:
      raise KeyError(f'Config {config} should not have key `hps`.')

    to_join = config.get('join', {})
    for key, val in to_join.items():
      to_join[key] = _kitchen_sink_helper(val)

    # Defaults to `None`, which chains child components together.
    by = config.get('by')
    by_kwargs = config.get('by_kwargs', {})

    tx = join(by, **by_kwargs)(**to_join)

  mask = _get_mask(config)
  if mask is not None:
    tx = optax.masked(tx, mask)

  return tx


def kitchen_sink(config: Dict[str, Any],
                 learning_rate: float = None) -> optax.GradientTransformation:
  """Runs a list of GradientTransforms in parallel and combines.

  Args:
    config: dictionary configuring an optimizer.
    learning_rate: learning rate that gets injected.

  Returns:
    optax.GradientTransform
  """
  # Cast to dict in case we have an ml_collections.ConfigDict.
  config = dict(config)

  # Syntactic sugar. If we have an implied chain, make it explicitly a chain.
  if all([str(i) in config for i in range(len(config))]):
    config = {'join': config}

  # Handle `one_minus_` hps, if any.
  config = utils.map_element(utils.handle_one_minus, config)

  # Apply learning rate to any existing scale_by_learning_rate
  if learning_rate is not None:
    config = utils.apply_and_maybe_scale_by_learning_rate(config, learning_rate)

  return utils.unfreeze_wrapper(*_kitchen_sink_helper(config))
