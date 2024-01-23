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

"""Optimizer utilities."""

import copy
import operator

from absl import logging
import flax
import jax
import jax.numpy as jnp
import optax


def total_tree_sum(pytree):
  """Compute the overall sum of a pytree."""
  sums = jax.tree_map(jnp.sum, pytree)
  return jax.tree_util.tree_reduce(operator.add, sums, 0)


def tree_norm_sql2(pytree):
  """Compute the param-wise squared L2 norm of a pytree."""
  return jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1)) ** 2, pytree)


def total_tree_norm_sql2(pytree):
  """Compute the overall squared L2 norm of a pytree."""
  sql2_norms = tree_norm_sql2(pytree)
  return jax.tree_util.tree_reduce(operator.add, sql2_norms, 0)


def is_leaf(x):
  return isinstance(x, dict) and 'element' in x


def map_element(fn, config, true_leaf_fn=None):
  if not isinstance(config, dict):
    if true_leaf_fn is not None:
      return true_leaf_fn(config)
    else:
      return config
  elif 'element' in config:
    return fn(config)
  else:
    return {k: map_element(fn, v, true_leaf_fn) for k, v in config.items()}


def unfreeze_wrapper(init_fn, update_fn):
  """Freeze/unfreeze params."""

  # NOTE(dsuo): We use plain dicts internally due to this issue
  # https://github.com/deepmind/optax/issues/160.
  def wrapped_init_fn(params):
    return init_fn(flax.core.unfreeze(params))

  def wrapped_update_fn(updates, state, params=None):
    new_updates, state = update_fn(
        flax.core.unfreeze(updates), state,
        None if params is None else flax.core.unfreeze(params))

    if isinstance(updates, flax.core.FrozenDict):
      new_updates = flax.core.freeze(new_updates)

    return new_updates, state

  return optax.GradientTransformation(wrapped_init_fn, wrapped_update_fn)


def handle_one_minus(x):
  if 'hps' in x:
    for hp in copy.deepcopy(x['hps']).keys():
      if 'one_minus_' in hp:
        x['hps'][hp.replace('one_minus_', '')] = 1 - x['hps'][hp]
        del x['hps'][hp]
  return x


def apply_and_maybe_scale_by_learning_rate(config, learning_rate):
  """Apply learning rate and possibly scale by learning rate."""

  def is_scale_by_lr(x):
    return not isinstance(x, str) and x['element'] == 'scale_by_learning_rate'

  def contains_lr_as_param(x):
    return not isinstance(x, str) and x.get(
        'hps', None) and 'learning_rate' in x['hps']

  def update_leaf(x):
    if contains_lr_as_param(x):
      x['hps']['learning_rate'] = learning_rate
      return x
    return x

  scaled = map_element(is_scale_by_lr, config, true_leaf_fn=lambda x: False)
  num_scaled = jax.tree_util.tree_reduce(lambda x, y: x + y, scaled, 0)

  if num_scaled == 0:
    return {
        'join': {
            '0': config,
            '1': {
                'element': 'scale_by_learning_rate',
                'hps': {
                    'learning_rate': learning_rate
                }
            }
        }
    }
  elif num_scaled == 1:
    return map_element(update_leaf, config)
  else:
    logging.warning('Kitchen Sink configuration has more than one '
                    'scale_by_learning_rate. Please double check config')
