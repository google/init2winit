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

import copy
import functools
from typing import Any, Callable, Dict, List, Union

import flax
from init2winit.optimizer_lib.transform import clip_updates
from init2winit.optimizer_lib.transform import first_moment_ema
from init2winit.optimizer_lib.transform import nesterov
from init2winit.optimizer_lib.transform import polyak_averaging
from init2winit.optimizer_lib.transform import polyak_hb
from init2winit.optimizer_lib.transform import precondition_by_amsgrad
from init2winit.optimizer_lib.transform import precondition_by_layered_adaptive_rms
from init2winit.optimizer_lib.transform import precondition_by_rms
from init2winit.optimizer_lib.transform import precondition_by_yogi
from init2winit.optimizer_lib.transform import sanitize_values
from init2winit.optimizer_lib.transform import scale_by_adam
from init2winit.optimizer_lib.transform import scale_by_amsgrad
from init2winit.optimizer_lib.transform import scale_by_learning_rate
from init2winit.optimizer_lib.transform import scale_by_nadam
from init2winit.optimizer_lib.utils import create_weight_decay_mask
from init2winit.optimizer_lib.utils import static_inject_hyperparams
import jax
import jax.numpy as jnp
import optax

# scale_by_rms exists only for backward compatability
_composites = {
    'scale_by_adam': scale_by_adam,
    'scale_by_yogi': optax.scale_by_yogi,
    'scale_by_amsgrad': scale_by_amsgrad,
    'scale_by_nadam': scale_by_nadam,
    'scale_by_rms': precondition_by_rms,
    'sgd': optax.sgd,
}

_first_moment_accumulators = {
    'nesterov': nesterov,
    'polyak_hb': polyak_hb,
    'first_moment_ema': first_moment_ema,
}

_preconditioners = {
    'precondition_by_rms': precondition_by_rms,
    'precondition_by_yogi': precondition_by_yogi,
    'precondition_by_rss': optax.scale_by_rss,
    'precondition_by_amsgrad': precondition_by_amsgrad,
    'precondition_by_layered_adaptive_rms': precondition_by_layered_adaptive_rms
}

_miscellaneous = {
    'add_decayed_weights': optax.add_decayed_weights,
    'polyak_averaging': polyak_averaging,
    'clip_updates': clip_updates,
    'sanitize_values': sanitize_values
}


_transformations = {}
_transformations.update(_composites)
_transformations.update(_preconditioners)
_transformations.update(_first_moment_accumulators)
_transformations.update(_miscellaneous)


def _sum_combinator(*args):
  return functools.reduce(
      lambda x, y: jax.tree_map(lambda i, j: i + j, x, y), args)


def _grafting_helper(chain, use_global_norm=False):
  norm = jax.tree_map(jnp.linalg.norm, chain)
  if use_global_norm:
    global_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.sqrt(x**2 + y**2),
                                            norm)
    norm = jax.tree_map(lambda x: global_norm, norm)
  return norm


def _grafting_combinator(mag_chain,
                         dir_chain,
                         eps: float = 1e-6,
                         use_global_norm: bool = False):
  """Grafting combinator.

  Args:
    mag_chain: transform_chain to determine magnitude of update.
    dir_chain: transform_chain to determine direction of update.
    eps (float, optional): term added to D normalization
        denominator for numerical stability (default: 1e-16)
    use_global_norm (bool, optional): graft global l2 norms rather
        than per-layer (default: False)
  Returns:
    updates in the shape of params.
  """
  mag_norm = _grafting_helper(mag_chain, use_global_norm=use_global_norm)
  dir_norm = _grafting_helper(dir_chain, use_global_norm=use_global_norm)

  norm = jax.tree_map(
      lambda dir, dirn, magn: dir / (dirn + eps) * magn,
      dir_chain, dir_norm, mag_norm
  )

  return norm


_combinators = {
    'sum': _sum_combinator,
    'grafting': _grafting_combinator,
}


def kitchen_sink(chains: List[optax.GradientTransformation],
                 scales: jnp.array = None,
                 combinator: Union[Callable[[Any, Any], Any], str] = 'sum',
                 combinator_args: Dict[str, float] = None,
                 learning_rate: float = None) -> optax.GradientTransformation:
  """Runs a list of GradientTransforms in parallel and combines.

  Args:
    chains: list of optax.GradientTransforms (typically from transform_chain).
    scales: a (len(chains),)-shaped jnp.array.
    combinator: a combinator that reduces a list of identical pytrees
    combinator_args: a dictionary of keyword arguments to the combinator func.
    learning_rate: learning rate that gets injected.

  Returns:
    optax.GradientTransform
  """
  if isinstance(combinator, str):
    combinator = _combinators.get(combinator, _sum_combinator)
  combinator_args = combinator_args or {}

  if scales is None:
    scales = jnp.ones(len(chains))

  chains = [
      optax.chain(chain, optax.scale(scale))
      for chain, scale in zip(chains, scales)
  ]

  def init_fn(params):
    return tuple([chain.init(params) for chain in chains])

  def update_fn(updates, state, params=None):
    result = [chain.update(updates, chain_state, params)
              for chain, chain_state in zip(chains, state)]
    new_updates, new_state = zip(*result)
    return combinator(*new_updates, **combinator_args), new_state

  transform = optax.GradientTransformation(init_fn, update_fn)

  if learning_rate is not None:
    transform = optax.chain(transform, scale_by_learning_rate(learning_rate))

  return transform


def transform_chain(
    elements: List[str],
    hps: List[Dict[str, float]] = None,
    masks: List[Any] = None,
    learning_rate: float = None) -> optax.GradientTransformation:
  """Utility function for chaining GradientTransforms based on string names.

  Args:
    elements: list of transform strings.
    hps: list of dicts of args for each transform.
    masks: list of masks for each transform.
    learning_rate: learning rate that gets injected.

  Returns:
    optax.GradientTransform
  """

  hps = hps or [{}] * len(elements)
  masks = masks or [None] * len(elements)
  transforms = []

  if len(hps) != len(elements):
    raise ValueError('Number of hps must equal number of elements.')

  if len(masks) != len(elements):
    raise ValueError('Number of masks must equal number of elements.')

  transforms = [_transformations[el](**hp) for el, hp in zip(elements, hps)]

  for i, (transform, mask) in enumerate(zip(transforms, masks)):
    if mask is not None:
      transforms[i] = optax.masked(transform, mask)

  if learning_rate is not None:
    transforms += [scale_by_learning_rate(learning_rate)]

  init_fn, update_fn = optax.chain(*transforms)

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


def from_hparams(opt_hparams):
  """Create kitchen_sink optimizer from init2winit config."""

  # If we have '0' in the top level, then this is a single transform_chain.
  if '0' in opt_hparams:
    chains = [transform_chain_from_hparams(opt_hparams)]

  # Otherwise, assume opt_hparams['chains'] is organized as a list of dicts,
  # one index for each chain.
  else:
    chains = [
        transform_chain_from_hparams(chain) for chain in opt_hparams['chains']
    ]

  combinator = opt_hparams.get('combinator', 'sum')

  # NOTE: we inject learning_rate = -1.0 to be a no-op. `scale_by_learning_rate`
  # has flip_sign=True by default, so optax.scale(m * learning_rate) where
  # m = -1 requires learning_rate = -1.0 top no-op.
  return static_inject_hyperparams(kitchen_sink)(
      learning_rate=-1.0, chains=chains, combinator=combinator)


def transform_chain_from_hparams(opt_hparams):
  """Create transform_chain optimizer from init2winit config."""
  elements = []
  hps = []
  masks = []
  index = 0

  while str(index) in opt_hparams:
    hparams = opt_hparams[str(index)]
    element = hparams['element']
    hp = hparams.get('hps', {})
    mask = hparams.get('mask')

    # TODO(dsuo): there is some unfortunate badness here where `mask` for
    # `add_decayed_weights` can show up in either the `hps` or in `mask` keys.
    # If we have less general requirements for masking than we thought, then
    # we should consider amending the `kitchen_sink` mask API.
    if element == 'add_decayed_weights' and (hp.get('mask') == 'bias_bn' or
                                             mask == 'bias_bn'):
      mask = create_weight_decay_mask()
      if 'mask' in hp:
        del hp['mask']

    for h in copy.deepcopy(hp).keys():
      if 'one_minus_' in h:
        hp[h.replace('one_minus_', '')] = 1 - hp[h]
        del hp[h]

    elements.append(element)
    hps.append(hp)
    masks.append(mask)
    index += 1

  return transform_chain(elements, hps, masks)
