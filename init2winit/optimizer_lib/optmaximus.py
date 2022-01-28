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

This project seeks to take ideas in optimization (e.g., weight decay,
momentum) and understand when, how, and some insight into why they are
effective.
"""

import copy
from typing import Any, Dict, List

import flax
from init2winit.optimizer_lib.transform import bias_correction
from init2winit.optimizer_lib.transform import nesterov
from init2winit.optimizer_lib.transform import polyak_ema
from init2winit.optimizer_lib.transform import polyak_hb
from init2winit.optimizer_lib.transform import precondition_by_adam
from init2winit.optimizer_lib.transform import scale_by_amsgrad
from init2winit.optimizer_lib.transform import scale_by_learning_rate
from init2winit.optimizer_lib.utils import create_weight_decay_mask
import optax

_transformations = {
    'scale_by_rss': optax.scale_by_rss,
    'scale_by_adam': optax.scale_by_adam,
    'scale_by_amsgrad': scale_by_amsgrad,
    'bias_correction': bias_correction,
    'scale_by_learning_rate': scale_by_learning_rate,
    'nesterov': nesterov,
    'polyak_ema': polyak_ema,
    'polyak_hb': polyak_hb,
    'precondition_by_adam': precondition_by_adam,
    'scale_by_rms': optax.scale_by_rms,
    'sgd': optax.sgd,
    'add_decayed_weights': optax.add_decayed_weights
}


def kitchen_sink(elements: List[str],
                 hps: List[Dict[str, float]] = None,
                 masks: List[Any] = None,
                 learning_rate=-1.) -> optax.GradientTransformation:
  """Utility function for chaining GradientTransforms based on string names."""

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
  """Create kitchen_sink optimizer from init2winit."""
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

  return optax.inject_hyperparams(kitchen_sink)(
      learning_rate=-1.0, elements=elements, hps=hps, masks=masks)
