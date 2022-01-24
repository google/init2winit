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

from typing import Any, Dict, List

from init2winit.optimizer_lib.transform import bias_correction
from init2winit.optimizer_lib.transform import nesterov
from init2winit.optimizer_lib.transform import polyak_ema
from init2winit.optimizer_lib.transform import polyak_hb
from init2winit.optimizer_lib.transform import precondition_by_adam
from init2winit.optimizer_lib.transform import scale_by_amsgrad
from init2winit.optimizer_lib.transform import scale_by_learning_rate
from init2winit.optimizer_lib.utils import create_weight_decay_mask
import optax

# pylint:disable=invalid-name

# TODO(dsuo): name transformations more clearly
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
                 learning_rate=None) -> optax.GradientTransformation:
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

  if learning_rate:
    transforms = transforms + [scale_by_learning_rate(learning_rate)]

  return optax.chain(*transforms)


def from_config(config, learning_rate):
  """Create optimizer from config."""
  els = [config.momentum]
  hps = [{'decay': 1 - config.one_minus_decay}]
  weight_decay_mask = create_weight_decay_mask()

  if config.decoupled_weight_decay:
    els += ['WeightDecay']

    hps += [{'weight_decay': config.weight_decay, 'mask': weight_decay_mask}]
  else:
    els = ['WeightDecay'] + els
    hps = [{
        'weight_decay': config.weight_decay,
        'mask': weight_decay_mask
    }] + hps

  return kitchen_sink(elements=els, hps=hps, learning_rate=learning_rate)
