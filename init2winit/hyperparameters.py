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

"""Hyperparameter management logic."""
import json
from typing import Dict

from absl import logging
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from ml_collections.config_dict import config_dict
from tensorflow.io import gfile


def expand_key(hparams, key_pieces, index, value):
  """Util to safely expand dotted keys in a dictionary.

  Args:
    hparams: the hparams dictionary containing dotted keys. 
    key_pieces: 
      list containing pieces of dotted key. e.g: ['a', 'b', 'c'] for 'a.b.c'
    index:
      current index being read within key_pieces.
    value:
      value to be inserted for dotted key.

  Raises:
    ValueError:
    1) if any prefix of dotted key is not a dictionary
    2) if dotted key overrides a constant value already set in dictionary. 
  """
  curr_p = key_pieces[index]

  if index == len(key_pieces) - 1:
    if curr_p not in hparams:
      hparams[curr_p] = value
    else:
      if isinstance(hparams[curr_p], Dict) and isinstance(value, Dict):
        hparams[curr_p].update(value)
      elif isinstance(hparams[curr_p], Dict):
        hparams[curr_p] = value
      else:
        raise ValueError(
            'prefix = {} already exists with value = {}'.format(
                '.'.join(key_pieces[: index + 1]), hparams[curr_p]
                )
            )
  else:
    if curr_p not in hparams:
      hparams[curr_p] = {}

    if isinstance(hparams[curr_p], Dict):
      expand_key(hparams[curr_p], key_pieces, index + 1, value)
    else:
      raise ValueError(
          'Aborting dotted key expansion as prefix of dotted key is not a dict:'
          ' prefix = {}, prefix_value = {}'.format(
              '.'.join(key_pieces[: index + 1]), hparams[curr_p]
          )
      )


def expand_dot_keys(d):
  """Expand keys with '.', {'a.b': 1} -> {'a': {'b': 1}}.

  Note that we assert there are no keys with dots that would override each
  other, such as 'a.b' and 'a.b.c'.

  Args:
    d: input dict.

  Returns:
    A dict with the keys expanded on dots.
  """
  expanded_dict = dict(d)
  items = list(expanded_dict.items())

  for key, value in items:
    if '.' in key:
      expand_key(expanded_dict, key.split('.'), 0, value)
      expanded_dict.pop(key)

  return expanded_dict


def build_hparams(model_name,
                  initializer_name,
                  dataset_name,
                  hparam_file,
                  hparam_overrides,
                  input_pipeline_hps=None,
                  allowed_unrecognized_hparams=None,
                  algoperf_submission_name=None):
  """Build experiment hyperparameters.

  Args:
    model_name: the string model name.
    initializer_name: the string initializer name.
    dataset_name: the string dataset name.
    hparam_file: the string to the hyperparameter override file (possibly on
      CNS).
    hparam_overrides: a dict of hyperparameter override names/values, or a JSON
      string encoding of this hyperparameter override dict. Note that this is
      applied after the hyperparameter file overrides.
    input_pipeline_hps: a dict of hyperparameters for performance tuning.
    allowed_unrecognized_hparams: An optional list of hparam keys that hparam
      overrides are allowed to introduce. There is no guaranteed these new
      hparam keys will be ignored. Downgrading an explicit list of unrecognized
      hparams from an error to a warning can be useful when trying to tune using
      a shared search space over multiple workloads that don't all support the
      same set of hyperparameters.
    algoperf_submission_name: The name of the algoperf submission.

  Returns:
    A ConfigDict of experiment hyperparameters.
  """
  model_hps = models.get_model_hparams(model_name)
  initializer_hps = initializers.get_initializer_hparams(initializer_name)
  dataset_hps = datasets.get_dataset_hparams(dataset_name)
  input_pipeline_hps = input_pipeline_hps or config_dict.ConfigDict()

  merged_dict = {}

  hps_dicts = [
      hps.to_dict()
      for hps in [model_hps, initializer_hps, dataset_hps, input_pipeline_hps]
  ]

  total_hps = 0
  for hps_dict in hps_dicts:
    merged_dict.update(hps_dict)
    total_hps += len(hps_dict.keys())

  # Check that all provided have no overlap.
  if total_hps != len(merged_dict.keys()):
    raise ValueError('There is overlap in the provided hparams.')

  # Convert to the Shallue and Lee label smoothing style.
  if merged_dict.get('use_shallue_label_smoothing', False):
    num_classes = merged_dict['output_shape'][-1]
    merged_dict['label_smoothing'] *= num_classes / float(num_classes - 1)

  merged = config_dict.ConfigDict(merged_dict)
  merged.lock()

  # Subconfig "opt_hparams" and "lr_hparams" are allowed to add new fields.
  for key in ['opt_hparams', 'lr_hparams']:
    if key not in merged:
      with merged.unlocked():
        merged[key] = config_dict.ConfigDict()

  for key in ['opt_hparams', 'lr_hparams']:
    merged[key].unlock()

  if algoperf_submission_name:
    with merged.unlocked():
      merged['algoperf_submission_name'] = algoperf_submission_name

  if hparam_file:
    logging.info('Loading hparams from %s', hparam_file)
    with gfile.GFile(hparam_file, 'r') as f:
      hparam_dict = json.load(f)
      merged.update_from_flattened_dict(hparam_dict)

  if hparam_overrides:
    if isinstance(hparam_overrides, str):
      hparam_overrides = json.loads(hparam_overrides)

    # If the user is changing the learning rate schedule or optimizer. We must
    # wipe all of the keys from the old dictionary.
    merged_schedule = None
    if merged.get('lr_hparams'):
      merged_schedule = merged['lr_hparams'].get('schedule')
    overrides_schedule = None
    if hparam_overrides.get('lr_hparams'):
      overrides_schedule = hparam_overrides['lr_hparams'].get('schedule')
    if overrides_schedule and merged_schedule != overrides_schedule:
      merged['lr_hparams'] = {}
    if ('optimizer' in hparam_overrides and
        merged['optimizer'] != hparam_overrides['optimizer']):
      merged['opt_hparams'] = {}
    hparam_overrides = expand_dot_keys(hparam_overrides)
    if allowed_unrecognized_hparams:
      new_keys = [k for k in hparam_overrides if k not in merged]
      if new_keys:
        logging.warning('Unrecognized top-level hparams: %s', new_keys)
      if any(k not in allowed_unrecognized_hparams for k in new_keys):
        raise ValueError(
            f'Unrecognized top-level hparams not in allowlist: {new_keys}')
      with merged.unlocked():
        merged.update(hparam_overrides)
    else:
      merged.update(hparam_overrides)

  return merged
