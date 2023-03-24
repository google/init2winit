# coding=utf-8
# Copyright 2023 The init2winit Authors.
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
from typing import Any, Dict

from absl import logging
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from ml_collections.config_dict import config_dict
from tensorflow.io import gfile


def _get_prefix_violations(trie):
  """Find all internal nodes in the trie that have a count > 0."""
  bad_keys = []
  for key, value in trie.items():
    count, child = value
    # If we have any keys that have children underneath them, error.
    if child and count > 0:
      bad_keys.append((key, child))
    bad_keys.extend(_get_prefix_violations(child))
  return bad_keys


def _add_to_trie(trie, key_pieces):
  curr_p = key_pieces[0]
  if curr_p not in trie:
    trie[curr_p] = (0, {})
  if len(key_pieces) == 1:
    trie[curr_p] = (trie[curr_p][0] + 1, trie[curr_p][1])
  else:
    _add_to_trie(trie[curr_p][1], key_pieces[1:])


def _assert_prefix_free_keys(d: Dict[str, Any]):
  """Check for keys that are dot-subsets of each other ('a.b', 'a.b.c'.)."""
  # We build a trie and assert there are only elements that terminate at a leaf
  # in the tree.
  trie = {}
  for key in d.keys():
    _add_to_trie(trie, key.split('.'))
  offending_keys = _get_prefix_violations(trie)
  if offending_keys:
    error_msgs = []
    for key, child in offending_keys:
      children = ', '.join(child.keys())
      error_msgs.append(
          f'Key {key} has dot children that would be overriden: {children}')
    raise ValueError('\n'.join(error_msgs))


def expand_dot_keys(
    d: Dict[str, Any], assert_prefix_free_keys: bool = True) -> Dict[str, Any]:
  """Expand keys with '.', {'a.b': 1} -> {'a': {'b': 1}}.

  Note that we assert there are no keys with dots that would override each
  other, such as 'a.b' and 'a.b.c'.

  Args:
    d: input dict.
    assert_prefix_free_keys: whether or not to call _assert_prefix_free_keys on
      the input dict d.

  Returns:
    A dict with the keys expanded on dots.
  """
  if assert_prefix_free_keys:
    _assert_prefix_free_keys(d)
  new_dict = {}
  for key, value in d.items():
    if '.' in key:
      new_key, rest = key.split('.', 1)
      new_dict.setdefault(new_key, {})[rest] = value
    else:
      new_dict[key] = value
  # In cases like {'a.b.c': 1}, we will have a dict like {'a': {'b.c': 1}}, so
  # we need to recursively try to expand all sub-dicts.
  for key, value in new_dict.items():
    if isinstance(value, dict):
      new_dict[key] = expand_dot_keys(value, assert_prefix_free_keys=False)
  return new_dict


def build_hparams(model_name,
                  initializer_name,
                  dataset_name,
                  hparam_file,
                  hparam_overrides,
                  input_pipeline_hps=None):
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
    merged.update(hparam_overrides)

  return merged
