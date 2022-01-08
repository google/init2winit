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

# Lint as: python3
"""Hyperparameter management logic."""

import json

from absl import logging
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from ml_collections.config_dict import config_dict
from tensorflow.io import gfile


def build_hparams(model_name,
                  initializer_name,
                  dataset_name,
                  hparam_file,
                  hparam_overrides):
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

  Returns:
    A ConfigDict of experiment hyperparameters.
  """
  model_hps = models.get_model_hparams(model_name)
  initializer_hps = initializers.get_initializer_hparams(initializer_name)
  dataset_hps = datasets.get_dataset_hparams(dataset_name)

  merged_dict = {}

  hps_dicts = [
      hps.to_dict() for hps in [model_hps, initializer_hps, dataset_hps]
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
    if 'lr_hparams.schedule' in hparam_overrides and merged[
        'lr_hparams']['schedule'] != hparam_overrides[
            'lr_hparams.schedule']:
      merged['lr_hparams'] = {}
    if 'optimizer' in hparam_overrides and merged[
        'optimizer'] != hparam_overrides['optimizer']:
      merged['opt_hparams'] = {}
    merged.update_from_flattened_dict(hparam_overrides)

  return merged
