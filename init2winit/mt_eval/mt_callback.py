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

"""Callback for machine translation, typically run inside the training loop.

This setups up custom code/evals to run after every few training steps.

Usually, we'll need to setup callbacks in a multilingual
machine translation setting. A common use case is training on mixture of
language pairs,
and evaluating on entirely different task/dataset/language pair.
A typical callback_config would be a `list` of following-such dicts.
{
  ## beginning of required keys
  'tfds_dataset_key': 'wmt15_translate/de-en',
  'reverse_translation': True,
  'dataset_name': 'translate_wmt',  # to get dataset builder.
  'eval_batch_size': 512,
  'eval_train_num_batches': 40  # to avoid evaluating on huge train set.
  'eval_num_batches': None  # None evaluates full 'validation' split.
  'eval_splits': ['train', 'test', 'valid']
  ## end of required keys

}
"""
from absl import logging
from init2winit import base_callback
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import datasets
import jax
from ml_collections.config_dict import config_dict
import numpy as np

_REQUIRED_KEYS = [
    'dataset_name', 'tfds_dataset_key', 'tfds_eval_dataset_key',
    'tfds_predict_dataset_key', 'reverse_translation', 'eval_batch_size',
    'eval_train_num_batches', 'eval_num_batches', 'eval_splits']
_SPLITS = ['train', 'valid', 'test']


class MTEvaluationCallback(base_callback.BaseCallBack):
  """Runs evals on MT models with datasets/params different than in training."""

  def _validate_callback_config(self):
    assert all(key in self.callback_config for key in _REQUIRED_KEYS), (
        'callback config must contain these required keys:', _REQUIRED_KEYS)
    assert ('vocab_path' not in self.callback_config), (
        'Eval must use same vocab file as used in training. No need to specify'
        ' vocab file. One from training config will be used.'
    )
    assert all(
        split_name in set(_SPLITS)
        for split_name in self.callback_config['eval_splits']
    ), ('callback_config.eval_splits must contain only subset of these splits:',
        _SPLITS)

  def _get_dataset(self, hps, rng):
    """Sets ups dataset builders."""
    hparams_dict = hps.to_dict()
    hparams_dict.update(self.callback_config)
    hparams = config_dict.ConfigDict(hparams_dict)
    dataset_builder = datasets.get_dataset(self.callback_config['dataset_name'])
    dataset = dataset_builder(
        rng,
        hparams.batch_size,
        eval_batch_size=self.callback_config['eval_batch_size'],
        hps=hparams)
    return dataset

  def _evaluate(self,
                flax_module,
                batch_stats,
                batch_iter,
                evaluate_batch_pmapped):
    """Compute aggregated metrics on the given data iterator.

    This function is taken as is from trainer.py to avoid circular dependency.
    TODO(ankugarg@): Refactor this function somewhere into eval_commons.py

    Args:
      flax_module: A flax module.
      batch_stats: A dict of batch_stats.
      batch_iter: Generator which yields batches. Must support the API
        for b in batch_iter:
      evaluate_batch_pmapped: A function with API
       evaluate_batch_pmapped(params, batch_stats, batch). Returns a dictionary
       mapping keys to the metric values across the sharded batch.

    Returns:
      A dictionary of aggregated metrics. The keys will match the keys returned
      by evaluate_batch_pmapped.
    """
    metrics = None
    for batch in batch_iter:
      batch = data_utils.shard(batch)
      computed_metrics = evaluate_batch_pmapped(
          params=flax_module, batch_stats=batch_stats, batch=batch)
      if metrics is None:
        metrics = computed_metrics
      else:
        metrics = metrics.merge(computed_metrics)

    # For data splits with no data (e.g. Imagenet no test set) no values
    # will appear for that split.
    if metrics is not None:
      metrics = metrics.unreplicate().compute()
      for key, val in metrics.items():
        if np.isnan(val):
          raise utils.TrainingDivergedError('NaN detected in {}'.format(key))
    return metrics

  def _merge_and_apply_prefix(self, d1, d2, prefix):
    """Merges metrics from one dict into another global metrics dict.

    Args:
      d1: global dict to merge metrics into.
      d2: dict of computed metrics.
      prefix: optionally apply string prefix before merging into d1.

    Returns:
      A dictionary of merged metrics.
    """
    d1 = d1.copy()
    for key in d2:
      d1[prefix+key] = d2[key]
    return d1

  def __init__(self, model, flax_module, batch_stats, optimizer_state,
               dataset, hps, callback_config, train_dir, rng):
    del flax_module
    del batch_stats
    del optimizer_state
    del train_dir
    del dataset
    self.callback_config = callback_config
    self._validate_callback_config()
    self.evaluate_batch_pmapped = jax.pmap(
        model.evaluate_batch, axis_name='batch', donate_argnums=(2,))
    self.dataset = self._get_dataset(hps, rng)

  def run_eval(self, flax_module, batch_stats, optimizer_state, global_step):
    """Runs the MT models to evals specified by MT model.

    Args:
      flax_module: Replicated flax module.
      batch_stats: Replicated batch_stats from the trainer.
      optimizer_state: Replicated optimizer state from the trainer.
      global_step: Current training step.

    Returns:
      Dictionary of metrics.
      Example:
        {'callback/wmt15_translate/de-en/valid/ce_loss': 0.11,
         'callback/wmt16_translate/ro-en/test/ce_loss': 0.13
        }
    """
    del optimizer_state

    ds_splits_dict = {}
    for eval_split in self.callback_config['eval_splits']:
      if 'train' in eval_split:
        ds_splits_dict[eval_split] = self.dataset.eval_train_epoch(
            self.callback_config['eval_train_num_batches'])
      elif 'valid' in eval_split:
        ds_splits_dict[eval_split] = self.dataset.valid_epoch(
            self.callback_config['eval_num_batches'])
      else:
        ds_splits_dict[eval_split] = self.dataset.test_epoch(
            self.callback_config['eval_num_batches'])

    metrics = {}
    for split_name, split_iter in ds_splits_dict.items():
      try:
        split_metrics = self._evaluate(flax_module,
                                       batch_stats,
                                       split_iter,
                                       self.evaluate_batch_pmapped)
        metrics = self._merge_and_apply_prefix(
            metrics, split_metrics, 'callback/' +
            self.callback_config['tfds_dataset_key'] + '/' + split_name + '/')
      except utils.TrainingDivergedError as err:
        # we don't want to stop training.
        del err
        logging.info('Callback evaluation diverged for dataset %s at step:%d',
                     self.tfds_dataset_key, global_step)
        continue

    return metrics
