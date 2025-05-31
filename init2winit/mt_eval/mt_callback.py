# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

If your decoder uses remat_scan, e.g. an xformer_translate with
'dec_remat_scan_lengths' equal to some tuple, then you need to specify
'scan_over_layers_offset' equal to the length of that tuple.
"""

import functools

from absl import logging
from init2winit import base_callback
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import datasets
from init2winit.model_lib import models
from init2winit.mt_eval import inference
import jax
from ml_collections.config_dict import config_dict
import numpy as np


_REQUIRED_KEYS = [
    'dataset_name', 'model_name', 'tfds_dataset_key', 'tfds_eval_dataset_key',
    'tfds_predict_dataset_key', 'reverse_translation', 'eval_batch_size',
    'eval_train_num_batches', 'eval_num_batches', 'eval_splits',
    'max_decode_length', 'tl_code', 'beam_size', 'decoding_type']
_SPLITS = ['train', 'valid', 'test']


class MTEvaluationCallback(base_callback.BaseCallBack):
  """Runs evals on MT models with datasets/params different than in training."""

  def __init__(self,
               model,
               params,
               batch_stats,
               optimizer_state,
               optimizer_update_fn,
               dataset,
               hps,
               callback_config,
               train_dir,
               rng,
               mesh):
    del optimizer_state
    del optimizer_update_fn
    del train_dir
    del dataset
    del params

    merged_callback_config = inference.DEFAULT_EVAL_CONFIG.copy()
    merged_callback_config.update(callback_config)

    self.callback_config = merged_callback_config

    self._validate_callback_config()
    self.evaluate_batch_jitted = jax.jit(
        model.evaluate_batch, donate_argnums=(2,)
    )
    self.batch_stats = batch_stats

    dataset, dataset_metadata = self._get_dataset(hps, rng)
    self.dataset = dataset
    self.mesh = mesh
    model_class = models.get_model(callback_config['model_name'])

    self.inference_manager = inference.InferenceManager(
        hps,
        rng,
        model_class,
        dataset,
        dataset_metadata,
        self.callback_config,
        mode='online',
        mesh=mesh)

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
    dataset_metadata = datasets.get_dataset_meta_data(
        self.callback_config['dataset_name'])
    dataset = dataset_builder(
        rng,
        hparams.batch_size,
        eval_batch_size=self.callback_config['eval_batch_size'],
        hps=hparams)
    return dataset, dataset_metadata

  def _evaluate(self,
                params,
                batch_stats,
                batch_iter,
                evaluate_batch_jitted):
    """Compute aggregated metrics on the given data iterator.

    This function is taken as is from trainer.py to avoid circular dependency.
    TODO(ankugarg@): Refactor this function somewhere into eval_commons.py

    Args:
      params: model params.
      batch_stats: A dict of batch_stats.
      batch_iter: Generator which yields batches. Must support the API
        for b in batch_iter:
      evaluate_batch_jitted: A function with API evaluate_batch_jitted(params,
        batch_stats, batch). Returns a dictionary mapping keys to the metric
        values across the sharded batch.

    Returns:
      A dictionary of aggregated metrics. The keys will match the keys returned
      by evaluate_batch_jitted.
    """
    metrics = None
    make_global_array_fn = functools.partial(
        data_utils.make_global_array, mesh=self.mesh
    )

    for batch in batch_iter:
      batch = jax.tree_util.tree_map(make_global_array_fn, batch)
      computed_metrics = evaluate_batch_jitted(
          params=params, batch_stats=batch_stats, batch=batch
      )
      if metrics is None:
        metrics = computed_metrics
      else:
        metrics = metrics.merge(computed_metrics)

    # For data splits with no data (e.g. Imagenet no test set) no values
    # will appear for that split.
    if metrics is not None:
      metrics = metrics.compute()
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

  def run_eval(
      self, params, batch_stats, optimizer_state, global_step):
    """Runs the MT models to evals specified by MT model.

    Args:
      params: Replicated model params.
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

    # Eval metrics evaluation.
    for split_name, split_iter in ds_splits_dict.items():
      try:
        decoding_output = (
            self.inference_manager.translate_and_calculate_bleu_single_model(
                params, split_name))
        split_metrics = self._evaluate(params, batch_stats, split_iter,
                                       self.evaluate_batch_jitted)
        split_metrics['bleu_score'] = decoding_output.bleu_score

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
