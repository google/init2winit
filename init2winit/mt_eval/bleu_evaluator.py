# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

r"""BLEU evaluator container class."""
import copy
import functools
import os

from absl import logging
from flax import jax_utils
from flax.training import common_utils
from init2winit.dataset_lib import mt_tokenizer
from init2winit.mt_eval import decode
from init2winit.mt_eval import eval_utils
from init2winit.optimizer_lib import optimizers
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

DEFAULT_EVAL_CONFIG = {
    'eval_batch_size': 16,
    'eval_split': 'test',
    'max_decode_length': 256,
    'beam_size': 4,
    'eval_num_batches': None,
    'ckpt_to_evaluate': None,
    'min_step': None,
    'ckpt_avg_window': 0,
}


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


class BLEUEvaluator(object):
  """Evaluates BLEU."""

  def __init__(self, checkpoint_dir, hps, rng, model_cls, dataset_builder,
               dataset_meta_data, mt_eval_config):
    self.checkpoint_dir = checkpoint_dir
    self.hps = hps
    self.eos_id = decode.EOS_ID
    self.max_length = mt_eval_config['max_decode_length']
    self.beam_size = mt_eval_config['beam_size']
    self.eval_batch_size = mt_eval_config['eval_batch_size']
    self.eval_num_batches = mt_eval_config['eval_num_batches']
    self.eval_split = mt_eval_config['eval_split']
    self.ckpt_to_evaluate = mt_eval_config['ckpt_to_evaluate']
    self.ckpt_avg_window = mt_eval_config['ckpt_avg_window']
    params_rng, dropout_rng, data_rng = jax.random.split(rng, num=3)
    self.get_dataset(data_rng, dataset_builder)
    self.encoder = self.load_tokenizer(hps.vocab_path)
    self.initialize_model(model_cls, dataset_meta_data, dropout_rng, params_rng)
    self.min_step = mt_eval_config['min_step']

  def iterate_checkpoints(self):
    """Iterates over all checkpoints."""
    if self.ckpt_to_evaluate:
      step = int(self.ckpt_to_evaluate.split('_')[-1])
      full_path = os.path.join(self.checkpoint_dir, self.ckpt_to_evaluate)
      yield full_path, step
    else:
      for checkpoint_path in gfile.glob(
          os.path.join(self.checkpoint_dir, 'ckpt_*')):
        step = int(checkpoint_path.split('_')[-1])
        if self.min_step and step < int(self.min_step):
          continue
        full_path = os.path.join(self.checkpoint_dir, checkpoint_path)
        yield full_path, step

  def get_dataset(self, data_rng, dataset_builder):
    """Get dataset."""
    eval_batch_size = self.eval_batch_size
    if not eval_batch_size:
      eval_batch_size = self.hps.batch_size
      if self.hps.eval_batch_size:
        eval_batch_size = self.hps.eval_batch_size
    self.dataset = dataset_builder(
        data_rng,
        self.hps.batch_size,
        eval_batch_size=eval_batch_size,
        hps=self.hps)
    self.eval_batch_size = eval_batch_size
    logging.info('Using evaluation batch size: %s', self.eval_batch_size)

  def get_ds_iter(self):
    """Dataset iterator."""
    if self.eval_split == 'train':
      logging.info('Loading train split')
      return self.dataset.eval_train_epoch(self.eval_num_batches)
    elif self.eval_split == 'valid':
      logging.info('Loading Validation split')
      return self.dataset.valid_epoch(self.eval_num_batches)
    else:
      logging.info('Loading test split')
      return self.dataset.test_epoch(self.eval_num_batches)

  def load_tokenizer(self, vocab_path):
    logging.info('Loading tokenizer from: %s', vocab_path)
    # pylint: disable=protected-access
    return mt_tokenizer._load_sentencepiece_tokenizer(vocab_path)
    # pylint: enable=protected-access

  def initialize_model(self, model_cls, dataset_meta_data, dropout_rng,
                       params_rng):
    """Initialie model, especially cache for fast auto-regressive decoding."""
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    hps = copy.deepcopy(self.hps)
    hps = hps.unlock()
    hps.decode = True
    hps = hps.lock()
    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)
    xs = [np.zeros((self.hps.batch_size, *x)) for x in self.hps.input_shape]
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn(
        {'params': params_rng, 'dropout': dropout_rng}, *xs)
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})
    self.flax_module = model.flax_module
    self.params = params
    self.batch_stats = jax_utils.replicate(batch_stats)
    optimizer_init_fn, _ = optimizers.get_optimizer(self.hps)
    optimizer_state = optimizer_init_fn(params)
    self.optimizer_state = jax_utils.replicate(optimizer_state)
    self.pmapped_init_cache = jax.pmap(
        functools.partial(
            self.initialize_cache,
            max_length=self.max_length,
            params_rng=params_rng,
            dropout_rng=dropout_rng))

  def initialize_cache(self, inputs, max_length, params_rng, dropout_rng):
    """Initialize a cache for a given input shape and max decode length."""
    targets_shape = (inputs.shape[0], max_length) + inputs.shape[2:]
    model_init_fn = jax.jit(
        functools.partial(self.flax_module.init, train=False))
    xs = [jnp.ones(inputs.shape), jnp.ones(targets_shape)]
    init_dict = model_init_fn({
        'params': params_rng,
        'dropout': dropout_rng
    }, *xs)
    return init_dict['cache']

  def decode_tokens(self, toks):
    valid_toks = toks[:np.argmax(toks == self.eos_id) + 1].astype(np.int32)
    return self.encoder.detokenize(valid_toks).numpy().decode('utf-8')

  def current_batch_size(self, batch):
    # we assume first token is non-zero in each target side example.
    return int(batch['weights'][:, 0].sum())

  def build_predictor(self):
    decoder = functools.partial(
        decode.decode_step,
        eos_id=self.eos_id,
        beam_size=self.beam_size)
    self.pmapped_predictor = jax.pmap(
        decoder, axis_name='batch', static_broadcasted_argnums=(1, 4))

  def translate_and_calculate_bleu(self):
    """Iterate over all checkpoints and calculate BLEU."""
    self.build_predictor()
    bleu_scores_list = []
    for _, step in self.iterate_checkpoints():
      # If already done, don't redo.
      bleu_score = eval_utils.load_evals(self.checkpoint_dir, step,
                                         self.eval_split)
      if bleu_score:
        bleu_scores_list.append(bleu_score)
        continue
      ckpt_paths = eval_utils.get_checkpoints_in_range(
          checkpoint_dir=self.checkpoint_dir,
          lower_bound=step - self.ckpt_avg_window,
          upper_bound=step)
      logging.info('Current checkpoints: %s', ckpt_paths)
      params, _, _ = eval_utils.average_checkpoints(
          checkpoint_paths=ckpt_paths,
          params=self.params,
          optimizer_state=self.optimizer_state,
          batch_stats=self.batch_stats)
      params_replicated = jax_utils.replicate(params)
      sources, references, predictions = [], [], []
      for batch in self.get_ds_iter():
        pred_batch = common_utils.shard(batch)
        cache = self.pmapped_init_cache(pred_batch['inputs'])
        model_predictions = self.pmapped_predictor(pred_batch, self.flax_module,
                                                   params_replicated, cache,
                                                   self.max_length)
        predicted = tohost(model_predictions)
        inputs = tohost(pred_batch['inputs'])
        targets = tohost(pred_batch['targets'])
        current_batch_size = self.current_batch_size(batch)
        logging.info('Run batch size: %f', batch['inputs'].shape[0])
        logging.info('Actual batch size: %f', current_batch_size)
        for i, s in enumerate(predicted[:current_batch_size]):
          curr_source = self.decode_tokens(inputs[i])
          curr_ref = self.decode_tokens(targets[i])
          curr_pred = self.decode_tokens(s)
          logging.info('Current source: %s', curr_source)
          logging.info('Current Reference: %s', curr_ref)
          logging.info('Current Translation: %s', curr_pred)
          sources.append(curr_source)
          references.append(curr_ref)
          predictions.append(curr_pred)

      bleu_score = eval_utils.compute_bleu_from_predictions(
          predictions, references, 'sacrebleu')['sacrebleu']
      logging.info('Sacre bleu score at step %d: %f', step, bleu_score)
      bleu_scores_list.append(bleu_score)
      eval_utils.save_evals(self.checkpoint_dir, step, self.eval_split,
                            bleu_score)
    return bleu_scores_list
