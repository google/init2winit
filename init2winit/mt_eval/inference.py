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

r"""BLEU evaluator container class."""

import copy
import dataclasses
import functools
import os
from typing import Any, Sequence

from absl import logging
from init2winit.dataset_lib import data_utils as utils
from init2winit.dataset_lib import mt_tokenizer
from init2winit.mt_eval import decode
from init2winit.mt_eval import eval_utils
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile


DEFAULT_EVAL_CONFIG = {
    'eval_batch_size': 16,
    'eval_splits': ['test'],
    'max_decode_length': 256,
    'eval_num_batches': None,
    'ckpt_to_evaluate': None,
    'min_step': None,
    'ckpt_avg_window': 0,
    'tl_code': None,
    'decoding_type': 'beam_search',  # `beam_search` or `sampling`.
    'beam_size': 4,
    'sample_size': 15,
    'temperature': 1.0,  # 1.0 means no temperature.
    'rescale_log_probs': True,
    'scan_over_layers_offset': 0,  # Models not using scan over layers.
}


@dataclasses.dataclass
class DecodingOutput:
  source_list: Sequence[Any] = dataclasses.field(default_factory=list)
  reference_list: Sequence[Any] = dataclasses.field(default_factory=list)
  translation_list: Sequence[Any] = dataclasses.field(default_factory=list)
  bleu_score: float = 0.0
  decoding_type: str = 'beam_search'


class InferenceManager(object):
  """Evaluates BLEU."""

  def __init__(self, *args, **kwargs):
    if kwargs['mode'] not in ['offline', 'online']:
      raise ValueError('BLEU score computation only support online or '
                       'offline modes.')
    self.mesh = kwargs['mesh']
    if kwargs['mode'] == 'offline':
      self.init_offline_evaluator(*args)
    else:
      self.init_online_evaluator(*args)

  def init_offline_evaluator(self,
                             checkpoint_dir,
                             hps,
                             rng,
                             model_cls,
                             dataset,
                             dataset_meta_data,
                             mt_eval_config):
    """Utility for initializing offline BLEU evaluator."""
    self.checkpoint_dir = checkpoint_dir
    self.hps = hps
    self.eos_id = decode.EOS_ID
    self.max_length = mt_eval_config.get('max_decode_length')
    self.eval_num_batches = mt_eval_config.get('eval_num_batches')
    self.eval_split = mt_eval_config.get('eval_split')
    self.ckpt_to_evaluate = mt_eval_config.get('ckpt_to_evaluate')
    self.ckpt_avg_window = mt_eval_config.get('ckpt_avg_window')
    self.min_step = mt_eval_config.get('min_step')
    self.mt_eval_config = mt_eval_config
    self.dataset = dataset
    params_rng, dropout_rng = jax.random.split(rng, num=2)
    if not hps.vocab_path:
      self.encoder = None
      self.use_test_vocab = True
    else:
      self.encoder = self.load_tokenizer(hps.vocab_path)
      self.use_test_vocab = False
    self.initialize_model(model_cls, dataset_meta_data, dropout_rng, params_rng)

  def init_online_evaluator(self,
                            hps,
                            rng,
                            model_cls,
                            dataset,
                            dataset_metadata,
                            mt_eval_config):
    """Utility for initializing online BLEU evaluator."""
    self.hps = hps
    self.eos_id = decode.EOS_ID
    self.max_length = mt_eval_config.get('max_decode_length')
    self.eval_num_batches = mt_eval_config.get('eval_num_batches')
    self.mt_eval_config = mt_eval_config
    self.dataset = dataset
    if not hps.vocab_path:
      self.encoder = None
      self.use_test_vocab = True
    else:
      self.encoder = self.load_tokenizer(hps.vocab_path)
      self.use_test_vocab = False
    params_rng, dropout_rng = jax.random.split(rng, num=2)
    self.initialize_model(model_cls, dataset_metadata, params_rng, dropout_rng)

  def iterate_checkpoints(self):
    """Iterates over all checkpoints."""
    if self.ckpt_to_evaluate:
      step = int(self.ckpt_to_evaluate.split('_')[-1])
      full_path = os.path.join(self.checkpoint_dir, self.ckpt_to_evaluate)
      yield full_path, step
    else:
      for checkpoint_path in glob(os.path.join(self.checkpoint_dir, 'ckpt_*')):
        step = int(checkpoint_path.split('_')[-1])
        if self.min_step and step < int(self.min_step):
          continue
        full_path = os.path.join(self.checkpoint_dir, checkpoint_path)
        yield full_path, step

  def get_ds_iter(self, eval_split=None):
    """Dataset iterator."""
    if eval_split is None:
      eval_split = self.eval_split

    if eval_split == 'train':
      logging.info('Loading train split')
      return self.dataset.eval_train_epoch(self.eval_num_batches)
    elif eval_split == 'valid':
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
    self.flax_module = model.flax_module
    self.params = params
    self.init_cache = jax.jit(
        functools.partial(
            self.initialize_cache,
            max_length=self.max_length,
            params_rng=params_rng,
            dropout_rng=dropout_rng))

  def initialize_cache(self, inputs, max_length, params_rng, dropout_rng):
    """Initialize a cache for a given input shape and max decode length."""
    logging.info('Initializing cache.')

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
    print('DEBUG: toke shape : ', toks.shape)
    print('DEBUG: toks: ', toks)
    if not self.use_test_vocab:
      valid_toks = toks[:np.argmax(toks == self.eos_id) + 1].astype(np.int32)
      return self.encoder.detokenize(valid_toks).numpy().decode('utf-8')
    else:
      valid_tok = toks[0]
      print('DEBUG: valid_toks: ', valid_tok)
      return TEST_VOCAB[valid_tok]

  def current_batch_size(self, batch):
    # we assume first token is non-zero in each target side example.
    return int(batch['weights'][:, 0].sum())

  def build_predictor(self):
    """Either build beam search decoder or sampling decoder."""
    if self.mt_eval_config.get('decoding_type') == 'sampling':
      decoder = functools.partial(
          decode.sampling_step,
          max_decode_len=self.max_length,
          rng=jax.random.PRNGKey(0),
          flax_module=self.flax_module,
          eos_id=self.eos_id,
          sample_size=self.mt_eval_config.get('sample_size'),
          temperature=self.mt_eval_config.get('temperature'),
          rescale_log_probs=self.mt_eval_config.get('rescale_log_probs'))
    else:
      decoder = functools.partial(
          decode.decode_step,
          max_decode_len=self.max_length,
          flax_module=self.flax_module,
          eos_id=self.eos_id,
          beam_size=self.mt_eval_config.get('beam_size'),
          offset=self.mt_eval_config.get('scan_over_layers_offset', 0))
    self.predictor = jax.jit(decoder)

  def translate_and_calculate_bleu(self):
    """Iterate over all checkpoints and calculate BLEU."""
    # Output is List of (step, bleu_score, (sources, references, predictions))
    # Its a list because we evaluate multiple checkpoints.
    decoding_outputs = []
    for _, step in self.iterate_checkpoints():
      ckpt_paths = eval_utils.get_checkpoints_in_range(
          checkpoint_dir=self.checkpoint_dir,
          lower_bound=step - self.ckpt_avg_window,
          upper_bound=step)
      logging.info('Current checkpoints: %s', ckpt_paths)
      params = eval_utils.average_checkpoints(
          checkpoint_paths=ckpt_paths,
          params=self.params)
      _, params_replicated = utils.shard_pytree(params, self.mesh)
      decoding_output = self.translate_and_calculate_bleu_single_model(
          params_replicated, self.eval_split)
      logging.info('Sacre bleu score at step %d: %f', step,
                   decoding_output.bleu_score)
      decoding_outputs.append(decoding_output)
    return decoding_outputs

  def translate_and_calculate_bleu_single_model(self, params, eval_split):
    """Decode one model on one dataset split."""
    self.build_predictor()
    decode_output = DecodingOutput()
    logging.info('Starting decoding..')

    make_global_array_fn = functools.partial(
        utils.make_global_array, mesh=self.mesh
    )

    for batch in self.get_ds_iter(eval_split):
      pred_batch = jax.tree_util.tree_map(make_global_array_fn, batch)
      cache = self.init_cache(pred_batch['inputs'])
      predicted = self.predictor(pred_batch, params, cache)
      inputs = pred_batch['inputs']
      targets = pred_batch['targets']
      weights = pred_batch['weights']

      predicted = np.array(predicted)
      inputs = np.array(inputs)
      targets = np.array(targets)
      weights = np.array(weights)
      current_batch_size = int(weights[:, 0].sum())
      if self.mt_eval_config.get('decoding_type') == 'beam_search':
        self.process_beam_search_output(inputs, targets, predicted,
                                        current_batch_size, decode_output)
      else:
        self.process_sampling_output(inputs, targets, predicted,
                                     current_batch_size, decode_output)

    logging.info('Predictions: %d References %d Sources %d.',
                 len(decode_output.translation_list),
                 len(decode_output.reference_list),
                 len(decode_output.source_list))
    if self.mt_eval_config.get('decoding_type') == 'beam_search':
      print('DEBUG: ', decode_output.translation_list)
      print('DEBUG: ', decode_output.reference_list)

      bleu_score = eval_utils.compute_bleu_from_predictions(
          decode_output.translation_list,
          decode_output.reference_list,
          self.mt_eval_config.get('tl_code'),
          'sacrebleu')['sacrebleu']
      decode_output.bleu_score = bleu_score
    decode_output.decoding_type = self.mt_eval_config.get('decoding_type')
    return decode_output

  def process_beam_search_output(self,
                                 inputs,
                                 targets,
                                 predicted,
                                 batch_size,
                                 decode_output):
    """Process output if its beam search decoding."""
    for i in range(batch_size):
      curr_source = self.decode_tokens(inputs[i])
      curr_ref = self.decode_tokens(targets[i])
      curr_pred = self.decode_tokens(predicted[i])
      decode_output.source_list.append(curr_source)
      decode_output.reference_list.append(curr_ref)
      decode_output.translation_list.append(curr_pred)

  def process_sampling_output(self,
                              inputs,
                              targets,
                              predicted,
                              batch_size,
                              decode_output):
    """Process output if its sampling decoding."""
    for i in range(batch_size):
      curr_source = self.decode_tokens(inputs[i])
      curr_ref = self.decode_tokens(targets[i])
      samples = []
      for j in range(int(self.mt_eval_config.get('sample_size'))):
        curr_pred = self.decode_tokens(predicted[i][j])
        samples.append(curr_pred)
      decode_output.source_list.append(curr_source)
      decode_output.reference_list.append(curr_ref)
      decode_output.translation_list.append(samples)
