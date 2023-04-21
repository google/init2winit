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

"""Tests for models.py.

"""

import copy
import functools
import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from init2winit.model_lib import models
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util
import jraph
from ml_collections.config_dict import config_dict
import numpy as np


HIDDEN_SIZES = (50, 50)

INPUT_SHAPE = {
    'classification': (64, 32, 32, 3),
    'autoencoder': (64, 28, 28, 1),
    'convolutional_autoencoder': (64, 28, 28, 1),
}
OUTPUT_SHAPE = {
    'classification': (5,),
    'autoencoder': (784,),
    'convolutional_autoencoder': (28, 28, 1),
}

DATA_HPS = {
    'adabelief_densenet': {
        'input_shape': (32, 32, 3),
        'num_layers': 1,
        'growth_rate': 1,
        'output_shape': (5,),
        'batch_size': 1
    },
    'adabelief_resnet': {
        'input_shape': (32, 32, 3),
        'num_layers': 1,
        'output_shape': (5,),
    },
    'adabelief_vgg': {
        'input_shape': (32, 32, 3),
        'num_layers': 1,
        'output_shape': (5,),
    },
    'autoencoder': {
        'input_shape': (28, 28, 1),
        'output_shape': (784,),
    },
    # TODO(kasimbeg): fix issue with tokenizer_vocab_path
    'conformer': {
        'input_shape': (64,),
        'max_eval_target_length':
            64,
        'max_target_length':
            64,
        'output_shape': (16,),
        'vocab_size':
            16,
        'tokenizer_type':
            'WPM',
    },
    'mlcommons_conformer': {
        'input_shape': (64,),
        'max_eval_target_length':
            64,
        'max_target_length':
            64,
        'output_shape': (16,),
        'vocab_size':
            16,
        'tokenizer_type':
            'WPM',
    },
    'convolutional_autoencoder': {
        'input_shape': ((28, 28, 1)),
        'output_shape': ((28, 28, 1)),
    },
    'deepspeech': {
        'max_input_length': 64,
        'max_target_length': 64,
        'input_shape': [(64,), (64,)],
        'output_shape': (-1, 32),
        'tokenizer_type': 'WPM',
    },
    'mlcommons_deepspeech': {
        'max_input_length': 64,
        'max_target_length': 64,
        'input_shape': [(64,), (64,)],
        'output_shape': (-1, 32),
        'tokenizer_type': 'WPM',
    },
    'dlrm': {
        'input_shape': (39,),  # Like criteo1tb
    },
    'dlrm_resnet': {
        'input_shape': (39,),  # Like criteo1tb
    },
    'fake_resnet': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'fully_connected': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'gnn': {
        'batch_size': 5,
        'hidden_dims': (10,),
        'input_edge_shape': (5,),
        'input_node_shape': (5,),
        'latent_dim': 10,
        'normalizer': 'batch_norm',
        'output_shape': (5,),
    },
    'local_attention_transformer': {
        'input_shape': (64,),
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
        'num_decoder_layers': 1,
    },
    'lstm': {
        'input_shape': (32,),
        'max_target_length': 64,
        'ouptut_shape': (16,),
        'sequence_length': 64,
        'vocab_size': 16,
    },
    'max_pooling_cnn': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'mlperf_resnet': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'nqm': {
        'input_shape': (10,),
        'output_shape': (1,),
    },
    'performer': {
        'input_shape': (64,),
        'max_eval_target_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
    'resnet': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'simple_cnn': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'transformer': {
        'input_shape': (64,),
        'max_eval_target_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
    'unet': {
        'input_shape': (64, 64),
        'output_shape': (64, 64),
    },
    'vit': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'wide_resnet': {
        'input_shape': (32, 32, 3),
        'output_shape': (5,),
    },
    'xformer_translate_binary': {
        'dec_num_layers': 2,
        'enc_num_layers': 2,
        'input_shape': [(64,), (64,)],
        'max_eval_target_length': 64,
        'max_predict_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
    'xformer_translate': {
        'dec_num_layers': 2,
        'enc_num_layers': 2,
        'input_shape': [(64,), (64,)],
        'max_eval_target_length': 64,
        'max_predict_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
    'xformer_translate_mlc_variant': {
        'dec_num_layers': 2,
        'enc_num_layers': 2,
        'input_shape': [(64,), (64,)],
        'max_eval_target_length': 64,
        'max_predict_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
    'mlcommons_xformer_translate': {
        'dec_num_layers': 2,
        'enc_num_layers': 2,
        'input_shape': [(64,), (64,)],
        'max_eval_target_length': 64,
        'max_predict_length': 64,
        'max_target_length': 64,
        'output_shape': (16,),
        'vocab_size': 16,
    },
}

LOSS_NAME = {
    'adabelief_densenet': 'cross_entropy',
    'adabelief_resnet': 'cross_entropy',
    'adabelief_vgg': 'cross_entropy',
    'autoencoder': 'sigmoid_binary_cross_entropy',
    'conformer': 'ctc',
    'mlcommons_conformer': 'ctc',
    'convolutional_autoencoder': 'sigmoid_binary_cross_entropy',
    'deepspeech': 'ctc',
    'mlcommons_deepspeech': 'ctc',
    'dlrm': 'sigmoid_binary_cross_entropy',
    'dlrm_resnet': 'sigmoid_binary_cross_entropy',
    'fake_resnet': 'cross_entropy',
    'fully_connected': 'cross_entropy',
    'gnn': 'sigmoid_binary_cross_entropy',
    'local_attention_transformer': 'cross_entropy',
    'lstm': 'cross_entropy',
    'max_pooling_cnn': 'cross_entropy',
    'mlperf_resnet': 'cross_entropy',
    'nqm': 'cross_entropy',
    'performer': 'cross_entropy',
    'resnet': 'cross_entropy',
    'simple_cnn': 'cross_entropy',
    'transformer': 'cross_entropy',
    'unet': 'mean_absolute_error',
    'vit': 'cross_entropy',
    'wide_resnet': 'cross_entropy',
    'xformer_translate_binary': 'cross_entropy',
    'xformer_translate': 'cross_entropy',
    'xformer_translate_mlc_variant': 'cross_entropy',
    'mlcommons_xformer_translate': 'cross_entropy',
}

METRICS_NAME = {
    'adabelief_densenet': 'classification_metrics',
    'adabelief_resnet': 'classification_metrics',
    'adabelief_vgg': 'classification_metrics',
    'autoencoder': 'binary_autoencoder_metrics',
    'conformer': 'ctc_metrics',
    'mlcommons_conformer': 'ctc_metrics',
    'convolutional_autoencoder': 'binary_autoencoder_metrics',
    'deepspeech': 'ctc_metrics',
    'mlcommons_deepspeech': 'ctc_metrics',
    'dlrm': 'binary_classification_metrics_dlrm_no_auc',
    'dlrm_resnet': 'binary_classification_metrics_dlrm_no_auc',
    'fake_resnet': 'classification_metrics',
    'fully_connected': 'classification_metrics',
    'gnn': 'binary_classification_metrics',
    'local_attention_transformer': 'classification_metrics',
    'lstm': 'classification_metrics',
    'max_pooling_cnn': 'classification_metrics',
    'mlperf_resnet': 'classification_metrics',
    'nqm': 'classification_metrics',
    'performer': 'classification_metrics',
    'resnet': 'classification_metrics',
    'simple_cnn': 'classification_metrics',
    'transformer': 'classification_metrics',
    'unet': 'image_reconstruction_metrics',
    'vit': 'classification_metrics',
    'wide_resnet': 'classification_metrics',
    'xformer_translate_binary': 'classification_metrics',
    'xformer_translate': 'classification_metrics',
    'mlcommons_xformer_translate': 'classification_metrics',
    'xformer_translate_mlc_variant': 'classification_metrics',
}

# Automatically test all defined models.
all_models = models._ALL_MODELS.keys()  # pylint: disable=protected-access

autoencoder_models = ['autoencoder', 'convolutional_autoencoder']
text_models = ['transformer', 'performer', 'lstm']
classification_models = [
    'fully_connected', 'simple_cnn', 'max_pooling_cnn', 'wide_resnet', 'resnet',
    'adabelief_densenet', 'adabelief_vgg', 'fake_resnet'
]
binary_classification_models = ['dlrm', 'dlrm_resnet']  # TODO(kasimbeg)
generative_models = ['unet']  # TODO(kasimbeg)

# Model arguments
dtypes = ['bfloat16', 'float32']

# Construct keys for tests for initialization
# TODO(kasimbeg): Add HPS for excluded models and include in tests.
skipped_models = [
    'conformer',
    'deepspeech',
    'local_attention_transformer',
    'mlcommons_conformer',
    'mlcommons_deepspeech',
]
# pylint: disable=g-complex-comprehension
model_init_keys = [
    ('test_model_{}_dtype_{}'.format(
        model_str,
        dtype,
    ), model_str, dtype)
    for model_str, dtype in itertools.product(all_models, dtypes)
    if model_str not in skipped_models
]

# Construct keys for tests for forward passes
autoencoder_keys = [('test_{}'.format(m), m) for m in autoencoder_models]
classification_keys = [('test_{}'.format(m), m) for m in classification_models]
binary_classification_keys = [
    ('test_{}'.format(m), m) for m in binary_classification_models
]
text_keys = [('test_{}_{}'.format(m, d), m, d)
             for m, d in itertools.product(text_models, dtypes)
             if d != 'bfloat16' or m == 'transformer']
dtype_keys = [('test_{}'.format(t), t) for t in dtypes]
remat_scan_keys = [('test_no_remat_scan', None), ('test_remat_scan', (2, 2))]
dtype_and_remat_scan_keys = [
    ('_'.join([x[0], y[0]]), x[1], y[1])
    for x, y in itertools.product(dtype_keys, remat_scan_keys)
]

# Construct keys for LSTM parameterized test.
# Params are (test_name, hidden_sizes, emb_dim, bidirectional,
# tie_embeddings, projection_layer)
lstm_keys = [
    ('test_lstm_3_layers', [32, 32, 16], 16, False, False, False),
    ('test_lstm_2_layers', [16, 16], 16, False, False, False),
    ('test_lstm_bidirectional', [32, 32, 16], 16, True, False, False),
    ('test_lstm_tie_embedding', [32, 32, 16], 16, False, True, False),
    ('test_lstm_projection', [32, 32, 32], 16, False, False, True),
]


# TODO(kasimbeg): clean this up after get_fake_inputs is implemented for
# all models
def _get_fake_inputs_for_initialization(model, hps):
  """Temporary helper method to get fake inputs for initialization test."""
  fake_inputs_hps = copy.copy(hps)
  fake_inputs_hps.batch_size = 2
  fake_inputs = model.get_fake_inputs(fake_inputs_hps)

  if fake_inputs:
    return fake_inputs
  else:
    raise NotImplementedError(
        'Method get_fake_inputs not implemented for model.')

  return fake_inputs


class ModelsTest(parameterized.TestCase):
  """Tests for initializers.py."""

  @parameterized.named_parameters(*model_init_keys)
  def test_model_initialization(self, model_str, model_dtype):
    """Test model initializations."""
    model_cls = models.get_model(model_str)
    hps = models.get_model_hparams(model_str)
    hps.update(DATA_HPS[model_str])
    hps.model_dtype = model_dtype

    model = model_cls(
        hps,
        dataset_meta_data={
            'shift_inputs': True,
            'causal': True
        },
        loss_name=LOSS_NAME[model_str],
        metrics_name=METRICS_NAME[model_str])

    fake_input_batch = _get_fake_inputs_for_initialization(model, hps)

    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng, num=2)

    # initialize model
    model_init_fn = functools.partial(model.flax_module.init, train=False)
    try:
      init_dict = model_init_fn({
          'params': params_rng,
          'dropout': dropout_rng
      }, *fake_input_batch)
    except Exception as e:
      logging.info(hps)
      raise e

    self.assertNotEmpty(init_dict)

  @parameterized.named_parameters(*classification_keys)
  def test_classification_models(self, model_str):
    """Test forward pass of the image models."""

    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)
    loss = 'cross_entropy'
    metrics = 'classification_metrics'
    hps = copy.copy(model_hps)
    hps.update({'output_shape': OUTPUT_SHAPE['classification']})
    rng = jax.random.PRNGKey(0)
    dropout_rng, params_rng = jax.random.split(rng)
    model = model_cls(hps, {}, loss, metrics)
    xs = jnp.array(np.random.normal(size=INPUT_SHAPE['classification']))
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, xs)
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    # Check that the forward pass works with mutated batch_stats.
    outputs, new_batch_stats = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        xs,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng},
        train=True)
    self.assertEqual(outputs.shape, (INPUT_SHAPE['classification'][0],
                                     OUTPUT_SHAPE['classification'][-1]))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats:
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats}, xs, train=False)
    self.assertEqual(
        outputs.shape,
        (INPUT_SHAPE['classification'][0], OUTPUT_SHAPE['classification'][-1]))

  @parameterized.named_parameters(*text_keys)
  def test_text_models(self, model_str, dtype_str):
    """Test forward pass of the transformer model."""

    # TODO(gilmer): Find a clean way to handle small test hparams.
    vocab_size = 16

    small_hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': 16,
        'emb_dim': 32,
        'num_heads': 2,
        'num_layers': 3,
        'tie_embeddings': False,
        'projection_layer': False,
        'qkv_dim': 32,
        'label_smoothing': 0.1,
        'mlp_dim': 64,
        'max_target_length': 64,
        'max_eval_target_length': 64,
        'dropout_rate': 0.1,
        'recurrent_dropout_rate': 0.1,
        'residual_connections': False,
        'cell_kwargs': {},
        'attention_dropout_rate': 0.1,
        'momentum': 0.9,
        'normalizer': 'layer_norm',
        'lr_hparams': {'base_lr': 0.005, 'schedule': 'constant'},
        'output_shape': (vocab_size,),
        'model_dtype': dtype_str,
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'decode': False,
        'vocab_size': vocab_size,
        'hidden_sizes': [32, 32, 32],
        'bidirectional': False,
        'normalize_attention': False,
    })

    text_input_shape = (32, 64)  # batch_size, max_target_length
    model_cls = models.get_model(model_str)
    rng = jax.random.PRNGKey(0)
    loss = 'cross_entropy'
    metrics = 'classification_metrics'
    model = model_cls(small_hps, {
        'max_len': 64,
        'shift_inputs': True,
        'causal': True
    }, loss, metrics)
    xs = jnp.array(
        np.random.randint(size=text_input_shape, low=1, high=vocab_size))
    dropout_rng, params_rng = jax.random.split(rng)

    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, xs)
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    param_type_matches_model_type = jax.tree_util.tree_map(
        lambda x: x.dtype == small_hps.model_dtype, params)
    self.assertTrue(
        jax.tree_util.tree_reduce(lambda x, y: x and y,
                                  param_type_matches_model_type))

    # Check that the forward pass works with mutated batch_stats.
    # Due to a bug in flax, this jit is required, otherwise the model errors.
    @jax.jit
    def forward_pass(params, xs, dropout_rng):
      outputs, new_batch_stats = model.flax_module.apply(
          {
              'params': params,
              'batch_stats': batch_stats
          },
          xs,
          mutable=['batch_stats'],
          capture_intermediates=True,
          rngs={'dropout': dropout_rng},
          train=True)
      return outputs, new_batch_stats

    outputs, new_batch_stats = forward_pass(params, xs, dropout_rng)

    if model_str == 'lstm':
      # This is to accomodate weight tying and mask token of value 0
      expected_output_size = vocab_size + 1
    else:
      expected_output_size = vocab_size

    self.assertEqual(
        outputs.shape,
        (text_input_shape[0], text_input_shape[1], expected_output_size),
    )

    intermediates_type_matches_model_type = jax.tree_util.tree_map(
        lambda x: x.dtype == small_hps.model_dtype,
        new_batch_stats['intermediates'],
    )
    self.assertTrue(
        jax.tree_util.tree_reduce(
            lambda x, y: x and y, intermediates_type_matches_model_type
        )
    )

    # If it's a batch norm model check the batch stats changed.
    if batch_stats:
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats}, xs, train=False
    )
    self.assertEqual(
        outputs.shape,
        (text_input_shape[0], text_input_shape[1], expected_output_size),
    )

  @parameterized.named_parameters(*dtype_and_remat_scan_keys)
  def test_translate_model(self, dtype_str, remat_scan_lengths):
    """Test forward pass of the translate model."""
    vocab_size = 16

    if remat_scan_lengths:
      num_layers = None
    else:
      num_layers = 2

    small_hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': 16,
        'share_embeddings': False,
        'logits_via_embedding': False,
        'emb_dim': 32,
        'num_heads': 2,
        'enc_num_layers': num_layers,
        'dec_num_layers': num_layers,
        'enc_remat_scan_lengths': remat_scan_lengths,
        'dec_remat_scan_lengths': remat_scan_lengths,
        'qkv_dim': 32,
        'label_smoothing': 0.1,
        'mlp_dim': 64,
        'max_target_length': 64,
        'max_eval_target_length': 64,
        'normalizer': 'pre_layer_norm',
        'max_predict_length': 64,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'momentum': 0.9,
        'lr_hparams': {
            'base_lr': 0.005,
            'schedule': 'constant'
        },
        'vocab_size': vocab_size,
        'output_shape': (vocab_size,),
        'model_dtype': dtype_str,
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'enc_self_attn_kernel_init': 'xavier_uniform',
        'dec_self_attn_kernel_init': 'xavier_uniform',
        'dec_cross_attn_kernel_init': 'xavier_uniform',
        'decode': False,
        'normalize_attention': False,
    })
    text_src_input_shape = (32, 64)  # batch_size, max_source_length
    text_tgt_input_shape = (32, 40)  # batch_size, max_target_length
    model_cls = models.get_model('xformer_translate')
    rng = jax.random.PRNGKey(0)
    loss = 'cross_entropy'
    metrics = 'classification_metrics'
    model = model_cls(small_hps, {
        'shift_outputs': True,
        'causal': True
    }, loss, metrics)
    xs = jnp.array(
        np.random.randint(size=text_src_input_shape, low=1,
                          high=vocab_size))
    ys = jnp.array(
        np.random.randint(size=text_tgt_input_shape, low=1,
                          high=vocab_size))
    dropout_rng, params_rng = jax.random.split(rng)
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, xs, ys)
    params = init_dict['params']

    param_type_matches_model_type = jax.tree_util.tree_map(
        lambda x: x.dtype == small_hps.model_dtype, params)
    self.assertTrue(
        jax.tree_util.tree_reduce(lambda x, y: x and y,
                                  param_type_matches_model_type))

    # Test forward pass.
    @jax.jit
    def forward_pass(params, xs, ys, dropout_rng):
      outputs, intermediates = model.flax_module.apply(
          {'params': params},
          xs,
          ys,
          rngs={'dropout': dropout_rng},
          capture_intermediates=True,
          train=True)
      return outputs, intermediates

    logits, intermediates = forward_pass(params, xs, ys, dropout_rng)
    # Testing only train mode
    # TODO(ankugarg): Add tests for individual encoder/decoder (inference mode).
    self.assertEqual(
        logits.shape,
        (text_tgt_input_shape[0], text_tgt_input_shape[1], vocab_size))

    intermediates_type_matches_model_type = jax.tree_util.tree_map(
        lambda x: x.dtype == small_hps.model_dtype, intermediates)
    self.assertTrue(
        jax.tree_util.tree_reduce(lambda x, y: x and y,
                                  intermediates_type_matches_model_type))

  def test_nqm(self):
    """Test the noisy quadratic model."""
    batch_size = 2
    dim = 10
    model_hps = config_dict.ConfigDict(
        dict(
            input_shape=(dim,),
            output_shape=(1,),
            rng_seed=-1,
            hessian_decay_power=1.0,
            noise_decay_power=1.0,
            nqm_mode='diagH_diagC',
            model_dtype='float32',
        ))

    model_cls = models.get_model('nqm')
    params_rng = jax.random.PRNGKey(0)
    model = model_cls(model_hps, {}, None, None)
    noise_eps = jnp.array(np.random.normal(size=(batch_size, dim)))
    xs = np.zeros((batch_size, dim))
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    params = model_init_fn({'params': params_rng}, xs)['params']
    model_x = params['x']

    def loss(params, inputs):
      return model.training_cost(params, batch=inputs)

    grad_loss = jax.grad(loss, has_aux=True)

    hessian = np.diag(
        np.array([
            1.0 / np.power(i, model_hps.hessian_decay_power)
            for i in range(1, dim + 1)
        ]))
    noise_matrix = np.diag(
        np.array([
            1.0 / np.power(i, model_hps.noise_decay_power / 2.0)
            for i in range(1, dim + 1)
        ]))

    noise = jnp.dot(noise_eps, noise_matrix)
    mean_noise = np.mean(noise, axis=0)

    # NQM gradient = Hx + eps   where eps ~ N(0, C / batch_size).
    expected_grad = np.dot(hessian, model_x) + mean_noise

    g = grad_loss(params, {'inputs': noise_eps})[0]['x']

    grad_error = np.sum(np.abs(g - expected_grad))
    self.assertAlmostEqual(grad_error, 0.0, places=5)

  @parameterized.named_parameters(*autoencoder_keys)
  def test_autoencoder_model(self, model_str):
    """Test forward pass of the autoencoder models."""

    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)
    loss = 'sigmoid_binary_cross_entropy'
    metrics = 'binary_autoencoder_metrics'
    hps = copy.copy(model_hps)
    hps.update({'output_shape': OUTPUT_SHAPE[model_str]})
    params_rng = jax.random.PRNGKey(0)
    model = model_cls(hps, {}, loss, metrics)
    xs = jnp.array(np.random.normal(size=INPUT_SHAPE[model_str]))
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, xs)
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    # Check that the forward pass works with mutated batch_stats.
    outputs, new_batch_stats = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        xs,
        mutable=['batch_stats'],
        train=True)
    self.assertEqual(
        outputs.shape,
        tuple([INPUT_SHAPE[model_str][0]] + list(OUTPUT_SHAPE[model_str])))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats:
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats}, xs, train=False)
    self.assertEqual(
        outputs.shape,
        tuple([INPUT_SHAPE[model_str][0]] + list(OUTPUT_SHAPE[model_str])))

  def test_graph_model(self):
    """Test forward pass of the GNN model."""
    edge_input_shape = (5,)
    node_input_shape = (5,)
    output_shape = (5,)
    model_str = 'gnn'
    model_hps = models.get_model_hparams(model_str)
    model_hps.update({'output_shape': output_shape,
                      'latent_dim': 10,
                      'hidden_dims': (10,),
                      'batch_size': 5,
                      'normalizer': 'batch_norm'})
    model_cls = models.get_model(model_str)
    rng = jax.random.PRNGKey(0)
    dropout_rng, params_rng = jax.random.split(rng)
    loss = 'sigmoid_binary_cross_entropy'
    metrics = 'binary_classification_metrics'
    model = model_cls(model_hps, {}, loss, metrics)

    num_graphs = 5
    node_per_graph = 3
    edge_per_graph = 9
    inputs = jraph.get_fully_connected_graph(
        n_node_per_graph=node_per_graph,
        n_graph=num_graphs,
        node_features=np.ones((num_graphs * node_per_graph,) +
                              node_input_shape),
    )
    inputs = inputs._replace(
        edges=np.ones((num_graphs * edge_per_graph,) + edge_input_shape))
    padded_inputs = jraph.pad_with_graphs(inputs, 20, 50, 7)
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, padded_inputs)
    params = init_dict['params']
    batch_stats = init_dict['batch_stats']

    # Check that the forward pass works with mutated batch_stats.
    outputs, _ = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats},
        padded_inputs,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng},
        train=True)
    self.assertEqual(outputs.shape, (7,) + output_shape)

  def test_local_attention_transformer(self):
    """Test forward pass of the local attention transformer."""
    model_str = 'local_attention_transformer'
    model_hps = models.get_model_hparams(model_str)
    model_hps.update({
        'output_shape': (8,),
        'max_target_length': 16,
        'vocab_size': 8,
        'num_decoder_layers': 1,
    })
    model_cls = models.get_model(model_str)
    dropout_rng, params_rng = jax.random.split(jax.random.PRNGKey(0))
    loss = 'cross_entropy'
    metrics = 'classification_metrics'
    model = model_cls(model_hps, {}, loss, metrics)
    inputs = jnp.array(np.random.randint(size=(1, 16), low=1, high=8))
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, inputs)
    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})
    outputs, _ = model.flax_module.apply(
        {
            'params': params,
            'batch_stats': batch_stats
        },
        inputs,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_rng},
        train=True)
    self.assertEqual(outputs.shape, (1, 16, 8))

  @parameterized.named_parameters(*lstm_keys)
  def test_lstm_model(
      self,
      hidden_sizes,
      emb_dim,
      bidirectional,
      tie_embeddings,
      projection_layer,
  ):
    """Test forward pass of the LSTM model."""
    small_model_hps = {
        'emb_dim': emb_dim,
        'hidden_sizes': hidden_sizes,
        'tie_embeddings': tie_embeddings,
        'bidirectional': bidirectional,
        'residual_connections': False,
        'cell_kwargs': {},
        'dropout_rate': 0.1,
        'recurrent_dropout_rate': 0.1,
        'batch_size': 16,
        'model_dtype': 'float32',
        'projection_layer': projection_layer,
    }
    model_cls = models.get_model('lstm')
    hps = models.get_model_hparams('lstm')
    hps.update(DATA_HPS['lstm'])
    hps.update(small_model_hps)

    assert isinstance(hps.hidden_sizes, list)

    # make test input
    input_batch_shape = (hps.batch_size, hps.sequence_length)
    xs = jnp.array(
        np.random.randint(size=input_batch_shape, low=1, high=hps.vocab_size)
    )

    # initialize model
    model = model_cls(
        hps,
        dataset_meta_data={'shift_inputs': True, 'causal': True},
        loss_name=LOSS_NAME['lstm'],
        metrics_name=METRICS_NAME['lstm'],
    )

    fake_input_batch = _get_fake_inputs_for_initialization(model, hps)

    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng, num=2)
    model_init_fn = functools.partial(model.flax_module.init, train=False)
    init_dict = model_init_fn(
        {'params': params_rng, 'dropout': dropout_rng}, *fake_input_batch
    )

    params = init_dict['params']
    batch_stats = init_dict.get('batch_stats', {})

    # Check that the forward pass works with mutated batch_stats.
    # Due to a bug in flax, this jit is required, otherwise the model errors.
    @jax.jit
    def forward_pass(params, xs, dropout_rng):
      outputs, new_batch_stats = model.flax_module.apply(
          {'params': params, 'batch_stats': batch_stats},
          xs,
          mutable=['batch_stats'],
          capture_intermediates=True,
          rngs={'dropout': dropout_rng},
          train=True,
      )
      return outputs, new_batch_stats

    outputs, _ = forward_pass(params, xs, dropout_rng)

    # Logit dimension includes mask token to support tying with embedding layer
    expected_output_size = hps.vocab_size + 1
    self.assertEqual(
        outputs.shape,
        (input_batch_shape[0], input_batch_shape[1], expected_output_size),
    )


if __name__ == '__main__':
  absltest.main()
