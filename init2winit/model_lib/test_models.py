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

"""Tests for models.py.

"""

import copy
import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.model_lib import local_attention_transformer
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

# Automatically test all defined models.
autoencoder_models = ['autoencoder', 'convolutional_autoencoder']
text_models = ['transformer', 'performer', 'lstm']
seq2seq_models = ['xformer_translate', 'xformer_translate_binary']
dtypes = ['bfloat16', 'float32']

autoencoder_keys = [('test_{}'.format(m), m) for m in autoencoder_models]
excluded_classification_models = text_models + autoencoder_models + seq2seq_models + [
    'nqm', 'gnn', 'dlrm', 'vit', 'unet', 'conformer', 'deepspeech',
    'local_attention_transformer'
]
classification_keys = [
    ('test_{}'.format(m), m)
    for m in models._ALL_MODELS.keys()  # pylint: disable=protected-access
    if m not in excluded_classification_models
]
text_keys = [('test_{}'.format(m), m) for m in text_models]
dtype_keys = [('test_{}'.format(t), t) for t in dtypes]
remat_scan_keys = [('test_no_remat_scan', None), ('test_remat_scan', (2, 2))]
dtype_and_remat_scan_keys = [('_'.join([x[0], y[0]]), x[1], y[1]) for
                             x, y in itertools.product(dtype_keys,
                                                       remat_scan_keys)]


class ModelsTest(parameterized.TestCase):
  """Tests for initializers.py."""

  @parameterized.named_parameters(*classification_keys)
  def test_classification_model(self, model_str):
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
  def test_text_models(self, model_str):
    """Test forward pass of the transformer model."""

    # TODO(gilmer): Find a clean way to handle small test hparams.
    vocab_size = 16

    small_hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': 16,
        'emb_dim': 32,
        'num_heads': 2,
        'num_layers': 3,
        'qkv_dim': 32,
        'label_smoothing': 0.1,
        'mlp_dim': 64,
        'max_target_length': 64,
        'max_eval_target_length': 64,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'momentum': 0.9,
        'normalizer': 'layer_norm',
        'lr_hparams': {
            'base_lr': 0.005,
            'schedule': 'constant'
        },
        'output_shape': (vocab_size,),
        'model_dtype': 'float32',
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'decode': False,
        'vocab_size': vocab_size,
        'hidden_size': 32,
        'bidirectional': False,
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

    # Check that the forward pass works with mutated batch_stats.
    # Due to a bug in flax, this jit is required, otherwise the model errors.
    @jax.jit
    def forward_pass(params, xs, dropout_rng):
      outputs, new_batch_stats = model.flax_module.apply(
          {'params': params, 'batch_stats': batch_stats},
          xs,
          mutable=['batch_stats'],
          rngs={'dropout': dropout_rng},
          train=True)
      return outputs, new_batch_stats

    outputs, new_batch_stats = forward_pass(params, xs, dropout_rng)
    self.assertEqual(outputs.shape,
                     (text_input_shape[0], text_input_shape[1], vocab_size))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats:
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_module.apply(
        {'params': params, 'batch_stats': batch_stats}, xs, train=False)
    self.assertEqual(outputs.shape,
                     (text_input_shape[0], text_input_shape[1], vocab_size))

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
        'output_shape': (98302,),
        'batch_size': 8,
        'max_target_length': 8192,
        'vocab_size': 98302,
    })
    model_cls = models.get_model(model_str)
    dropout_rng, params_rng = jax.random.split(jax.random.PRNGKey(0))
    loss = 'cross_entropy'
    metrics = 'classification_metrics'
    model = model_cls(model_hps, {}, loss, metrics)
    inputs = jnp.array(np.random.randint(size=(1, 8192), low=1, high=98302))
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
    self.assertEqual(outputs.shape, (1, 8192, 98302))


class ExtraLocalAttentionTransformerTests(parameterized.TestCase):
  """Tests support functions in local_attention_transformer.py.

  Reference shapes and data types come from Colab notebooks,
  where TF and JAX/FLAX support functions where compared.

  """

  def test_decode_step_to_index(self):
    """Tests support function decode_step_to_index."""
    decode_step = 10
    array_shape = (8192,)
    output = local_attention_transformer.decode_step_to_index(
        decode_step, array_shape)

    self.assertEqual(output, (10,))

  def test_get_item_at_decode_step(self):
    """Tests support function get_item_at_decode_step."""
    x = np.array(np.random.rand(1, 256, 16), dtype=np.float32)
    decode_step = 10
    output = local_attention_transformer.get_item_at_decode_step(x, decode_step)

    self.assertEqual(output.shape, (1, 1, 16))
    self.assertEqual(output.dtype, np.float32)

  def test_embedding_to_padding(self):
    """Tests support function embedding_to_padding."""
    x = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    output = local_attention_transformer.embedding_to_padding(x)

    self.assertEqual(output.shape, (8, 1, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_ones_matrix_band_part(self):
    """Tests support function ones_matrix_band_part."""
    num_rows = 2
    num_cols = 2
    max_backward = 10
    max_forward = 10
    output_shape = [1, 1, 2, 2]
    output = local_attention_transformer.ones_matrix_band_part(
        num_rows, num_cols, max_backward, max_forward, output_shape)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_bias_local(self):
    """Tests support function attention_bias_local."""
    length = 2
    max_backward = 10
    max_forward = 10
    output = local_attention_transformer.attention_bias_local(
        length, max_backward, max_forward)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_bias_lower_triangle(self):
    """Tests support function attention_bias_lower_triangle."""
    length = 2
    output = local_attention_transformer.attention_bias_lower_triangle(length)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_causal_attention_bias_nd(self):
    """Tests support function causal_attention_bias_nd."""
    output = local_attention_transformer.causal_attention_bias_nd()

    self.assertEqual(output.shape, (1, 1, 256, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_maybe_tile(self):
    """Tests support function maybe_tile."""
    x = np.array(np.random.rand(1, 2, 2), dtype=np.float32)
    y = np.array(np.random.rand(1, 4, 3), dtype=np.float32)
    output = local_attention_transformer.maybe_tile(x, y)

    self.assertEqual(output[0].shape, (1, 4, 2))
    self.assertEqual(output[1].shape, (1, 4, 3))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_local_attention_bias_nd(self):
    """Tests support function local_attention_bias_nd."""
    x = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    output = local_attention_transformer.local_attention_bias_nd(x)

    self.assertEqual(output.shape, (8, 1, 256, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_pad_to_multiple_nd(self):
    """Tests support function pad_to_multiple_nd."""
    x = np.array(np.random.rand(1, 4, 4), dtype=np.float32)
    block_shape = (2, 10)
    output = local_attention_transformer.pad_to_multiple_nd(x, block_shape)

    self.assertEqual(output.shape, (1, 4, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_select_block_for_decode_step(self):
    """Tests support function select_block_for_decode_step."""
    x = np.array(np.random.rand(1, 10, 256, 16), dtype=np.float32)
    decode_step = 10
    output = local_attention_transformer.select_block_for_decode_step(
        x, decode_step)

    self.assertEqual(output.shape, (1, 1, 256, 16))
    self.assertEqual(output.dtype, np.float32)

  def test_break_into_blocks_nd(self):
    """Tests support function break_into_blocks_nd."""
    x = np.array(np.random.rand(9, 9, 9, 9), dtype=np.float32)
    block_shape = (3,)
    output = local_attention_transformer.break_into_blocks_nd(x, block_shape)

    self.assertEqual(output.shape, (81, 3, 3, 9))
    self.assertEqual(output.dtype, np.float32)

  def test_break_into_memory_blocks_nd(self):
    """Tests support function break_into_memory_blocks_nd."""
    x = np.array(np.random.rand(1, 256, 1), dtype=np.float32)
    output = local_attention_transformer.break_into_memory_blocks_nd(x)

    self.assertEqual(output.shape, (1, 1, 512, 1))
    self.assertEqual(output.dtype, np.float32)

  def test_flatten_blocks_nd(self):
    """Tests support function flatten_blocks_nd."""
    x = np.array(np.random.rand(1, 1, 3, 4), dtype=np.float32)
    output = local_attention_transformer.flatten_blocks_nd(x)

    self.assertEqual(output.shape, (1, 1, 3, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_unflatten_blocks_nd(self):
    """Tests support function unflatten_blocks_nd."""
    x = np.array(np.random.rand(1, 2, 3, 4), dtype=np.float32)
    blocks_per_dimension = [2]
    output = local_attention_transformer.unflatten_blocks_nd(
        x, blocks_per_dimension)

    self.assertEqual(output.shape, (1, 2, 3, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_break_bias_into_blocks(self):
    """Tests support function break_bias_into_blocks."""
    x = np.array(np.random.rand(2, 10), dtype=np.float32)
    output = local_attention_transformer.break_bias_into_blocks(x)

    self.assertEqual(output.shape, (16, 1, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_cast_like(self):
    """Tests support function cast_like."""
    x = np.array(np.random.rand(1, 2, 10), dtype=np.float32)
    y = np.array(np.random.rand(1, 2, 10), dtype=np.int32)
    output = local_attention_transformer.cast_like(x, y)

    self.assertEqual(output.shape, (1, 2, 10))
    self.assertEqual(output.dtype, np.int32)

  def test_generate_relative_positions_matrix(self):
    """Tests support function generate_relative_positions_matrix."""
    length_q = 10
    length_k = 10
    output = local_attention_transformer.generate_relative_positions_matrix(
        length_q, length_k)

    self.assertEqual(output.shape, (10, 10))
    self.assertEqual(output.dtype, np.int32)

  def test_generate_relative_positions_embeddings(self):
    """Tests support function generate_relative_positions_embeddings."""
    length_q = 10
    length_k = 10
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.RelativePositionEmbeddings(
        embed_layer_name='unit_test')
    params = model.init(rng_key, length_q, length_k)
    output = model.apply(params, length_q, length_k)

    self.assertEqual(output.shape, (10, 10, 129))
    self.assertEqual(output.dtype, np.float32)

  def test_relative_attention_inner(self):
    """Tests support function relative_attention_inner."""
    x = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)
    y = np.array(np.random.rand(1, 1, 4, 1), dtype=np.float32)
    z = np.array(np.random.rand(1, 4, 3), dtype=np.float32)
    transpose = False
    output = local_attention_transformer.relative_attention_inner(
        x, y, z, transpose)

    self.assertEqual(output.shape, (1, 1, 1, 3))
    self.assertEqual(output.dtype, np.float32)

  def test_dot_product_attention_relative(self):
    """Tests support function dot_product_attention_relative."""
    q = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)
    k = np.array(np.random.rand(1, 1, 4, 4), dtype=np.float32)
    v = np.array(np.random.rand(1, 1, 4, 4), dtype=np.float32)
    b = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)

    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.RelativeDotProductAttention()
    params = model.init(rng_key, q, k, v, b)
    output = model.apply(params, q, k, v, b)

    self.assertEqual(output[0].shape, (1, 1, 1, 4))
    self.assertEqual(output[1].shape, (1, 1, 1, 4))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_dot_product_attention(self):
    """Tests support function dot_product_attention."""
    q = np.array(np.random.rand(8, 1, 256, 1024), dtype=np.float32)
    k = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    v = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    b = np.array(np.random.rand(1, 1, 256, 512), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.DotProductAttention()
    params = model.init(rng_key, q, k, v, b)
    output = model.apply(params, q, k, v, b)

    self.assertEqual(output[0].shape, (8, 1, 256, 1024))
    self.assertEqual(output[1].shape, (8, 1, 256, 512))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_combine_heads_nd(self):
    """Tests support function combine_heads_nd."""
    x = np.array(np.random.rand(1, 2, 3), dtype=np.float32)
    output = local_attention_transformer.combine_heads_nd(x)

    self.assertEqual(output.shape, (1, 6))
    self.assertEqual(output.dtype, np.float32)

  def test_put_back_blocks_nd(self):
    """Tests support function put_back_blocks_nd."""
    x = np.array(np.random.rand(1, 2, 3, 5), dtype=np.float32)
    block_shape = (3,)
    output = local_attention_transformer.put_back_blocks_nd(x, block_shape)

    self.assertEqual(output.shape, (1, 6, 5))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_nd(self):
    """Tests support function attention_nd."""
    q = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    k = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    v = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.LocalAttention()
    params = model.init(rng_key, q, k, v)
    output = model.apply(params, q, k, v)

    self.assertEqual(output.shape, (1, 8, 256, 1024))
    self.assertEqual(output.dtype, np.float32)

  def test_split_heads_nd(self):
    """Tests support function split_heads_nd."""
    x = np.array(np.random.rand(8, 8), dtype=np.float32)
    output = local_attention_transformer.split_heads_nd(x)

    self.assertEqual(output.shape, (8, 8, 1))
    self.assertEqual(output.dtype, np.float32)

  def test_put_item_in_decode_step(self):
    """Tests support function put_item_in_decode_step."""
    x = jnp.array(np.random.rand(3, 8, 1, 4), dtype=np.float32)
    output = local_attention_transformer.put_item_in_decode_step(
        input_x=x, decode_step=1)

    self.assertEqual(output.shape, (3, 8, 1, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_multihead_attention_nd(self):
    """Tests support function multihead_attention_nd."""
    x = np.array(np.random.rand(1, 2, 3), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.MultiHeadAttention()
    params = model.init(rng_key, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 2, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_get_timing_signal_1d(self):
    """Tests support function get_timing_signal_1d."""
    length = 10
    channels = 2
    output = local_attention_transformer.get_timing_signal_1d(length, channels)

    self.assertEqual(output.shape, (1, 10, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_process_input(self):
    """Tests support function process_input."""
    x = jnp.array(np.random.rand(1, 8192), dtype=jnp.int32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.ProcessInput()
    params = model.init(rng_key, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 8192, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_process_partial_targets_decoding(self):
    """Tests support function process_partial_targets_decoding."""
    x = np.array(np.random.rand(1, 256), dtype=np.float32)
    output = local_attention_transformer.process_partial_targets_decoding(
        targets=x)

    self.assertEqual(output.shape, (1, 256))
    self.assertEqual(output.dtype, np.float32)

  def test_feedforward(self):
    """Tests support function feedforward."""
    x = np.array(np.random.rand(1, 256), dtype=np.float32)
    feedforward_depths = [4096, 1032]

    key1 = jax.random.PRNGKey(0)
    model = local_attention_transformer.FeedForward(
        feedforward_depths=feedforward_depths)
    params = model.init(key1, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_layer_postprocess(self):
    """Tests support function layer_postprocess."""
    x = np.array(np.random.rand(2, 3), dtype=np.float32)
    y = np.array(np.random.rand(2, 3), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.LayerPostProcess()
    params = model.init(rng_key, x, y)
    output = model.apply(params, x, y)

    self.assertEqual(output.shape, (2, 3))
    self.assertEqual(output.dtype, np.float32)

  def test_decoder_block(self):
    """Tests support function decoder_block."""
    feedforward_depths = [4096, 1032]
    x = jnp.array(np.random.rand(1, 2, 1032), dtype=jnp.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.DecoderBlock(
        feedforward_depths=feedforward_depths)
    params = model.init(rng_key, x)
    imp_output = model.apply(params, x)

    self.assertEqual(imp_output.shape, (1, 2, 1032))
    self.assertEqual(imp_output.dtype, np.float32)


if __name__ == '__main__':
  absltest.main()
