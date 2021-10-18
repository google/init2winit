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

"""Tests for models.py.

"""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from flax import nn
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
text_models = ['transformer', 'performer']
seq2seq_models = ['xformer_translate']

autoencoder_keys = [('test_{}'.format(m), m) for m in autoencoder_models]
excluded_classification_models = text_models + autoencoder_models + seq2seq_models + [
    'nqm', 'gnn'
]
classification_keys = [
    ('test_{}'.format(m), m)
    for m in models._ALL_MODELS.keys()  # pylint: disable=protected-access
    if m not in excluded_classification_models
]
text_keys = [('test_{}'.format(m), m) for m in text_models]


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
    model = model_cls(hps, {}, loss, metrics)
    xs = jnp.array(np.random.normal(size=INPUT_SHAPE['classification']))
    rng, params_rng = jax.random.split(rng)
    with nn.stateful() as batch_stats:
      with nn.stochastic(params_rng):
        _, flax_module = model.flax_module_def.create(params_rng, xs)

    # Check that the forward pass works with mutated batch_stats.
    with nn.stateful(batch_stats) as new_batch_stats:
      with nn.stochastic(params_rng):
        outputs = flax_module(xs)
        self.assertEqual(outputs.shape, (INPUT_SHAPE['classification'][0],
                                         OUTPUT_SHAPE['classification'][-1]))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats.as_dict():
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    with nn.stateful(batch_stats, mutable=False):
      outputs = flax_module(xs, train=False)
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
        # Training HParams.
        'l2_decay_factor': 1e-4
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
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)

    with nn.stateful() as batch_stats:
      _, flax_module = model.flax_module_def.create_by_shape(
          params_rng, [(text_input_shape, jnp.float32)], train=False)

    # Check that the forward pass works with mutated batch_stats.
    # Due to a bug in flax, this jit is required, otherwise the model errors.
    @jax.jit
    def forward_pass(flax_module, xs, dropout_rng):
      with batch_stats.mutate() as new_batch_stats:
        with nn.stochastic(dropout_rng):
          return flax_module(xs, train=True), new_batch_stats

    outputs, new_batch_stats = forward_pass(flax_module, xs, dropout_rng)
    self.assertEqual(outputs.shape,
                     (text_input_shape[0], text_input_shape[1], vocab_size))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats.as_dict():
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    with nn.stateful(batch_stats, mutable=False):
      outputs = flax_module(xs, train=False)
    self.assertEqual(outputs.shape,
                     (text_input_shape[0], text_input_shape[1], vocab_size))

  def test_translate_model(self):
    """Test forward pass of the translate model."""
    vocab_size = 16
    small_hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': 16,
        'share_embeddings': False,
        'logits_via_embedding': False,
        'emb_dim': 32,
        'num_heads': 2,
        'enc_num_layers': 2,
        'dec_num_layers': 2,
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
        'output_shape': (vocab_size,),
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'enc_self_attn_kernel_init': 'xavier_uniform',
        'dec_self_attn_kernel_init': 'xavier_uniform',
        'dec_cross_attn_kernel_init': 'xavier_uniform'
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
        np.random.randint(size=text_src_input_shape, low=1, high=vocab_size))
    ys = jnp.array(
        np.random.randint(size=text_tgt_input_shape, low=1, high=vocab_size))
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    with nn.stateful() as batch_stats:
      _, flax_module = model.flax_module_def.create_by_shape(
          params_rng, [(text_src_input_shape, jnp.float32),
                       (text_tgt_input_shape, jnp.float32)],
          train=False)

    # Test forward pass.
    @jax.jit
    def forward_pass(flax_module, xs, ys, dropout_rng):
      with batch_stats.mutate() as new_batch_stats:
        with nn.stochastic(dropout_rng):
          return flax_module(xs, ys, train=True), new_batch_stats

    logits, _ = forward_pass(flax_module, xs, ys, dropout_rng)
    # Testing only train mode
    # TODO(ankugarg): Add tests for individual encoder/decoder (inference mode).
    self.assertEqual(
        logits.shape,
        (text_tgt_input_shape[0], text_tgt_input_shape[1], vocab_size))

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
    rng = jax.random.PRNGKey(0)
    model = model_cls(model_hps, {}, None, None)
    noise_eps = jnp.array(np.random.normal(size=(batch_size, dim)))
    rng, params_rng = jax.random.split(rng)
    _, flax_module = model.flax_module_def.create_by_shape(
        params_rng, [(batch_size, dim)])

    model_x = flax_module.params['x']

    def loss(model, inputs):
      return model(inputs)

    grad_loss = jax.grad(loss)

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

    g = grad_loss(flax_module, noise_eps).params['x']

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
    rng = jax.random.PRNGKey(0)
    model = model_cls(hps, {}, loss, metrics)
    xs = jnp.array(np.random.normal(size=INPUT_SHAPE[model_str]))
    rng, params_rng = jax.random.split(rng)
    with nn.stateful() as batch_stats:
      with nn.stochastic(params_rng):
        _, flax_module = model.flax_module_def.create(params_rng, xs)

    # Check that the forward pass works with mutated batch_stats.
    with nn.stateful(batch_stats) as new_batch_stats:
      with nn.stochastic(params_rng):
        outputs = flax_module(xs)
        self.assertEqual(
            outputs.shape,
            tuple([INPUT_SHAPE[model_str][0]] + list(OUTPUT_SHAPE[model_str])))

    # If it's a batch norm model check the batch stats changed.
    if batch_stats.as_dict():
      bflat, _ = ravel_pytree(batch_stats)
      new_bflat, _ = ravel_pytree(new_batch_stats)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    with nn.stateful(batch_stats, mutable=False):
      outputs = flax_module(xs, train=False)
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
    model_hps.update({'output_shape': output_shape})
    model_cls = models.get_model(model_str)
    rng = jax.random.PRNGKey(0)
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
    rng, params_rng = jax.random.split(rng)
    with nn.stateful() as batch_stats:
      _, flax_module = model.flax_module_def.create(
          params_rng, padded_inputs, train=False)

    # Check that the forward pass works with mutated batch_stats.
    with nn.stateful(batch_stats) as _:
      with nn.stochastic(params_rng):
        outputs = flax_module(padded_inputs)
        self.assertEqual(outputs.shape, (7,) + output_shape)


if __name__ == '__main__':
  absltest.main()
