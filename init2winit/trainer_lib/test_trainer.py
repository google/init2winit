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

"""Unit tests for trainer.py."""

import copy
import functools
import itertools
import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
from flax import linen as nn
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit import utils
from init2winit.dataset_lib import datasets
from init2winit.dataset_lib.small_image_datasets import Dataset
from init2winit.init_lib import init_utils
from init2winit.init_lib import initializers
from init2winit.model_lib import base_model
from init2winit.model_lib import metrics
from init2winit.model_lib import models
from init2winit.trainer_lib import trainer
import jax.numpy as jnp
import jax.random
import jraph
from ml_collections.config_dict import config_dict
import numpy as np
import pandas
import tensorflow.compat.v1 as tf  # importing this is needed for tfds mocking.
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

_VOCAB_SIZE = 4
_MAX_LEN = 32
_TEXT_BATCH_SIZE = 16
_TEXT_TRAIN_NUM_BATCHES = 20
_TEXT_TRAIN_SIZE = _TEXT_TRAIN_NUM_BATCHES * _TEXT_BATCH_SIZE


# These should match what is returned from the report function in trainer.py
def get_column_names():
  """Returns a list of the expected column names."""
  column_names = [
      'train/error_rate',
      'valid/error_rate',
      'test/error_rate',
      'train/num_examples',
      'valid/num_examples',
      'test/num_examples',
      'train/ce_loss',
      'valid/ce_loss',
      'test/ce_loss',
      'global_step',
      'learning_rate',
      'epoch',
      'eval_time',
      'train_steps_per_sec',
      'overall_steps_per_sec',
      'preemption_count',
      'train_cost',
      'grad_norm',
  ]
  return column_names


def _get_fake_text_dataset(batch_size, eval_num_batches):
  """Yields a single text batch repeatedly for train and test."""
  inputs = jnp.array(
      np.random.randint(low=0, high=_VOCAB_SIZE, size=(batch_size, _MAX_LEN)))
  batch = {
      'inputs': inputs,
      'targets': inputs,
      'weights': jnp.ones(inputs.shape),
  }

  def train_iterator_fn():
    while True:
      yield batch

  def eval_train_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def valid_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def test_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  meta_data = {
      'apply_one_hot_in_loss': True,
      'shift_inputs': True,
      'causal': True
  }
  return (Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                  test_epoch), meta_data)


def _get_fake_graph_dataset(batch_size, eval_num_batches, hps):
  """Yields graph batches with label 1 if graph has 4 or 6 nodes and 0 otherwise."""

  def _get_batch(n_nodes_list, batch_size):
    # Hardcode batch_size to be 2 for simplicity
    del batch_size
    graphs_list = []
    labels_list = []
    weights_list = []
    for n_nodes in n_nodes_list:
      n_edges = n_nodes**2
      graph = jraph.get_fully_connected_graph(
          n_nodes, 1,
          np.ones((n_nodes, *hps.input_node_shape)))
      graph = graph._replace(
          edges=np.ones((n_edges, *hps.input_edge_shape)))
      labels = np.ones(hps.output_shape) * (1 if n_nodes in [4, 6] else 0)
      weights = np.ones(*hps.output_shape)
      graphs_list.append(graph)
      labels_list.append(labels)
      weights_list.append(weights)
    return {
        'inputs': jraph.batch(graphs_list),
        'targets': np.stack(labels_list),
        'weights': np.stack(weights_list),
    }

  # Ensure each batch has one positive and one negative example for
  # average precision to not be NaN.
  def train_iterator_fn():
    for ns in itertools.cycle([[3, 6], [4, 5]]):
      yield _get_batch(ns, batch_size)

  def eval_train_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for ns in itertools.islice(itertools.cycle([[3, 6], [4, 5]]), num_batches):
      yield _get_batch(ns, batch_size)

  def valid_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for ns in itertools.islice(itertools.cycle([[3, 6], [4, 5]]), num_batches):
      yield _get_batch(ns, batch_size)

  def test_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for ns in itertools.islice(itertools.cycle([[3, 6], [4, 5]]), num_batches):
      yield _get_batch(ns, batch_size)

  return (Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                  test_epoch), {
                      'apply_one_hot_in_loss': False,
                  })


def _get_fake_dlrm_dataset(batch_size, eval_num_batches, hps):
  """Yields a single text batch repeatedly for train and test."""
  cat_features = []
  for vocab_size in hps.vocab_sizes:
    cat_features.append(
        np.random.randint(low=0, high=vocab_size, size=(batch_size, 1)))
  cat_features = np.concatenate(cat_features, 1)
  int_features = np.random.normal(size=(batch_size, hps.num_dense_features))
  inputs = np.concatenate((int_features, cat_features), 1)
  targets = np.random.randint(low=0, high=2, size=(batch_size, 1))
  batch = {
      'inputs': inputs,
      'targets': targets,
      'weights': jnp.ones(targets.shape),
  }

  def train_iterator_fn():
    while True:
      yield batch

  def eval_train_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def valid_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def test_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  meta_data = {
      'apply_one_hot_in_loss': False,
  }
  return (Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                  test_epoch), meta_data)


class TrainerTest(parameterized.TestCase):
  """Tests training for 2 epochs on MNIST."""

  def setUp(self):
    super(TrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # To not delete a directory in which a file might be created:
    checkpoint.wait_for_checkpoint_save()
    shutil.rmtree(self.test_dir)
    super(TrainerTest, self).tearDown()

  def test_initialize_rescale(self):
    """Test rescaling a single layer of a model."""
    input_shape = (28, 28, 1)
    output_shape = (10,)
    model_str = 'fully_connected'
    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    hps = copy.copy(model_hps)
    hps.update({'output_shape': output_shape})
    rng = jax.random.PRNGKey(0)
    model = model_cls(hps, {}, loss_name, metrics_name)
    initializer = initializers.get_initializer('noop')

    rng, init_rng = jax.random.split(rng)

    # First initialize with no rescale.
    params, _ = init_utils.initialize(
        model.flax_module,
        initializer,
        model.loss_fn,
        input_shape,
        output_shape,
        hps,
        init_rng,
        metrics_logger=None)

    utils.log_pytree_shape_and_statistics(params)
    # Now rescale a layer by 100.
    rescale_factor = 100
    hps.layer_rescale_factors = {
        '/Dense_1/kernel': rescale_factor,
    }

    rescaled_params, _ = init_utils.initialize(
        model.flax_module,
        initializer,
        model.loss_fn,
        input_shape,
        output_shape,
        hps,
        init_rng,
        metrics_logger=None)

    # Check the right variable is rescaled
    v1 = params['Dense_1']['kernel']
    v2 = rescaled_params['Dense_1']['kernel']
    diff = np.linalg.norm(v1.reshape(-1) * rescale_factor - v2.reshape(-1))
    self.assertAlmostEqual(diff, 0.0)

    # Check that other variables are the same
    v1 = params['Dense_2']['kernel']
    v2 = rescaled_params['Dense_2']['kernel']
    diff = np.linalg.norm(v1.reshape(-1) - v2.reshape(-1))
    self.assertAlmostEqual(diff, 0.0)

  def test_classifaction_model_evaluate(self):
    """Test trainer evaluate end to end with classification model metrics."""
    # Define a fake model that always outputs the same logits.
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, 1))

    class FakeModel(nn.Module):

      @nn.compact
      def __call__(self, x, train):
        # Make a single linear layer with the identity as the init.
        identity_fn = lambda *_: np.eye(4)
        x = nn.Dense(features=4, use_bias=False, kernel_init=identity_fn)(
            fake_batch_logits)
        return x

    key = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(key, num=2)
    fake_flax_module = FakeModel()
    model_init_fn = jax.jit(fake_flax_module.init)
    init_dict = model_init_fn(
        rngs={'params': params_rng, 'dropout': dropout_rng},
        x=None,
        train=False)
    params = jax_utils.replicate(init_dict['params'])
    batch_stats = init_dict.get('batch_stats', {})

    # 4 evaluation batches of size 4.
    weights = np.ones((4))
    fake_batches = [
        {
            'inputs': None,
            'targets': np.array([3, 2, 1, 0]),
            'weights': weights
        },
        {
            'inputs': None,
            'targets': np.array([0, 3, 2, 0]),
            'weights': weights
        },
        {
            'inputs': None,
            'targets': np.array([0, 0, 0, 0]),
            'weights': weights
        },
        {
            'inputs': None,
            'targets': np.array([1, 1, 1, 1]),
            'weights': weights
        },
    ]

    def fake_batches_gen():
      for batch in fake_batches:
        yield batch

    # pylint: disable=protected-access
    eval_fn = functools.partial(
        base_model._evaluate_batch,
        flax_module=fake_flax_module,
        metrics_bundle=metrics.get_metrics('classification_metrics'),
        apply_one_hot_in_loss=True)
    evaluate_batch_pmapped = jax.pmap(eval_fn, axis_name='batch')
    # pylint: enable=protected-access
    evaluated_metrics = trainer.evaluate(params,
                                         batch_stats,
                                         fake_batches_gen(),
                                         evaluate_batch_pmapped)

    def batch_ce_loss(logits, targets):
      one_hot_targets = np.eye(4)[targets]
      loss = -np.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)
      return loss

    expected_error_rate = 14.0/16.0  # FakeModel always predicts class 2.
    expected_ce_loss = np.mean(
        [batch_ce_loss(fake_batch_logits, b['targets']) for b in fake_batches])

    self.assertEqual(expected_error_rate, evaluated_metrics['error_rate'])
    self.assertAlmostEqual(
        expected_ce_loss, evaluated_metrics['ce_loss'], places=4)
    self.assertEqual(16, evaluated_metrics['num_examples'])

  def test_graph_model_trainer(self):
    """Tests that graph model training decreases loss."""
    rng = jax.random.PRNGKey(1337)
    model_str = 'gnn'
    model_cls = models.get_model(model_str)
    hps = models.get_model_hparams(model_str)
    hps.update({
        'batch_size': 2,
        'input_edge_shape': (7,),
        'input_node_shape': (3,),
        'input_shape': (7, 3),
        'output_shape': (5,),
        'model_dtype': 'float32',
        'train_size': 15,
        'valid_size': 10,
        'test_size': 10,
        'num_message_passing_steps': 1,
        'normalizer': 'none',
        'dropout_rate': 0.0,
        'lr_hparams': {
            'base_lr': 0.001,
            'schedule': 'constant'
        },
    })
    eval_num_batches = 5
    eval_batch_size = hps.batch_size
    loss_name = 'sigmoid_binary_cross_entropy'
    metrics_name = 'binary_classification_metrics'
    dataset, dataset_meta_data = _get_fake_graph_dataset(
        batch_size=hps.batch_size, eval_num_batches=eval_num_batches, hps=hps)
    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)
    initializer = initializers.get_initializer('noop')

    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    _ = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=10,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            # Note that for some reason, moving from the deprecated to linen
            # Flax model API made training less stable so we need to eval more
            # frequently in order to get a `train_loss[0]` that is earlier in
            # training.
            eval_frequency=2,
            checkpoint_steps=[],
            metrics_logger=metrics_logger,
            init_logger=init_logger))

    with tf.io.gfile.GFile(os.path.join(self.test_dir,
                                        'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_loss = df['train/ce_loss'].values
      self.assertLess(train_loss[-1], train_loss[0])

  def test_dlrm_model_trainer(self):
    """Tests that dlrm model training decreases loss."""
    rng = jax.random.PRNGKey(1337)
    model_str = 'dlrm'
    dataset_str = 'criteo1tb'
    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)
    dataset_hps = datasets.get_dataset_hparams(dataset_str)
    dataset_hps.update({
        'batch_size': model_hps.batch_size,
        'num_dense_features': model_hps.num_dense_features,
        'vocab_sizes': model_hps.vocab_sizes,
    })
    eval_num_batches = 5
    eval_batch_size = dataset_hps.batch_size
    loss_name = 'sigmoid_binary_cross_entropy'
    metrics_name = 'binary_classification_metrics'
    dataset, dataset_meta_data = _get_fake_dlrm_dataset(
        dataset_hps.batch_size, eval_num_batches, dataset_hps)
    hps = copy.copy(model_hps)
    hps.update({
        'train_size': 15,
        'valid_size': 10,
        'test_size': 10,
        'input_shape':
            (model_hps.num_dense_features + len(model_hps.vocab_sizes),),
        'output_shape': (1,),
        'l2_decay_factor': 1e-4,
        'l2_decay_rank_threshold': 2,
    })
    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)
    initializer = initializers.get_initializer('noop')

    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    _ = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=10,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=2,
            checkpoint_steps=[],
            metrics_logger=metrics_logger,
            init_logger=init_logger))

    with tf.io.gfile.GFile(os.path.join(self.test_dir,
                                        'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_loss = df['train/ce_loss'].values
      self.assertLess(train_loss[-1], train_loss[0])

  def test_text_model_trainer(self):
    """Test training of a small transformer model on fake data."""
    rng = jax.random.PRNGKey(42)

    # Set the numpy seed to make the fake data deterministc. mocking.mock_data
    # ultimately calls numpy.random.
    np.random.seed(0)

    model_cls = models.get_model('transformer')
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': _TEXT_BATCH_SIZE,
        'emb_dim': 32,
        'num_heads': 2,
        'num_layers': 3,
        'qkv_dim': 32,
        'mlp_dim': 64,
        'max_target_length': 64,
        'max_eval_target_length': 64,
        'input_shape': (64,),
        'output_shape': (_VOCAB_SIZE,),
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'layer_rescale_factors': {},
        'optimizer': 'momentum',
        'normalizer': 'layer_norm',
        'opt_hparams': {
            'momentum': 0.9,
        },
        'lr_hparams': {
            'base_lr': 0.005,
            'schedule': 'constant'
        },
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'l2_decay_rank_threshold': 2,
        'train_size': _TEXT_TRAIN_SIZE,
        'gradient_clipping': 0.0,
        'model_dtype': 'float32',
        'decode': False,
    })
    initializer = initializers.get_initializer('noop')
    eval_num_batches = 5
    dataset, dataset_meta_data = _get_fake_text_dataset(
        batch_size=hps.batch_size, eval_num_batches=eval_num_batches)
    eval_batch_size = hps.batch_size

    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)

    eval_every = 10
    checkpoint_steps = []
    num_train_steps = _TEXT_TRAIN_SIZE // _TEXT_BATCH_SIZE * 3

    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    _ = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger))

    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_err = df['train/error_rate'].values[-1]
      # Note that upgrading to Linen made this fail at 0.6.
      self.assertLess(train_err, 0.7)

    self.assertEqual(set(df.columns.values), set(get_column_names()))
    prev_train_err = train_err

    # Test reload from the checkpoint by increasing num_train_steps.
    num_train_steps_reload = _TEXT_TRAIN_SIZE // _TEXT_BATCH_SIZE * 6
    _ = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps_reload,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger))
    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_err = df['train/error_rate'].values[-1]
      train_loss = df['train/ce_loss'].values[-1]
      # Note that upgrading to Linen made this fail at 0.45.
      self.assertLess(train_err, 0.67)
      self.assertLess(train_err, prev_train_err)
      # Note that upgrading to Linen made this fail at 0.9.
      self.assertLess(train_loss, 1.35)

      self.assertEqual(
          df['valid/num_examples'].values[-1],
          eval_num_batches * eval_batch_size * _MAX_LEN)
      # Check that the correct learning rate was saved in the measurements file.
      final_step = df['global_step'].values[-1]
      self.assertEqual(num_train_steps_reload, final_step)

    self.assertEqual(set(df.columns.values), set(get_column_names()))

  def test_trainer(self):
    """Test training for two epochs on MNIST with a small model."""
    rng = jax.random.PRNGKey(0)

    # Set the numpy seed to make the fake data deterministc. mocking.mock_data
    # ultimately calls numpy.random.
    np.random.seed(0)

    model_name = 'fully_connected'
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    initializer_name = 'noop'
    dataset_name = 'mnist'
    model_cls = models.get_model(model_name)
    initializer = initializers.get_initializer(initializer_name)
    dataset_builder = datasets.get_dataset(dataset_name)
    hparam_overrides = {
        'lr_hparams': {
            'base_lr': 0.1,
            'schedule': 'cosine'
        },
        'batch_size': 8,
        'train_size': 160,
        'valid_size': 96,
        'test_size': 80,
    }
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides)

    eval_batch_size = 16
    num_examples = 256

    def as_dataset(self, *args, **kwargs):
      del args
      del kwargs

      # pylint: disable=g-long-lambda,g-complex-comprehension
      return tf.data.Dataset.from_generator(
          lambda: ({
              'image': np.ones(shape=(28, 28, 1), dtype=np.uint8),
              'label': 9,
          } for i in range(num_examples)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )

    # This will override the tfds.load(mnist) call to return 100 fake samples.
    with tfds.testing.mock_data(
        as_dataset_fn=as_dataset, num_examples=num_examples):
      dataset = dataset_builder(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=hps.batch_size,
          eval_batch_size=eval_batch_size,
          hps=hps)

    model = model_cls(hps, datasets.get_dataset_meta_data(dataset_name),
                      loss_name, metrics_name)

    num_train_steps = 40
    eval_num_batches = 5
    eval_every = 10
    checkpoint_steps = [1, 3, 15]
    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    epoch_reports = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger))

    # check that the additional checkpoints are saved.
    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    saved_steps = []
    for f in tf.io.gfile.listdir(checkpoint_dir):
      if f[:5] == 'ckpt_':
        saved_steps.append(int(f[5:]))

    self.assertEqual(set(saved_steps), set(checkpoint_steps))

    self.assertLen(epoch_reports, num_train_steps / eval_every)
    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_err = df['train/error_rate'].values[-1]
      self.assertEqual(df['preemption_count'].values[-1], 0)
      self.assertLess(train_err, 0.9)

    self.assertEqual(set(df.columns.values), set(get_column_names()))

    model = model_cls(hps, {'apply_one_hot_in_loss': False}, loss_name,
                      metrics_name)

    # Test reload from the checkpoint by increasing num_train_steps.
    num_train_steps_reload = 100
    epoch_reports = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps_reload,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger))
    self.assertLen(
        epoch_reports, (num_train_steps_reload - num_train_steps) / eval_every)
    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_err = df['train/error_rate'].values[-1]
      train_loss = df['train/ce_loss'].values[-1]
      self.assertLess(train_err, 0.35)
      self.assertLess(train_loss, 0.1)

      self.assertEqual(df['valid/num_examples'].values[-1],
                       eval_num_batches * eval_batch_size)
      self.assertEqual(df['preemption_count'].values[-1], 1)
      # Check that the correct learning rate was saved in the measurements file.
      final_learning_rate = df['learning_rate'].values[-1]
      final_step = df['global_step'].values[-1]
      self.assertEqual(num_train_steps_reload, final_step)

      # final_step will be one larger than the last step used to calculate the
      # lr_decay, hense we plug in (final_step - 1) to the decay formula.
      decay_factor = (1 + jnp.cos(
          (final_step - 1) / num_train_steps_reload * jnp.pi)) * 0.5
      self.assertEqual(float(final_learning_rate),
                       hps.lr_hparams['base_lr'] * decay_factor)

    self.assertEqual(set(df.columns.values), set(get_column_names()))

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_classification',
          metrics_name='classification_metrics',
          logits=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [1, 0]]),
          weights=np.array([1, 1, 0, 0]),
          test_metric_names=['error_rate', 'num_examples'],
          test_metric_vals=[0.5, 2]),
      dict(
          testcase_name='fractional_weights',
          metrics_name='classification_metrics',
          logits=np.array([[0.1, 0.9], [0.8, 0.2]]),
          targets=np.array([[1, 0], [1, 0]]),
          weights=np.array([0.3, 0.7]),
          test_metric_names=['error_rate', 'num_examples'],
          test_metric_vals=[0.3, 1]),
      dict(
          testcase_name='binary_classification_basic',
          metrics_name='binary_classification_metrics',
          logits=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
                            [1., 1.]]),
          test_metric_names=['ce_loss'],
          test_metric_vals=[0.724077]),
      dict(
          testcase_name='binary_classification_no_weights',
          metrics_name='binary_classification_metrics',
          logits=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=None,
          test_metric_names=['ce_loss'],
          test_metric_vals=[1.448154]),
      dict(
          testcase_name='binary_classification_zero_weights',
          metrics_name='binary_classification_metrics',
          logits=np.array([[100, 0.5], [100, 0.5], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=np.array([[0., 1.], [0., 1.], [1., 1.], [1., 1.], [1., 1.],
                            [1., 1.]]),
          test_metric_names=['ce_loss'],
          test_metric_vals=[0.724077]),
      dict(
          testcase_name='binary_classification_1d_weights',
          metrics_name='binary_classification_metrics',
          logits=np.array([[100, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 100],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=np.array([0., 1., 1., 0., 1., 1.,]),
          test_metric_names=['ce_loss'],
          test_metric_vals=[1.448154]),
      dict(
          testcase_name='binary_autoencoder_2d_weights',
          metrics_name='binary_autoencoder_metrics',
          logits=np.array([[0.5, 100], [0.5, 100], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=np.array([[1., 0.], [1., 0.], [1., 1.], [1., 1.], [1., 1.],
                            [1., 1.]]),
          test_metric_names=['sigmoid_mean_squared_error'],
          test_metric_vals=[0.26499629]),
      dict(
          testcase_name='binary_autoencoder_1d_weights',
          metrics_name='binary_autoencoder_metrics',
          logits=np.array([[0.5, 100], [0.5, 0.5], [0.5, 100], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=np.array([0., 1., 0., 1., 1., 1.,]),
          test_metric_names=['sigmoid_mean_squared_error'],
          test_metric_vals=[0.52999258]),
      dict(
          testcase_name='binary_autoencoder_no_weights',
          metrics_name='binary_autoencoder_metrics',
          logits=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5], [0.5, 0.5]]),
          targets=np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]),
          weights=None,
          test_metric_names=['sigmoid_mean_squared_error'],
          test_metric_vals=[0.5299926]),
  )
  def test_evaluate(self, metrics_name, logits, targets, weights,
                    test_metric_names, test_metric_vals):
    """Test metrics merging and evaluation including zero weights."""

    def mock_evaluate_batch(params, batch_stats, batch):
      """Always returns ones."""
      del params, batch_stats
      metrics_bundle = metrics.get_metrics(metrics_name)

      return metrics_bundle.gather_from_model_output(
          logits=batch.get('logits'),
          targets=batch.get('targets'),
          weights=batch.get('weights'))

    logits = np.split(logits, 2)
    targets = np.split(targets, 2)
    weights = np.split(weights, 2) if weights is not None else [None, None]

    # pylint: disable=g-complex-comprehension
    batch_iter = [{
        'logits': ls,
        'targets': ts,
        'weights': ws
    } for ls, ts, ws in zip(logits, targets, weights)]

    result = trainer.evaluate(
        params=None,
        batch_stats=None,
        batch_iter=batch_iter,
        evaluate_batch_pmapped=jax.pmap(mock_evaluate_batch, axis_name='batch'))

    for metric, val in zip(test_metric_names, test_metric_vals):
      self.assertAlmostEqual(result[metric], val, places=5)

  def test_early_stopping(self):
    """Test training early stopping on MNIST with a small model."""
    rng = jax.random.PRNGKey(0)

    # Set the numpy seed to make the fake data deterministc. mocking.mock_data
    # ultimately calls numpy.random.
    np.random.seed(0)

    model_name = 'fully_connected'
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    initializer_name = 'noop'
    dataset_name = 'mnist'
    model_cls = models.get_model(model_name)
    initializer = initializers.get_initializer(initializer_name)
    dataset_builder = datasets.get_dataset(dataset_name)
    hparam_overrides = {
        'lr_hparams': {
            'base_lr': 0.1,
            'schedule': 'cosine'
        },
        'batch_size': 8,
        'train_size': 160,
        'valid_size': 96,
        'test_size': 80,
    }
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides)

    eval_batch_size = 16
    num_examples = 256

    def as_dataset(self, *args, **kwargs):
      del args
      del kwargs

      # pylint: disable=g-long-lambda,g-complex-comprehension
      return tf.data.Dataset.from_generator(
          lambda: ({
              'image': np.ones(shape=(28, 28, 1), dtype=np.uint8),
              'label': 9,
          } for i in range(num_examples)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )

    # This will override the tfds.load(mnist) call to return 100 fake samples.
    with tfds.testing.mock_data(
        as_dataset_fn=as_dataset, num_examples=num_examples):
      dataset = dataset_builder(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=hps.batch_size,
          eval_batch_size=eval_batch_size,
          hps=hps)

    model = model_cls(hps, datasets.get_dataset_meta_data(dataset_name),
                      loss_name, metrics_name)

    num_train_steps = 40
    early_stopping_target_name = 'test/ce_loss'
    early_stopping_target_value = 0.005
    early_stopping_mode = 'less'
    eval_num_batches = 5
    eval_every = 10
    checkpoint_steps = [1, 3, 15]
    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    epoch_reports = list(
        trainer.train(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            early_stopping_target_name=early_stopping_target_name,
            early_stopping_target_value=early_stopping_target_value,
            early_stopping_mode=early_stopping_mode,
            metrics_logger=metrics_logger,
            init_logger=init_logger))
    self.assertLen(epoch_reports, 3)
    self.assertGreater(
        epoch_reports[-2][early_stopping_target_name],
        early_stopping_target_value)
    self.assertLess(
        epoch_reports[-1][early_stopping_target_name],
        early_stopping_target_value)


if __name__ == '__main__':
  absltest.main()
