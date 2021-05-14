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

"""Unit tests for trainer.py."""

import copy
import functools
import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from flax import nn
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit import trainer
from init2winit import utils
from init2winit.dataset_lib import datasets
from init2winit.dataset_lib.small_image_datasets import Dataset
from init2winit.init_lib import initializers
from init2winit.model_lib import base_model
from init2winit.model_lib import metrics
from init2winit.model_lib import models
import jax.numpy as jnp
import jax.random
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
      'train/denominator',
      'valid/denominator',
      'test/denominator',
      'train/ce_loss',
      'valid/ce_loss',
      'test/ce_loss',
      'global_step',
      'learning_rate',
      'epoch',
      'eval_time',
      'steps_per_sec',
      'preemption_count',
      'train_cost',
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


class TrainerTest(absltest.TestCase):
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
    flax_module, _ = trainer.initialize(
        model.flax_module_def,
        initializer,
        model.loss_fn,
        input_shape,
        output_shape,
        hps,
        init_rng,
        metrics_logger=None)

    utils.log_pytree_shape_and_statistics(flax_module.params)
    # Now rescale a layer by 100.
    rescale_factor = 100
    hps.layer_rescale_factors = {
        '/Dense_1/kernel': rescale_factor,
    }

    rescaled_module, _ = trainer.initialize(
        model.flax_module_def,
        initializer,
        model.loss_fn,
        input_shape,
        output_shape,
        hps,
        init_rng,
        metrics_logger=None)

    # Check the right variable is rescaled
    v1 = flax_module.params['Dense_1']['kernel']
    v2 = rescaled_module.params['Dense_1']['kernel']
    diff = np.linalg.norm(v1.reshape(-1) * rescale_factor - v2.reshape(-1))
    self.assertAlmostEqual(diff, 0.0)

    # Check that other variables are the same
    v1 = flax_module.params['Dense_2']['kernel']
    v2 = rescaled_module.params['Dense_2']['kernel']
    diff = np.linalg.norm(v1.reshape(-1) - v2.reshape(-1))
    self.assertAlmostEqual(diff, 0.0)

  def test_classifaction_model_evaluate(self):
    """Test trainer evaluate end to end with classification model metrics."""
    # Define a fake model that always outputs the same logits.
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, 1))

    class FakeModel(nn.Module):

      def apply(self, x, train=False):
        return fake_batch_logits

    key = jax.random.PRNGKey(0)
    with nn.stateful() as batch_stats:
      _, fake_flax_module = FakeModel.create_by_shape(key,
                                                      [((10, 10), jnp.float32)])

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
        metrics_bundle=metrics.get_metrics('classification_metrics'),
        apply_one_hot_in_loss=True)
    evaluate_batch_pmapped = jax.pmap(eval_fn, axis_name='batch')
    # pylint: enable=protected-access
    evaluated_metrics = trainer.evaluate(fake_flax_module, batch_stats,
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
    self.assertEqual(16, evaluated_metrics['denominator'])

  def test_text_model_trainer(self):
    """Test training of a small transformer model on fake data."""
    rng = jax.random.PRNGKey(0)

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
            'initial_value': 0.005,
            'schedule': 'constant'
        },
        # Training HParams.
        'l2_decay_factor': 1e-4,
        'l2_decay_rank_threshold': 2,
        'train_size': _TEXT_TRAIN_SIZE,
        'gradient_clipping': 0.0,
        'model_dtype': 'float32',
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

    metrics_logger, init_logger = trainer.set_up_loggers(self.test_dir)
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
      self.assertLess(train_err, 0.6)

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
      self.assertLess(train_err, 0.45)
      self.assertLess(train_err, prev_train_err)
      self.assertLess(train_loss, 0.9)

      self.assertEqual(
          df['valid/denominator'].values[-1],
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
            'initial_value': 0.1,
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
    metrics_logger, init_logger = trainer.set_up_loggers(self.test_dir)
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

      self.assertEqual(df['valid/denominator'].values[-1],
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
                       hps.lr_hparams['initial_value'] * decay_factor)

    self.assertEqual(set(df.columns.values), set(get_column_names()))


if __name__ == '__main__':
  absltest.main()
