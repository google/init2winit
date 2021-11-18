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

"""Unit tests for trainer.py.

"""

import os
import shutil
import tempfile

from absl.testing import absltest
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit import trainer
from init2winit.dataset_lib import datasets
from init2winit.hessian import hessian_eval
from init2winit.hessian import run_lanczos
from init2winit.init_lib import initializers
from init2winit.model_lib import models
import jax.random
import numpy as np
import tensorflow.compat.v1 as tf  # importing this is needed for tfds mocking.
import tensorflow_datasets as tfds


class TrainerTest(absltest.TestCase):
  """Tests training for 2 epochs on MNIST."""

  def setUp(self):
    super(TrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(TrainerTest, self).tearDown()

  def test_run_lanczos(self):
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
    model = model_cls(hps, datasets.get_dataset_meta_data(dataset_name),
                      loss_name, metrics_name)

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

    num_train_steps = 41
    eval_num_batches = 5
    eval_every = 10
    checkpoint_steps = [10, 30, 40]
    metrics_logger, init_logger = None, None
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

    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    rng = jax.random.PRNGKey(0)

    run_lanczos.eval_checkpoints(
        checkpoint_dir,
        hps,
        rng,
        eval_num_batches,
        model_cls=model_cls,
        dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
        dataset_meta_data=datasets.get_dataset_meta_data(dataset_name),
        hessian_eval_config=hessian_eval.DEFAULT_EVAL_CONFIG,
    )

    # Load the saved file.
    hessian_file = os.path.join(checkpoint_dir, 'hessian')
    latest = checkpoint.load_latest_checkpoint(hessian_file)
    state_list = latest.pytree if latest else []

    # Test that the logged steps are correct.
    saved_steps = [row['step'] for row in state_list]
    self.assertEqual(saved_steps, checkpoint_steps)

  def test_hessian_callback(self):
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

    callback_configs = [hessian_eval.DEFAULT_EVAL_CONFIG.copy()]
    hessian_save_name = 'hessian2'
    callback_configs[0]['callback_name'] = 'hessian'
    callback_configs[0]['name'] = hessian_save_name
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides)
    model = model_cls(hps, datasets.get_dataset_meta_data(dataset_name),
                      loss_name, metrics_name)

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

    num_train_steps = 41
    eval_num_batches = 5
    eval_every = 10
    checkpoint_steps = [10, 20, 30, 40]
    metrics_logger, init_logger = None, None
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
            callback_configs=callback_configs,
            init_logger=init_logger))

    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    # Load the saved file.
    hessian_file = os.path.join(checkpoint_dir, hessian_save_name)
    latest = checkpoint.load_latest_checkpoint(hessian_file)
    state_list = latest.pytree if latest else []

    # Test that the logged steps are correct.
    saved_steps = [int(row['step']) for row in state_list]
    self.assertEqual(saved_steps, checkpoint_steps)

    # Check the dict keys.
    expected_keys = [
        'step', 'tridiag_hess', 'max_eig_hess', 'tridiag_hess_grad_overlap'
    ]
    self.assertEqual(set(state_list[0].keys()), set(expected_keys))


if __name__ == '__main__':
  absltest.main()
