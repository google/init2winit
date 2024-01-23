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

"""Unit tests for trainer.py.

"""

import os
import shutil
import tempfile

from absl.testing import absltest
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
from init2winit.hessian import hessian_eval
from init2winit.hessian import run_lanczos
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainer
import jax.random
from ml_collections import config_dict
import numpy as np
import tensorflow.compat.v1 as tf  # importing this is needed for tfds mocking.
import tensorflow_datasets as tfds


class RunLanczosTest(absltest.TestCase):
  """Tests run_lanczos.py."""

  def setUp(self):
    super(RunLanczosTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(RunLanczosTest, self).tearDown()

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
    input_pipeline_hps = config_dict.ConfigDict(dict(
        num_tf_data_prefetches=-1,
        num_device_prefetches=0,
        num_tf_data_map_parallel_calls=-1,
    ))
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides,
        input_pipeline_hps=input_pipeline_hps)
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
    test_num_batches = 0
    eval_every = 10
    checkpoint_steps = [10, 30, 40]
    metrics_logger, init_logger = None, None
    _ = list(
        trainer.Trainer(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_use_ema=False,
            eval_num_batches=eval_num_batches,
            test_num_batches=test_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger).train())

    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    rng = jax.random.PRNGKey(0)

    run_lanczos.eval_checkpoints(
        checkpoint_dir,
        hps,
        rng,
        eval_num_batches,
        test_num_batches,
        model_cls=model_cls,
        dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
        dataset_meta_data=datasets.get_dataset_meta_data(dataset_name),
        hessian_eval_config=hessian_eval.DEFAULT_EVAL_CONFIG,
    )

    # Load the saved file.
    hessian_dir = os.path.join(checkpoint_dir, 'hessian', 'training_metrics')
    pytree_list = checkpoint.load_pytree(hessian_dir)

    # Convert to a regular list (checkpointer will have converted the saved
    # list to a dict of keys '0', '1', ...
    pytree_list = [pytree_list[str(i)] for i in range(len(pytree_list))]
    # Test that the logged steps are correct.
    saved_steps = [row['step'] for row in pytree_list]
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
    input_pipeline_hps = config_dict.ConfigDict(dict(
        num_tf_data_prefetches=-1,
        num_device_prefetches=0,
        num_tf_data_map_parallel_calls=-1,
    ))
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides,
        input_pipeline_hps=input_pipeline_hps)
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
        trainer.Trainer(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_use_ema=False,
            eval_num_batches=eval_num_batches,
            test_num_batches=0,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            callback_configs=callback_configs,
            init_logger=init_logger).train())

    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    # Load the saved file.
    hessian_dir = os.path.join(checkpoint_dir, hessian_save_name,
                               'training_metrics')
    pytree_list = checkpoint.load_pytree(hessian_dir)
    # Test that the logged steps are correct.
    pytree_list = [pytree_list[str(i)] for i in range(len(pytree_list))]
    saved_steps = [int(row['step']) for row in pytree_list]
    self.assertEqual(saved_steps, checkpoint_steps + [41])

    # Check the dict keys.
    expected_keys = [
        'step', 'tridiag_hess', 'max_eig_hess', 'tridiag_hess_grad_overlap',
        'pos_neg_ratio', 'max_eig_hess_ratio', 'max_evec_decomp',
    ]
    self.assertEqual(set(pytree_list[0].keys()), set(expected_keys))


if __name__ == '__main__':
  absltest.main()
