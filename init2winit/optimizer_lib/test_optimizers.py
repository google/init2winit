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

"""Tests for optimizers."""
import os
import shutil
import tempfile

from absl.testing import absltest
from init2winit import hyperparameters
from init2winit import utils
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainer
import jax
from jax import lax
from ml_collections import config_dict
import pandas
import tensorflow.compat.v1 as tf



class OptimizersTest(absltest.TestCase):
  """Tests for optimizers.py."""

  def setUp(self):
    super(OptimizersTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    self.trainer.wait_until_orbax_checkpointer_finished()
    shutil.rmtree(self.test_dir)
    super(OptimizersTest, self).tearDown()

  def test_shampoo_wrn(self):
    """Test distributed shampoo on fake dataset."""
    model_name = 'simple_cnn'
    model_cls = models.get_model(model_name)
    hparam_overrides = {
        'optimizer': 'distributed_shampoo',
        'batch_size': 1,
        'train_size': 10,
        'valid_size': 10,
        'input_shape': (32, 32, 3),
        'output_shape': (10,),
        'opt_hparams': {
            'block_size': 32,
            'beta1': 0.9,
            'beta2': 0.999,
            'diagonal_epsilon': 1e-10,
            'matrix_epsilon': 1e-6,
            'weight_decay': 0.0,
            'start_preconditioning_step': 5,
            'preconditioning_compute_steps': 1,
            'statistics_compute_steps': 1,
            'best_effort_shape_interpretation': True,
            'graft_type': distributed_shampoo.GraftingType.SGD,
            'nesterov': True,
            'exponent_override': 0,
            'batch_axis_name': 'batch',
            'num_devices_for_pjit': None,
            'shard_optimizer_states': False,
            'inverse_failure_threshold': 0.1,
            'clip_by_scaled_gradient_norm': None,
            'precision': lax.Precision.HIGHEST,
            'moving_average_for_momentum': False,
            'skip_preconditioning_dim_size_gt': 4096,
            'best_effort_memory_usage_reduction': False,
        },
    }
    input_pipeline_hps = config_dict.ConfigDict(dict(
        num_tf_data_prefetches=-1,
        num_device_prefetches=0,
        num_tf_data_map_parallel_calls=-1,
    ))
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name='noop',
        dataset_name='fake',
        hparam_file=None,
        hparam_overrides=hparam_overrides,
        input_pipeline_hps=input_pipeline_hps)
    initializer = initializers.get_initializer('noop')
    dataset_builder = datasets.get_dataset('fake')
    dataset = dataset_builder(
        shuffle_rng=jax.random.PRNGKey(0),
        batch_size=hps.batch_size,
        eval_batch_size=hps.batch_size,
        hps=hps)

    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    dataset_meta_data = datasets.get_dataset_meta_data('fake')
    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)

    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    self.trainer = trainer.Trainer(
        train_dir=self.test_dir,
        model=model,
        dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
        initializer=initializer,
        num_train_steps=1,
        hps=hps,
        rng=jax.random.PRNGKey(42),
        eval_batch_size=hps.batch_size,
        eval_use_ema=False,
        eval_num_batches=None,
        test_num_batches=0,
        eval_train_num_batches=None,
        eval_frequency=10,
        checkpoint_steps=[],
        metrics_logger=metrics_logger,
        init_logger=init_logger,
    )
    _ = list(self.trainer.train())

    with tf.io.gfile.GFile(os.path.join(self.test_dir,
                                        'measurements.csv')) as f:
      df = pandas.read_csv(f)
      valid_ce_loss = df['valid/ce_loss'].values[-1]
      self.assertLess(valid_ce_loss, 1e-3)


if __name__ == '__main__':
  absltest.main()
