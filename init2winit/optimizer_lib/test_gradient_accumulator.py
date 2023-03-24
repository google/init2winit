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

r"""Tests for gradient_accumulator.py.

"""
import copy
import functools
import itertools
import os
import shutil
import tempfile

from absl.testing import absltest
from flax import core
from init2winit import checkpoint
from init2winit import trainer
from init2winit.dataset_lib import datasets
from init2winit.dataset_lib.small_image_datasets import Dataset
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.optimizer_lib import gradient_accumulator
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import optax
import pandas
import tensorflow.compat.v1 as tf


def _init_model(model_cls, hps):
  """Initialize the Flax model."""
  loss_name = 'cross_entropy'
  metrics_name = 'classification_metrics'
  key = jax.random.PRNGKey(0)
  dataset_metadata = {
      'apply_one_hot_in_loss': False,
  }
  model = model_cls(hps, dataset_metadata, loss_name, metrics_name)
  params_rng, dropout_rng = jax.random.split(key, num=2)
  model_init_fn = jax.jit(
      functools.partial(model.flax_module.init, train=False))
  init_dict = model_init_fn(
      rngs={'params': params_rng, 'dropout': dropout_rng},
      x=np.zeros((2, *hps.input_shape)))
  params = init_dict['params']
  batch_stats = init_dict.get('batch_stats', {})
  return params, batch_stats, model.training_cost


def _optimize(num_steps,
              params,
              batch_stats,
              training_cost,
              train_iter,
              opt_init,
              opt_update):
  """Update the Flax model for num_steps steps."""
  opt_state = opt_init(params)

  def opt_cost(params, batch_stats, batch):
    return training_cost(
        params,
        batch=batch,
        batch_stats=batch_stats,
        dropout_rng=jax.random.PRNGKey(2))
  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  for _ in range(num_steps):
    data_batch = next(train_iter)
    (_, updated_vars), grad = grad_fn(params, batch_stats, data_batch)
    batch_stats = updated_vars.get('batch_stats', {})
    model_updates, opt_state = opt_update(grad, opt_state, params=params)
    params = optax.apply_updates(params, model_updates)
  return params, batch_stats


def _get_fake_text_dataset(batch_size, eval_num_batches):
  """Yields a single text batch repeatedly for train and test."""
  inputs = jnp.array(
      np.random.randint(low=0, high=4, size=(batch_size, 32)))
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


class GradientAccumulatorTest(absltest.TestCase):
  """Tests for gradient_accumulator.py."""

  def setUp(self):
    super(GradientAccumulatorTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # To not delete a directory in which a file might be created:
    checkpoint.wait_for_checkpoint_save()
    shutil.rmtree(self.test_dir)
    super(GradientAccumulatorTest, self).tearDown()

  def test_virtual_batch_size_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Gradient accumulation does not currently support using '):
      gradient_accumulator.accumulate_gradients(
          per_step_batch_size=32,
          total_batch_size=96,
          virtual_batch_size=48,
          base_opt_init_fn=None,
          base_opt_update_fn=None)

  def test_accumulation(self):
    """Test simple gradient accumulation."""
    num_steps = 3
    per_step_batch_size = 16
    total_batch_size = 48
    virtual_batch_size = 8
    model_str = 'wide_resnet'  # Pick a model with batch norm.
    model_cls = models.get_model(model_str)
    model_hps = models.get_model_hparams(model_str)
    dataset_name = 'cifar10'
    dataset_builder = datasets.get_dataset(dataset_name)
    hps = copy.copy(model_hps)
    hps.update(datasets.get_dataset_hparams(dataset_name))

    # Compute updates using gradient accumulation.
    hps.update({
        'batch_size': per_step_batch_size,
        'virtual_batch_size': virtual_batch_size,
        'normalizer': 'virtual_batch_norm',
        'total_accumulated_batch_size': total_batch_size,
    })
    grad_acc_params, grad_acc_batch_stats, grad_acc_training_cost = _init_model(
        model_cls, hps)
    total_dataset = dataset_builder(
        shuffle_rng=jax.random.PRNGKey(1),
        batch_size=total_batch_size,
        eval_batch_size=10,
        hps=hps)
    # Ensure we see the same exact batches.
    train_iter = total_dataset.train_iterator_fn()
    train_iter = itertools.islice(train_iter, 0, num_steps)
    train_iter = itertools.cycle(train_iter)
    def grad_acc_train_iter():
      for _ in range(num_steps):
        total_batch = next(train_iter)
        # Split each total batch into sub batches.
        num_sub_batches = total_batch_size // per_step_batch_size
        start_index = 0
        end_index = int(total_batch_size / num_sub_batches)
        for bi in range(num_sub_batches):
          yield jax.tree_map(lambda x: x[start_index:end_index], total_batch)  # pylint: disable=cell-var-from-loop
          start_index = end_index
          end_index = int(total_batch_size * (bi + 2) / num_sub_batches)

    lrs = jnp.array([1.0, 0.1, 1e-2])
    sgd_opt_init, sgd_opt_update = optax.sgd(
        learning_rate=lambda t: lrs.at[t].get())
    opt_init, opt_update = gradient_accumulator.accumulate_gradients(
        per_step_batch_size=per_step_batch_size,
        total_batch_size=total_batch_size,
        virtual_batch_size=virtual_batch_size,
        base_opt_init_fn=sgd_opt_init,
        base_opt_update_fn=sgd_opt_update)
    grad_acc_params, grad_acc_batch_stats = _optimize(
        # Run for 3x the number of steps to see the same number of examples.
        num_steps=3 * num_steps,
        params=grad_acc_params,
        batch_stats=grad_acc_batch_stats,
        training_cost=grad_acc_training_cost,
        train_iter=grad_acc_train_iter(),
        opt_init=opt_init,
        opt_update=opt_update)

    # Compute the same updates, but without gradient accumulation.
    hps.update({
        'batch_size': total_batch_size,
        'total_accumulated_batch_size': None,
    })
    params, batch_stats, training_cost = _init_model(model_cls, hps)
    params, batch_stats = _optimize(
        num_steps=num_steps,
        params=params,
        batch_stats=batch_stats,
        training_cost=training_cost,
        train_iter=train_iter,
        opt_init=sgd_opt_init,
        opt_update=sgd_opt_update)

    diffs_params = jax.tree_map(lambda a, b: jnp.mean(jnp.abs(a - b)),
                                grad_acc_params, params)

    def batch_stats_reduce(a, b):
      if len(a.shape) > 0:  # pylint: disable=g-explicit-length-test
        return jnp.mean(
            jnp.abs(jnp.mean(a, axis=0) - jnp.mean(b, axis=0)))
      # The gradient accumulator counters are scalars.
      return a - b

    diffs_batch_stats = jax.tree_map(batch_stats_reduce, grad_acc_batch_stats,
                                     batch_stats)
    # We sometimes get small floating point errors in the gradients, so we
    # cannot test for the values being exactly the same.
    acceptable_params_diff = 1e-4
    acceptable_batch_stats_diff = 5e-3

    def check_closeness(root_name, d, max_diff):
      not_close_dict = {}
      for name, dd in d.items():
        new_name = root_name + '/' + name if root_name else name
        if isinstance(dd, (dict, core.FrozenDict)):
          not_close_dict.update(check_closeness(new_name, dd, max_diff))
        else:
          if dd > max_diff:
            not_close_dict[new_name] = dd
      return not_close_dict

    not_close_params = check_closeness(
        '', diffs_params, acceptable_params_diff)
    self.assertEmpty(not_close_params)
    not_close_batch_stats = check_closeness(
        '', diffs_batch_stats, acceptable_batch_stats_diff)
    # Note that for the variance variables in the batch stats collection, they
    # sometimes can start to diverge slightly over time (with a higher number of
    # training steps), likely due to numerical issues.
    self.assertEmpty(not_close_batch_stats)

  def test_text_model(self):
    """Test gradient accumulator training of a small transformer."""
    rng = jax.random.PRNGKey(42)

    # Set the numpy seed to make the fake data deterministc. mocking.mock_data
    # ultimately calls numpy.random.
    np.random.seed(0)

    model_cls = models.get_model('transformer')
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    batch_size = 16
    train_size = 20 * batch_size
    hps = config_dict.ConfigDict({
        # Architecture Hparams.
        'batch_size': batch_size,
        'emb_dim': 32,
        'num_heads': 2,
        'num_layers': 3,
        'qkv_dim': 32,
        'mlp_dim': 64,
        'max_target_length': 64,
        'max_eval_target_length': 64,
        'input_shape': (64,),
        'output_shape': (4,),
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
        'train_size': train_size,
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
    num_train_steps = train_size // batch_size * 3

    metrics_logger, init_logger = trainer.set_up_loggers(self.test_dir)
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
            eval_num_batches=eval_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps,
            metrics_logger=metrics_logger,
            init_logger=init_logger).train())

    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')) as f:
      df = pandas.read_csv(f)
      train_err = df['train/error_rate'].values[-1]
      # Note that upgrading to Linen made this fail at 0.6.
      self.assertLess(train_err, 0.7)


if __name__ == '__main__':
  absltest.main()
