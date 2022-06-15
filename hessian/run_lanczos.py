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

"""Used to evaluate a the Hessian eigenspectrum of a given checkpoint."""

import json
import os

from absl import logging
from flax import jax_utils
from init2winit import checkpoint
from init2winit.hessian import hessian_eval
from init2winit.init_lib import init_utils
from init2winit.init_lib import initializers
from init2winit.optimizer_lib import optimizers
from init2winit.trainer_lib import trainer
from init2winit.trainer_lib import trainer_utils
import utils as utils  # local file import
import jax
import optax

from tensorflow.io import gfile


def iterate_checkpoints(checkpoint_dir, min_global_step, max_global_step):
  """Iterates over all checkpoints in the interval [lb, ub)."""
  for checkpoint_path in gfile.glob(os.path.join(checkpoint_dir, 'ckpt_*')):

    step = int(checkpoint_path.split('_')[-1])
    if min_global_step is None or (min_global_step <= step < max_global_step):
      full_path = os.path.join(checkpoint_dir, checkpoint_path)
      yield full_path, step


# TODO(cheolmin): maybe we can pass a model rather than model_cls
def eval_checkpoints(
    checkpoint_dir,
    hps,
    rng,
    eval_num_batches,
    test_num_batches,
    model_cls,
    dataset_builder,
    dataset_meta_data,
    hessian_eval_config,
    min_global_step=None,
    max_global_step=None,
):
  """Evaluate the Hessian of the given checkpoints.

  Iterates over all checkpoints in the specified directory, loads the checkpoint
  then evaluates the Hessian on the given checkpoint. A list of dicts will be
  saved to cns at checkpoint_dir/hessian_eval_config['name'].

  Args:
    checkpoint_dir: Directory of checkpoints to load.
    hps: (tf.HParams) Model, initialization and training hparams.
    rng: (jax.random.PRNGKey) Rng seed used in model initialization and data
      shuffling.
    eval_num_batches: (int) The batch size used for evaluating on
      validation sets. Set to None to evaluate on the whole set.
    test_num_batches: (int) The batch size used for evaluating on
      test sets. Set to None to evaluate on the whole set.
    model_cls: One of the model classes (not an instance) defined in model_lib.
    dataset_builder: dataset builder returned by datasets.get_dataset.
    dataset_meta_data: dict of meta_data about the dataset.
    hessian_eval_config: a dict specifying the configuration of the Hessian
      eval.
    min_global_step: Lower bound on what steps to filter checkpoints. Set to
      None to evaluate all checkpoints in the directory.
    max_global_step: Upper bound on what steps to filter checkpoints.
  """
  rng, init_rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, data_rng = jax.random.split(rng)

  initializer = initializers.get_initializer('noop')

  loss_name = 'cross_entropy'
  metrics_name = 'classification_metrics'
  model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)

  # Maybe run the initializer.
  unreplicated_params, unreplicated_batch_stats = init_utils.initialize(
      model.flax_module,
      initializer, model.loss_fn,
      hps.input_shape,
      hps.output_shape, hps, init_rng,
      None)

  # Fold in a the unreplicated batch_stats and rng into the loss used by
  # hessian eval.
  def batch_loss(params, batch_rng):
    batch, rng = batch_rng
    return model.training_cost(
        params, batch, batch_stats=unreplicated_batch_stats, dropout_rng=rng)[0]
  batch_stats = jax_utils.replicate(unreplicated_batch_stats)

  if jax.process_index() == 0:
    utils.log_pytree_shape_and_statistics(unreplicated_params)
    logging.info('train_size: %d,', hps.train_size)
    logging.info(hps)
    # Save the hessian computation hps to the experiment directory
    exp_dir = os.path.join(checkpoint_dir, hessian_eval_config['name'])
    if not gfile.exists(exp_dir):
      gfile.mkdir(exp_dir)
    if min_global_step == 0:
      hparams_fname = os.path.join(exp_dir, 'hparams.json')
      with gfile.GFile(hparams_fname, 'w') as f:
        f.write(hps.to_json())
      config_fname = os.path.join(exp_dir, 'hconfig.json')
      with gfile.GFile(config_fname, 'w') as f:
        f.write(json.dumps(hessian_eval_config))

  optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(hps)
  unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)
  # Note that we do not use the learning rate.
  # The optimizer state is a list of all the optax transformation states, and
  # we inject the learning rate into all states that will accept it.
  for state in unreplicated_optimizer_state:
    if (isinstance(state, optax.InjectHyperparamsState) and
        'learning_rate' in state.hyperparams):
      state.hyperparams['learning_rate'] = jax_utils.replicate(1.0)
  optimizer_state = jax_utils.replicate(unreplicated_optimizer_state)
  params = jax_utils.replicate(unreplicated_params)
  data_rng = jax.random.fold_in(data_rng, 0)

  assert hps.batch_size % (jax.device_count()) == 0
  dataset = dataset_builder(
      data_rng,
      hps.batch_size,
      eval_batch_size=hps.batch_size,  # eval iterators not used.
      hps=hps,
  )

  # pmap functions for the training loop
  evaluate_batch_pmapped = jax.pmap(model.evaluate_batch, axis_name='batch')

  if jax.process_index() == 0:
    logging.info('Starting eval!')
    logging.info('Number of hosts: %d', jax.process_count())

  hessian_evaluator = hessian_eval.CurvatureEvaluator(
      params,
      hessian_eval_config,
      dataset=dataset,
      loss=batch_loss)
  if min_global_step is None:
    suffix = ''
  else:
    suffix = '{}_{}'.format(min_global_step, max_global_step)
  pytree_path = os.path.join(checkpoint_dir, hessian_eval_config['name'],
                             suffix)
  logger = utils.MetricLogger(pytree_path=pytree_path)
  for checkpoint_path, step in iterate_checkpoints(checkpoint_dir,
                                                   min_global_step,
                                                   max_global_step):
    unreplicated_checkpoint_state = dict(
        params=unreplicated_params,
        optimizer_state=unreplicated_optimizer_state,
        batch_stats=unreplicated_batch_stats,
        global_step=0,
        preemption_count=0,
        sum_train_cost=0.0)
    ckpt = checkpoint.load_checkpoint(
        checkpoint_path,
        target=unreplicated_checkpoint_state)
    results, _ = checkpoint.replicate_checkpoint(
        ckpt,
        pytree_keys=['params', 'optimizer_state', 'batch_stats'])
    params = results['params']
    optimizer_state = results['optimizer_state']
    batch_stats = results['batch_stats']
    # pylint: disable=protected-access
    batch_stats = trainer_utils.maybe_sync_batchnorm_stats(batch_stats)
    # pylint: enable=protected-access
    report, _ = trainer.eval_metrics(params, batch_stats, dataset,
                                     eval_num_batches, test_num_batches,
                                     eval_num_batches, evaluate_batch_pmapped)
    if jax.process_index() == 0:
      logging.info('Global Step: %d', step)
      logging.info(report)
    row = {}
    grads, updates = [], []
    hess_evecs, cov_evecs = [], []
    stats, hess_evecs, cov_evecs = hessian_evaluator.evaluate_spectrum(
        params, step)
    row.update(stats)
    if hessian_eval_config[
        'compute_stats'] or hessian_eval_config['compute_interps']:
      grads, updates = hessian_evaluator.compute_dirs(
          params, optimizer_state, optimizer_update_fn)
    row.update(hessian_evaluator.evaluate_stats(params, grads,
                                                updates, hess_evecs,
                                                cov_evecs, step))
    row.update(hessian_evaluator.compute_interpolations(params, grads,
                                                        updates, hess_evecs,
                                                        cov_evecs, step))
    if jax.process_index() == 0:
      logger.append_pytree(row)
