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

"""Trainer for the init2winit project."""

import collections
import functools
import itertools
import json
import os
import struct
import time

from absl import flags
from absl import logging
from flax import jax_utils
from init2winit import callbacks
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit import schedules
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
from init2winit.optimizer_lib import optimizers
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow.io import gfile


FLAGS = flags.FLAGS

_GRAD_CLIP_EPS = 1e-6


def evaluate(
    params,
    batch_stats,
    batch_iter,
    evaluate_batch_pmapped):
  """Compute aggregated metrics on the given data iterator.

  WARNING: The caller is responsible for synchronizing the batch norm statistics
  before calling this function!

  Assumed API of evaluate_batch_pmapped:
  metrics = evaluate_batch_pmapped(params, batch_stats, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. The metric
  'denominator' must also be defined. evaluate will aggregate (by summing)
  all per example measurements and divide by the aggregated denominator.
  For each given metric we compute
    1/N sum_{b in batch_iter} metric(b).
  Where N will be the sum of 'denominator' over all batches. See
  classification_metrics.py for a definition of evaluate_batch.

  Args:
    params: a dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: a dict of non-trainable model state. Passed as
      {'batch_stats': batch_stats} into flax_module.apply().
    batch_iter: Generator which yields batches. Must support the API
      for b in batch_iter:
    evaluate_batch_pmapped: A function with API
       evaluate_batch_pmapped(params, batch_stats, batch). Returns a dictionary
       mapping keys to the summed metric across the sharded batch. The key
       'denominator' is required, as this indicates how many real samples were
       in the sharded batch.

  Returns:
    A dictionary of aggregated metrics. The keys will match the keys returned by
    evaluate_batch_pmapped.
  """
  # TODO(gilmer) Currently we only support metrics of the form 1/N sum f(x_i).
  # May need a more general framework to stuff like precision and recall.
  # np allows for the total_losses += syntax to work (array assignment).
  total_metrics = collections.defaultdict(float)
  for batch in batch_iter:
    batch = data_utils.shard(batch)
    computed_metrics = evaluate_batch_pmapped(
        params=params, batch_stats=batch_stats, batch=batch)
    for key in computed_metrics:
      # The shape of computed_metrics[key] is [n_local_devices]. However,
      # because evaluate_batch_pmapped has a psum, we have already summed
      # across the whole sharded batch, and what's returned is n_local_devices
      # copies of the same summed metric. So here we just grab the 0'th entry.
      total_metrics[key] += np.float32(computed_metrics[key][0])

  # For data splits with no data (e.g. Imagenet no test set) no values
  # will appear for that split.
  for key in total_metrics:
    # Convert back to numpy
    if np.isnan(total_metrics[key]):
      raise utils.TrainingDivergedError('NaN detected in {}'.format(key))
    if key != 'denominator':
      total_metrics[key] = total_metrics[key] / np.float32(
          total_metrics['denominator'])
  return total_metrics


def _inject_learning_rate(optimizer_state, lr):
  """Inject the given LR into any optimizer state that will accept it."""
  # The optimizer state should always be an InjectHyperparamsState, and we
  # inject the learning rate into all states that will accept it. We need to do
  # this to allow arbitrary (non-jittable) LR schedules.
  if isinstance(optimizer_state, optax.InjectHyperparamsState):
    if 'learning_rate' in optimizer_state.hyperparams:
      optimizer_state.hyperparams['learning_rate'] = lr
  else:
    raise ValueError(
        'Unsupported optimizer_state type given when trying to inject the '
        'learning rate:\n\n{}.'.format(optimizer_state))


def update(
    optimizer_state,
    params,
    batch_stats,
    batch,
    step,
    lr,
    rng,
    local_device_index,
    training_metrics_grabber,
    training_cost,
    grad_clip,
    optimizer_update_fn):
  """Single step of the training loop.

  Args:
    optimizer_state: the optax optimizer state.
    params: a dict of trainable model parameters. Passed into training_cost(...)
      which then passes into flax_module.apply() as {'params': params} as part
      of the variables dict.
    batch_stats: a dict of non-trainable model state. Passed into
      training_cost(...) which then passes into flax_module.apply() as
      {'batch_stats': batch_stats} as part of the variables dict.
    batch: the per-device batch of data to process.
    step: the current global step of this update. Used to fold in to `rng` to
      produce a unique per-device, per-step RNG.
    lr: the floating point learning rate for this step.
    rng: the RNG used for calling the model. `step` and `local_device_index`
      will be folded into this to produce a unique per-device, per-step RNG.
    local_device_index: an integer that is unique to this device amongst all
      devices on this host, usually in the range [0, jax.local_device_count()].
      It is folded in to `rng` to produce a unique per-device, per-step RNG.
    training_metrics_grabber: See the TrainingMetricsGrabber in utils.py
    training_cost: a function used to calculate the training objective that will
      be differentiated to generate updates. Takes
      (`params`, `batch`, `batch_stats`, `dropout_rng`) as inputs.
    grad_clip: Clip the l2 norm of the gradient at the specified value. For
      minibatches with gradient norm ||g||_2 > grad_clip, we rescale g to the
      value g / ||g||_2 * grad_clip. If None, then no clipping will be applied.
    optimizer_update_fn: the optimizer update function.

  Returns:
    A tuple of the new optimizer, the new batch stats, the scalar training cost,
    and the updated metrics_grabber.
  """
  # `jax.random.split` is very slow outside the train step, so instead we do a
  # `jax.random.fold_in` here.
  rng = jax.random.fold_in(rng, step)
  rng = jax.random.fold_in(rng, local_device_index)

  _inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost(
        params, batch=batch, batch_stats=batch_stats, dropout_rng=rng)

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  cost_value, grad = lax.pmean((cost_value, grad), axis_name='batch')

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  # TODO(znado): move to inside optax gradient clipping.
  if grad_clip:
    scaled_grad = jax.tree_map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
    grad = jax.lax.cond(grad_norm > grad_clip, lambda _: scaled_grad,
                        lambda _: grad, None)
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats)
  new_params = optax.apply_updates(params, model_updates)

  new_metrics_grabber = None
  if training_metrics_grabber:
    new_metrics_grabber = training_metrics_grabber.update(
        grad, params, new_params)

  return (new_optimizer_state, new_params, new_batch_stats, cost_value,
          new_metrics_grabber, grad_norm)


def _merge_and_apply_prefix(d1, d2, prefix):
  d1 = d1.copy()
  for key in d2:
    d1[prefix+key] = d2[key]
  return d1


@utils.timed
def eval_metrics(params, batch_stats, dataset, eval_num_batches,
                 eval_train_num_batches, evaluate_batch_pmapped):
  """Evaluates the given network on the train, validation, and test sets.

  WARNING: we assume that `batch_stats` has already been synchronized across
  devices before being passed to this function! See
  `_maybe_sync_batchnorm_stats`.

  The metric names will be of the form split/measurement for split in the set
  {train, valid, test} and measurement in the set {loss, error_rate}.

  Args:
    params: a dict of trainable model parameters. Passed as {'params': params}
      into flax_module.apply().
    batch_stats: a dict of non-trainable model state. Passed as
      {'batch_stats': batch_stats} into flax_module.apply().
    dataset: Dataset returned from datasets.get_dataset. train, validation, and
      test sets.
    eval_num_batches: (int) The batch size used for evaluating on validation,
      and test sets. Set to None to evaluate on the whole test set.
    eval_train_num_batches: (int) The batch size used for evaluating on train
      set. Set to None to evaluate on the whole training set.
    evaluate_batch_pmapped: Computes the metrics on a sharded batch.

  Returns:
    A dictionary of all computed metrics.
  """
  train_iter = dataset.eval_train_epoch(eval_train_num_batches)
  valid_iter = dataset.valid_epoch(eval_num_batches)
  test_iter = dataset.test_epoch(eval_num_batches)

  metrics = {}
  for split_iter, split_name in zip([train_iter, valid_iter, test_iter],
                                    ['train', 'valid', 'test']):
    split_metrics = evaluate(params, batch_stats, split_iter,
                             evaluate_batch_pmapped)
    metrics = _merge_and_apply_prefix(metrics, split_metrics,
                                      (split_name + '/'))
  return metrics


def initialize(flax_module,
               initializer,
               loss_fn,
               input_shape,
               output_shape,
               hps,
               rng,
               metrics_logger,
               fake_batch=None):
  """Run the given initializer.

  We initialize in 3 phases. First we run the default initializer that is
  specified by the model constructor. Next we apply any rescaling as specified
  by hps.layer_rescale_factors. Finally we run the black box initializer
  provided by the initializer arg (the default is noop).

  Args:
    flax_module: The Flax nn.Module.
    initializer: An initializer defined in init_lib.
    loss_fn: A loss function.
    input_shape: The input shape of a single data example.
    output_shape: The output shape of a single data example.
    hps: A dictionary specifying the model and initializer hparams.
    rng: An rng key to seed the initialization.
    metrics_logger: Used for black box initializers that have learning curves.
    fake_batch: Fake input used to intialize the network or None if using
      init_by_shape.

  Returns:
    A tuple (model, batch_stats), where model is the initialized
    flax.nn.Model and batch_stats is the collection used for batch norm.
  """
  model_dtype = utils.dtype_from_str(hps.model_dtype)
  # init_by_shape should either pass in a tuple or a list of tuples.
  # For example, for vision tasks typically input_shape is (image_shape)
  # For seq2seq tasks, shape can be a list of two tuples corresponding to
  # input_sequence_shape for encoder and output_sequence_shape for decoder.
  # TODO(gilmer,ankugarg): Support initializers for list of tuples.
  #
  # Note that this fake input batch will be optimized away because the init
  # function is jitted. However, this can still cause memory issues if it is
  # large because it is passed in as an XLA argument. Therefore we use a fake
  # batch size of 2 (we do not want to use 1 in case there is any np.squeeze
  # calls that would remove it), because we assume that there is no dependence
  # on the batch size with the model (batch norm reduces across a batch dim of
  # any size). This is similar to how the Flax examples initialize models:
  # https://github.com/google/flax/blob/44ee6f2f4130856d47159dc58981fb26ea2240f4/examples/imagenet/train.py#L70.
  if fake_batch:
    fake_input_batch = fake_batch
  elif isinstance(input_shape, list):  # Typical case for seq2seq models
    fake_input_batch = [
        np.zeros((2, *x), model_dtype) for x in input_shape
    ]
  else:  # Typical case for classification models
    fake_input_batch = [np.zeros((2, *input_shape), model_dtype)]
  params_rng, init_rng, dropout_rng = jax.random.split(rng, num=3)

  # By jitting the model init function, we initialize the model parameters
  # lazily without computing a full forward pass. For further documentation, see
  # https://flax.readthedocs.io/en/latest/flax.linen.html?highlight=jax.jit#flax.linen.Module.init.
  # We need to close over train=False here because otherwise the jitted init
  # function will convert the train Python bool to a jax boolean, which will
  # mess up Pythonic boolean statements like `not train` inside the model
  # construction.
  model_init_fn = jax.jit(functools.partial(flax_module.init, train=False))
  init_dict = model_init_fn(
      {'params': params_rng, 'dropout': dropout_rng},
      *fake_input_batch)
  # Trainable model parameters.
  params = init_dict['params']
  batch_stats = init_dict.get('batch_stats', {})

  if hps.get('layer_rescale_factors'):
    params = model_utils.rescale_layers(params, hps.layer_rescale_factors)
  # We don't pass batch_stats to the initializer, the initializer will just
  # run batch_norm in train mode and does not need to maintain the batch_stats.
  # TODO(gilmer): We hardcode here weighted_cross_entropy, but this will need
  # to change for other models. Maybe have meta_loss_inner as an initializer
  # hyper_param?
  # TODO(gilmer): instead of passing in weighted_xent, pass in the model and get
  # the loss from that.
  params = initializer(loss_fn, flax_module, params, hps, input_shape,
                       output_shape, init_rng, metrics_logger)

  return params, batch_stats


def save_checkpoint(train_dir,
                    pytree,
                    global_step,
                    preemption_count,
                    sum_train_cost,
                    max_to_keep=1,
                    use_deprecated_checkpointing=True):
  """Saves the pytree to train_dir."""
  checkpoint_name = 'ckpt_{}'.format(global_step)
  logging.info('Saving checkpoint to %s', checkpoint_name)
  unstructured_state = jax.device_get([x[0] for x in jax.tree_leaves(pytree)])
  state = checkpoint.CheckpointState(pytree=unstructured_state,
                                     global_step=global_step,
                                     preemption_count=preemption_count,
                                     sum_train_cost=sum_train_cost)
  checkpoint.save_checkpoint_background(
      train_dir,
      checkpoint_name,
      state,
      max_to_keep=max_to_keep,
      use_deprecated_checkpointing=use_deprecated_checkpointing)
  logging.info('Done saving checkpoint.')


# TODO(znado,gilmer,gdahl): make this take the unreplicated pytree to avoid
# having to store two replicated copies in memory.
def restore_checkpoint(
    latest,
    replicated_pytree,
    replicate=True,
    use_deprecated_checkpointing=True):
  """Restores from the provided checkpoint.

  Args:
    latest: A checkpoint.CheckpointState representing the state of the
      checkpoint we want to restore.
    replicated_pytree: The pytree with the structure we expect to restore to.
    replicate: If set, replicate the pytree across devices.
    use_deprecated_checkpointing: Whether to use deprecated checkpointing.

  Returns:
    Tuple of (pytree, extra_dict) where pytree is a JAX pytree holding the
    arrays that need to be replicated/unreplicated and extra_dict holds any
    additional python state. We expect extra_dict to have the keys of
    'global_step', 'preemption_count', 'sum_train_cost', but old checkpoints
    might be missing something.
  """
  logging.info('Loaded model parameters from latest checkpoint.')
  # Old checkpoints without 'sum_train_cost' can still be restored, but the
  # train() function will break. Evals and curvature stuff should be fine,
  # however.
  expected = ['global_step', 'preemption_count', 'sum_train_cost']
  if any(k not in latest.pystate for k in expected):
    logging.warn('Checkpoint pystate missing keys, obtained %s expected %s',
                 list(latest.pystate.keys()), expected)
  unstructured_pytree = latest.pytree
  if replicate:
    unstructured_pytree = jax_utils.replicate(unstructured_pytree)
  if use_deprecated_checkpointing:
    structure = jax.tree_util.tree_structure(replicated_pytree)
    pytree = jax.tree_unflatten(structure, unstructured_pytree)
  else:
    pytree = unstructured_pytree
  return pytree, latest.pystate


def _replicate_and_maybe_restore_latest_checkpoint(
    unreplicated_optimizer_state,
    unreplicated_params,
    unreplicated_batch_stats,
    unreplicated_training_metrics_grabber,
    train_dir,
    use_deprecated_checkpointing):
  """Restore from the latest checkpoint, if it exists."""
  unreplicated_checkpoint_state = checkpoint.CheckpointState(
      {
          'optimizer_state': unreplicated_optimizer_state,
          'params': unreplicated_params,
          'batch_stats': unreplicated_batch_stats,
          'training_metrics_grabber': unreplicated_training_metrics_grabber,
      },
      global_step=0,
      preemption_count=0,
      sum_train_cost=0.0)
  latest = checkpoint.load_latest_checkpoint(
      train_dir,
      target=unreplicated_checkpoint_state,
      recents_filename='latest',
      use_deprecated_checkpointing=use_deprecated_checkpointing)

  optimizer_state = jax_utils.replicate(unreplicated_optimizer_state)
  params = jax_utils.replicate(unreplicated_params)
  batch_stats = jax_utils.replicate(unreplicated_batch_stats)
  training_metrics_grabber = jax_utils.replicate(
      unreplicated_training_metrics_grabber)

  if latest is None:
    return optimizer_state, params, batch_stats, training_metrics_grabber, 0, 0.0, 0, False

  pytree_dict, extra_state = restore_checkpoint(
      latest,
      replicated_pytree={
          'optimizer_state': optimizer_state,
          'params': params,
          'batch_stats': batch_stats,
          'training_metrics_grabber': training_metrics_grabber,
      },
      use_deprecated_checkpointing=use_deprecated_checkpointing)
  return (
      pytree_dict['optimizer_state'],
      pytree_dict['params'],
      pytree_dict['batch_stats'],
      pytree_dict['training_metrics_grabber'],
      extra_state['global_step'],
      extra_state['sum_train_cost'],
      extra_state['preemption_count'],
      True)


def _log_epoch_report(report, metrics_logger):
  logging.info('Step %d, steps/second: %f, report: %r', report['global_step'],
               report['steps_per_sec'], report)
  if metrics_logger:
    metrics_logger.append_scalar_metrics(report)
  logging.info('Finished (estimated) epoch %d. Saving checkpoint.',
               report['epoch'])


def _maybe_log_training_metrics(training_metrics_grabber, metrics_logger):
  if training_metrics_grabber:
    summary_tree = utils.get_summary_tree(training_metrics_grabber)
    metrics_logger.append_pytree(summary_tree)


def _write_trial_meta_data(meta_data_path, meta_data):
  d = meta_data.copy()
  d['timestamp'] = time.time()
  with gfile.GFile(meta_data_path, 'w') as f:
    f.write(json.dumps(d, indent=2))


def _maybe_sync_batchnorm_stats(batch_stats):
  """Sync batch_stats across devices."""
  # We first check that batch_stats is used (pmap will throw an error if
  # it's a non batch norm model). If batch norm is not used then
  # batch_stats = None. Note that, in the case of using our implementation of
  # virtual batch norm, this will also handle synchronizing the multiple moving
  # averages on each device before doing a cross-host sync.
  if batch_stats:
    batch_stats = jax.pmap(
        model_utils.sync_batchnorm_stats, axis_name='batch')(
            batch_stats)
  return batch_stats


def should_eval(global_step, eval_frequency, eval_steps):
  if eval_steps:
    return global_step in eval_steps
  return global_step % eval_frequency == 0


def train(train_dir,
          model,
          dataset_builder,
          initializer,
          num_train_steps,
          hps,
          rng,
          eval_batch_size,
          eval_num_batches,
          eval_train_num_batches,
          eval_frequency,
          checkpoint_steps,
          eval_steps=None,
          metrics_logger=None,
          init_logger=None,
          training_metrics_config=None,
          callback_configs=None,
          use_deprecated_checkpointing=True):
  """Main training loop.

  Trains the given network on the specified dataset for the given number of
  epochs. Saves the training curve in train_dir/r=3/results.tsv.

  Args:
    train_dir: (str) Path of the training directory.
    model: (BaseModel) Model object to be trained.
    dataset_builder: dataset builder returned by datasets.get_dataset.
    initializer: Must have API as defined in initializers.py
    num_train_steps: (int) Number of steps to train on.
    hps: (tf.HParams) Model, initialization and training hparams.
    rng: (jax.random.PRNGKey) Rng seed used in model initialization and data
      shuffling.
    eval_batch_size: the evaluation batch size. If None, use hps.batch_size.
    eval_num_batches: (int) The number of batches used for evaluating on
      validation and test sets. Set to None to evaluate on the whole train set.
    eval_train_num_batches: (int) The number of batches for evaluating on train.
      Set to None to evaluate on the whole training set.
    eval_frequency: (int) Evaluate every k steps.
    checkpoint_steps: List of integers indicating special steps to save
      checkpoints at. These checkpoints do not get used for preemption recovery.
    eval_steps: List of integers indicating which steps to perform evals. If
      provided, eval_frequency will be ignored. Performing an eval implies
      saving a checkpoint that will be used to resume training in the case of
      preemption.
    metrics_logger: Used to log all eval metrics during training. See
      utils.MetricLogger for API definition.
    init_logger: Used for black box initializers that have learning curves.
    training_metrics_config: Dict specifying the configuration of the
      training_metrics_grabber. Set to None to skip logging of advanced training
      metrics.
    callback_configs: List of configs specifying general callbacks to run
      during the eval phase. Empty list means no callbacks are run. See
      callbacks.py for details on what is expected in a config.
    use_deprecated_checkpointing: Whether to use deprecated checkpointing.

  Yields:
    metrics: A dictionary of all eval metrics from the given epoch.
  """
  # NOTE: the initialization RNG should *not* be per-host, as this will create
  # different sets of weights per host. However, all other RNGs should be
  # per-host.
  # TODO(znado,gilmer,gdahl): implement replicating the same initialization
  # across hosts.
  rng, init_rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.process_index())
  rng, data_rng = jax.random.split(rng)

  # only used if checkpoints_steps is non-empty.
  checkpoint_dir = os.path.join(train_dir, 'checkpoints')

  if jax.process_index() == 0:
    logging.info('Let the training begin!')
    logging.info('Dataset input shape: %r', hps.input_shape)
    logging.info('Hyperparameters: %s', hps)

  if eval_batch_size is None:
    eval_batch_size = hps.batch_size
  if callback_configs is None:
    callback_configs = []

  # Maybe run the initializer.
  unreplicated_params, unreplicated_batch_stats = initialize(
      model.flax_module,
      initializer,
      model.loss_fn,
      hps.input_shape,
      hps.output_shape,
      hps,
      init_rng,
      init_logger,
      model.get_fake_batch(hps))

  if jax.process_index() == 0:
    utils.log_pytree_shape_and_statistics(unreplicated_params)
    logging.info('train_size: %d,', hps.train_size)

  lr_fn = schedules.get_schedule_fn(hps.lr_hparams, num_train_steps)

  optimizer_init_fn, optimizer_update_fn = optimizers.get_optimizer(hps, model)
  unreplicated_optimizer_state = optimizer_init_fn(unreplicated_params)

  unreplicated_training_metrics_grabber = None
  if training_metrics_config:
    unreplicated_training_metrics_grabber = utils.TrainingMetricsGrabber.create(
        unreplicated_params, training_metrics_config)

  (optimizer_state, params, batch_stats, training_metrics_grabber,
   global_step, sum_train_cost,
   preemption_count,
   is_restored) = _replicate_and_maybe_restore_latest_checkpoint(
       unreplicated_optimizer_state=unreplicated_optimizer_state,
       unreplicated_params=unreplicated_params,
       unreplicated_batch_stats=unreplicated_batch_stats,
       unreplicated_training_metrics_grabber=(
           unreplicated_training_metrics_grabber),
       train_dir=train_dir,
       use_deprecated_checkpointing=use_deprecated_checkpointing)

  if is_restored:
    preemption_count += 1
    # Fold the restored step into the dataset RNG so that we will get a
    # different shuffle each time we restore, so that we do not repeat a
    # previous dataset ordering again after restoring. This is not the only
    # difference in shuffling each pre-emption, because we often times reshuffle
    # the input files each time in a non-deterministic manner.
    #
    # Note that if we are pre-empted more than once per epoch then we will
    # retrain more on the beginning of the training split, because each time we
    # restore we refill the shuffle buffer with the first `shuffle_buffer_size`
    # elements from the training split to continue training.
    #
    # Also note that for evaluating on the training split, because we are
    # reshuffling each time, we will get a new eval_train split each time we are
    # pre-empted.
    data_rng = jax.random.fold_in(data_rng, global_step)

  assert hps.batch_size % (jax.device_count()) == 0
  assert eval_batch_size % (jax.device_count()) == 0
  dataset = dataset_builder(
      data_rng,
      hps.batch_size,
      eval_batch_size=eval_batch_size,
      hps=hps,
  )

  update_fn = functools.partial(
      update,
      training_cost=model.training_cost,
      grad_clip=hps.get('grad_clip'),
      optimizer_update_fn=optimizer_update_fn)
  # in_axes = (
  #     optimizer_state = 0,
  #     params = 0,
  #     batch_stats = 0,
  #     batch = 0,
  #     step = None,
  #     lr = None,
  #     rng = None,
  #     local_device_index = 0,
  #     training_metrics_grabber = 0,
  #     training_cost,
  #     grad_clip,
  #     optimizer_update_fn)
  # Also, we can donate buffers for 'optimizer', 'batch_stats',
  # 'batch' and 'training_metrics_grabber' for update's pmapped computation.
  update_pmapped = jax.pmap(
      update_fn,
      axis_name='batch',
      in_axes=(0, 0, 0, 0, None, None, None, 0, 0),
      donate_argnums=(0, 1, 2, 7))
  # During eval, we can donate the 'batch' buffer. We don't donate the
  # 'params' and 'batch_stats' buffers as we don't re-assign those values in
  # eval, we do that only in train.
  evaluate_batch_pmapped = jax.pmap(
      model.evaluate_batch, axis_name='batch', donate_argnums=(2,))
  start_time = time.time()
  start_step = global_step
  prev_eval_step = start_step
  def get_step_frequency(cur_step):
    return float(cur_step - start_step) / (time.time() - start_time)

  if jax.process_index() == 0:
    logging.info('Starting training!')

  # Numpy array of range(0, local_device_count) to send to each device to be
  # folded into the RNG inside each train step to get a unique per-device RNG.
  local_device_indices = np.arange(jax.local_device_count())

  # Start at the resumed step and continue until we have finished the number of
  # training steps. If building a dataset iterator using a tf.data.Dataset, in
  # the case of a batch size that does not evenly divide the training dataset
  # size, if using `ds.batch(..., drop_remainer=True)` on the training dataset
  # then the final batch in this iterator will be a partial batch. However, if
  # `drop_remainer=False`, then this iterator will always return batches of the
  # same size, and the final batch will have elements from the start of the
  # (num_epochs + 1)-th epoch.
  train_iter = itertools.islice(
      dataset.train_iterator_fn(), global_step, num_train_steps)

  eval_callbacks = []
  rng, callback_rng = jax.random.split(rng)
  callback_rngs = jax.random.split(callback_rng, len(callback_configs))
  for callback_rng, config in zip(callback_rngs, callback_configs):
    eval_callback = callbacks.get_callback(
        config['callback_name'])(model, params, batch_stats,
                                 dataset, hps, config, train_dir, callback_rng)
    eval_callbacks.append(eval_callback)

  for batch in train_iter:
    if global_step in checkpoint_steps and jax.process_index() == 0:
      save_checkpoint(
          checkpoint_dir, {
              'optimizer_state': optimizer_state,
              'params': params,
              'batch_stats': batch_stats,
              'training_metrics_grabber': training_metrics_grabber,
          },
          global_step,
          preemption_count,
          sum_train_cost,
          max_to_keep=None,
          use_deprecated_checkpointing=use_deprecated_checkpointing)
    batch = data_utils.shard(batch)
    lr = lr_fn(global_step)
    optimizer_state, params, batch_stats, cost_val, training_metrics_grabber, grad_norm = update_pmapped(
        optimizer_state,
        params,
        batch_stats,
        batch,
        global_step,
        lr,
        rng,
        local_device_indices,
        training_metrics_grabber)
    # Calling float is needed since cost_val is a shape (1,) DeviceArray.
    sum_train_cost += float(np.mean(cost_val))
    global_step += 1
    # TODO(gdahl, gilmer): consider moving this test up.
    # NB: Since this test is after we increment global_step, having 0 in
    # eval_steps does nothing.
    if should_eval(global_step, eval_frequency, eval_steps):
      batch_stats = _maybe_sync_batchnorm_stats(batch_stats)
      report, eval_time = eval_metrics(params,
                                       batch_stats,
                                       dataset,
                                       eval_num_batches,
                                       eval_train_num_batches,
                                       evaluate_batch_pmapped)
      mean_train_cost = sum_train_cost / max(1, global_step - prev_eval_step)
      report.update(learning_rate=float(lr),
                    global_step=global_step,
                    epoch=global_step * hps.batch_size // hps.train_size,
                    steps_per_sec=get_step_frequency(global_step),
                    eval_time=eval_time,
                    grad_norm=np.mean(grad_norm),
                    preemption_count=preemption_count,
                    train_cost=mean_train_cost)

      for eval_callback in eval_callbacks:
        callback_metrics = eval_callback.run_eval(params, batch_stats,
                                                  global_step)
        if set(callback_metrics.keys()).intersection(set(report.keys())):
          raise ValueError('There was a collision between the callback metrics'
                           'and the standard eval metrics keys')
        report.update(callback_metrics)
      yield report
      if jax.process_index() == 0:
        _log_epoch_report(report, metrics_logger)
        _maybe_log_training_metrics(training_metrics_grabber, metrics_logger)
        save_checkpoint(
            train_dir, {
                'optimizer_state': optimizer_state,
                'params': params,
                'batch_stats': batch_stats,
                'training_metrics_grabber': training_metrics_grabber
            },
            global_step,
            preemption_count,
            sum_train_cost,
            use_deprecated_checkpointing=use_deprecated_checkpointing)
      sum_train_cost = 0.0
      prev_eval_step = global_step

  # Always log and checkpoint on host 0 at the end of training.
  # If we moved where in the loop body evals happen then we would not need this
  # test.
  if prev_eval_step != num_train_steps:
    batch_stats = _maybe_sync_batchnorm_stats(batch_stats)
    report, eval_time = eval_metrics(params,
                                     batch_stats,
                                     dataset,
                                     eval_num_batches,
                                     eval_train_num_batches,
                                     evaluate_batch_pmapped)
    lr = lr_fn(global_step)
    # Correct the average for the final partial epoch.
    mean_train_cost = sum_train_cost / max(1, global_step - prev_eval_step)
    report.update(learning_rate=float(lr),
                  global_step=global_step,
                  epoch=global_step * hps.batch_size // hps.train_size,
                  steps_per_sec=get_step_frequency(global_step),
                  eval_time=eval_time,
                  grad_norm=np.mean(grad_norm),
                  preemption_count=preemption_count,
                  train_cost=mean_train_cost)
    yield report
    if jax.process_index() == 0:
      _log_epoch_report(report, metrics_logger)
      _maybe_log_training_metrics(training_metrics_grabber, metrics_logger)
      save_checkpoint(
          train_dir, {
              'optimizer_state': optimizer_state,
              'params': params,
              'batch_stats': batch_stats,
              'training_metrics_grabber': training_metrics_grabber
          },
          global_step,
          preemption_count,
          sum_train_cost,
          use_deprecated_checkpointing=use_deprecated_checkpointing)
  # To make sure the last checkpoint was correctly saved.
  checkpoint.wait_for_checkpoint_save()


def set_up_loggers(
    train_dir, xm_work_unit=None, use_deprecated_checkpointing=True):
  """Creates a logger for eval metrics as well as initialization metrics."""
  csv_path = os.path.join(train_dir, 'measurements.csv')
  pytree_path = os.path.join(train_dir, 'training_metrics')
  metrics_logger = utils.MetricLogger(
      csv_path=csv_path,
      pytree_path=pytree_path,
      xm_work_unit=xm_work_unit,
      events_dir=train_dir,
      use_deprecated_checkpointing=use_deprecated_checkpointing)

  init_csv_path = os.path.join(train_dir, 'init_measurements.csv')
  init_json_path = os.path.join(train_dir, 'init_scalars.json')
  init_logger = utils.MetricLogger(
      csv_path=init_csv_path,
      json_path=init_json_path,
      xm_work_unit=xm_work_unit,
      use_deprecated_checkpointing=use_deprecated_checkpointing)
  return metrics_logger, init_logger


@functools.partial(jax.pmap, axis_name='hosts')
def _sum_seeds_pmapped(seed):
  return lax.psum(seed, 'hosts')


def create_synchronized_rng_seed():
  rng_seed = np.int64(struct.unpack('q', os.urandom(8))[0])
  rng_seed = _sum_seeds_pmapped(jax_utils.replicate(rng_seed))
  rng_seed = np.sum(rng_seed)
  return rng_seed


def run(
    dataset_name,
    eval_batch_size,
    eval_num_batches,
    eval_train_num_batches,
    eval_frequency,
    checkpoint_steps,
    eval_steps,
    hparam_file,
    hparam_overrides,
    initializer_name,
    model_name,
    loss_name,
    metrics_name,
    num_train_steps,
    experiment_dir,
    worker_id,
    training_metrics_config,
    callback_configs,
    use_deprecated_checkpointing):
  """Function that runs a Jax experiment. See flag definitions for args."""
  model_cls = models.get_model(model_name)
  initializer = initializers.get_initializer(initializer_name)
  dataset_builder = datasets.get_dataset(dataset_name)
  dataset_meta_data = datasets.get_dataset_meta_data(dataset_name)

  merged_hps = hyperparameters.build_hparams(
      model_name=model_name,
      initializer_name=initializer_name,
      dataset_name=dataset_name,
      hparam_file=hparam_file,
      hparam_overrides=hparam_overrides)

  # Note that one should never tune an RNG seed!!! The seed is only included in
  # the hparams for convenience of running hparam trials with multiple seeds per
  # point.
  rng_seed = merged_hps.rng_seed
  if merged_hps.rng_seed < 0:
    rng_seed = create_synchronized_rng_seed()
  xm_experiment = None
  if jax.process_index() == 0:
    logging.info('Running with seed %d', rng_seed)
  rng = jax.random.PRNGKey(rng_seed)

  # Build the loss_fn, metrics_bundle, and flax_module.
  model = model_cls(merged_hps, dataset_meta_data, loss_name, metrics_name)
  trial_dir = os.path.join(experiment_dir, str(worker_id))
  meta_data_path = os.path.join(trial_dir, 'meta_data.json')
  meta_data = {'worker_id': worker_id, 'status': 'incomplete'}
  if jax.process_index() == 0:
    logging.info('rng: %s', rng)
    gfile.makedirs(trial_dir)
    # Set up the metric loggers for host 0.
    xm_work_unit = None
    metrics_logger, init_logger = set_up_loggers(
        trial_dir,
        xm_work_unit,
        use_deprecated_checkpointing)
    hparams_fname = os.path.join(trial_dir, 'hparams.json')
    logging.info('saving hparams to %s', hparams_fname)
    with gfile.GFile(hparams_fname, 'w') as f:
      f.write(merged_hps.to_json())
    _write_trial_meta_data(meta_data_path, meta_data)
  else:
    metrics_logger = None
    init_logger = None
  try:
    epoch_reports = list(
        train(
            trial_dir,
            model,
            dataset_builder,
            initializer,
            num_train_steps,
            merged_hps,
            rng,
            eval_batch_size,
            eval_num_batches,
            eval_train_num_batches,
            eval_frequency,
            checkpoint_steps,
            eval_steps,
            metrics_logger,
            init_logger,
            training_metrics_config=training_metrics_config,
            callback_configs=callback_configs,
            use_deprecated_checkpointing=use_deprecated_checkpointing,
        ))
    logging.info(epoch_reports)
    meta_data['status'] = 'done'
  except utils.TrainingDivergedError as err:
    meta_data['status'] = 'diverged'
    raise err
  finally:
    if jax.process_index() == 0:
      _write_trial_meta_data(meta_data_path, meta_data)
