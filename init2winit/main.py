# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

r"""Main file for the init2winit project.

"""

import json
import os
import struct
import sys
import time

from absl import flags
from absl import logging
from init2winit import hyperparameters
from init2winit import utils
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainers
from init2winit.trainer_lib import training_algorithms
import jax
from jax.experimental.multihost_utils import process_allgather
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf
from vizier import pyvizier


gfile = tf.io.gfile

# For internal compatibility reasons, we need to pull this function out.
makedirs = tf.io.gfile.makedirs

# Setting jax default prng implementation to protect against jax defaults
# change.
jax.config.update('jax_default_prng_impl', 'threefry2x32')
jax.config.update('jax_threefry_partitionable', True)

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_string('trainer', 'standard', 'Name of the trainer to use.')
flags.DEFINE_string('model', 'fully_connected', 'Name of the model to train.')
flags.DEFINE_string('loss', 'cross_entropy', 'Loss function.')
flags.DEFINE_string('metrics', 'classification_metrics',
                    'Metrics to be used for evaluation.')
flags.DEFINE_string('initializer', 'noop', 'Must be in [noop, meta_init].')
flags.DEFINE_string('experiment_dir', None,
                    'Path to save weights and other results. Each trial '
                    'directory will have path experiment_dir/worker_id/.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to train on.')
flags.DEFINE_string('data_selector', 'noop', 'Which data selector to use.')
flags.DEFINE_integer('num_train_steps', None, 'The number of steps to train.')
flags.DEFINE_integer(
    'num_tf_data_prefetches', -1, 'The number of batches to to prefetch from '
    'network to host at each step. Set to -1 for tf.data.AUTOTUNE.')
flags.DEFINE_integer(
    'num_device_prefetches', 0, 'The number of batches to to prefetch from '
    'host to device at each step.')
flags.DEFINE_integer(
    'num_tf_data_map_parallel_calls', -1, 'The number of parallel calls to '
    'make from tf.data.map. Set to -1 for tf.data.AUTOTUNE.'
)
flags.DEFINE_integer('eval_batch_size', None, 'Batch size for evaluation.')
flags.DEFINE_bool('eval_use_ema', None, 'If True evals will use ema of params.')
flags.DEFINE_integer(
    'eval_num_batches', None,
    'Number of batches for evaluation. Leave None to evaluate '
    'on the entire validation and test set.')
flags.DEFINE_integer(
    'test_num_batches', None,
    'Number of batches for eval on test set. Leave None to evaluate '
    'on the entire test set.')
flags.DEFINE_integer('eval_train_num_batches', None,
                     'Number of batches when evaluating on the training set.')
flags.DEFINE_integer('eval_frequency', 1000, 'Evaluate every k steps.')
flags.DEFINE_string(
    'hparam_overrides', '', 'JSON representation of a flattened dict of hparam '
    'overrides. For nested dictionaries, the override key '
    'should be specified as lr_hparams.base_lr.')
flags.DEFINE_string(
    'callback_configs', '', 'JSON representation of a list of dictionaries '
    'which specify general callbacks to be run during eval of training.')
flags.DEFINE_list(
    'checkpoint_steps', [], 'List of steps to checkpoint the'
    ' model. The checkpoints will be saved in a separate'
    'directory train_dir/checkpoints. Note these checkpoints'
    'will be in addition to the normal checkpointing that'
    'occurs during training for preemption purposes.')
flags.DEFINE_string('external_checkpoint_path', None,
                    'If this argument is set, the trainer will initialize'
                    'the parameters, batch stats, optimizer state, and training'
                    'metrics by loading them from the checkpoint at this path.')

flags.DEFINE_string(
    'early_stopping_target_name',
    None,
    'A string naming the metric to use to perform early stopping. If this '
    'metric reaches the value `early_stopping_target_value`, training will '
    'stop. Must include the dataset split (ex: validation/error_rate).')
flags.DEFINE_float(
    'early_stopping_target_value',
    None,
    'A float indicating the value at which to stop training.')
flags.DEFINE_enum(
    'early_stopping_mode',
    None,
    enum_values=['above', 'below'],
    help=(
        'One of "above" or "below", indicates if we should stop when the '
        'metric is above or below the threshold value. Example: if "above", '
        'then training will stop when '
        '`report[early_stopping_target_name] >= early_stopping_target_value`.'))
flags.DEFINE_integer(
    'early_stopping_min_steps',
    0,
    help='Only allows early stopping after at least this many steps.',
)
flags.DEFINE_list(
    'eval_steps', [],
    'List of steps to evaluate the model. Evaluating implies saving a '
    'checkpoint for preemption recovery.')
flags.DEFINE_string(
    'hparam_file', None, 'Optional path to hparam json file for overriding '
    'hyperparameters. Hyperparameters are loaded before '
    'applying --hparam_overrides.')
flags.DEFINE_list(
    'allowed_unrecognized_hparams', [],
    'Downgrades unrecognized hparam override keys from an error to a warning '
    'for the supplied list of keys.')
flags.DEFINE_string(
    'training_metrics_config', '',
    'JSON representation of the training metrics config.')

flags.DEFINE_integer('worker_id', 1,
                     'Client id for hparam sweeps and tuning studies.')
flags.DEFINE_string(
    'training_algorithm',
    'optax_training_algorithm',
    'Name of the training algorithm to use.',
)
flags.DEFINE_boolean(
    'compile_init_on_cpu',
    False,
    'Whether or not to compile the init function on CPU or on device.',
)

FLAGS = flags.FLAGS


def _write_trial_meta_data(meta_data_path, meta_data):
  d = meta_data.copy()
  d['timestamp'] = time.time()
  with gfile.GFile(meta_data_path, 'w') as f:
    f.write(json.dumps(d, indent=2))


def _create_synchronized_rng_seed():
  """Create an RNG seed synchronized across all processes."""
  # Each process generates its own random seed
  rng_seed = np.int64(struct.unpack('q', os.urandom(8))[0])

  # Gather seeds from all processes and sum them
  seeds = process_allgather(rng_seed)
  return np.int64(np.sum(seeds))


def _run(
    *,
    trainer_cls,
    dataset_name,
    data_selector_name,
    eval_batch_size,
    eval_use_ema,
    eval_num_batches,
    test_num_batches,
    eval_train_num_batches,
    eval_frequency,
    checkpoint_steps,
    num_tf_data_prefetches,
    num_device_prefetches,
    num_tf_data_map_parallel_calls,
    early_stopping_target_name,
    early_stopping_target_value,
    early_stopping_mode,
    early_stopping_min_steps,
    eval_steps,
    hparam_file,
    allowed_unrecognized_hparams,
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
    external_checkpoint_path,
    training_algorithm_name,
    compile_init_on_cpu,
):
  """Function that runs a Jax experiment. See flag definitions for args."""
  model_cls = models.get_model(model_name)
  initializer = initializers.get_initializer(initializer_name)
  dataset_builder = datasets.get_dataset(dataset_name)
  data_selector = datasets.get_data_selector(data_selector_name)
  dataset_meta_data = datasets.get_dataset_meta_data(dataset_name)
  input_pipeline_hps = config_dict.ConfigDict(dict(
      num_tf_data_prefetches=num_tf_data_prefetches,
      num_device_prefetches=num_device_prefetches,
      num_tf_data_map_parallel_calls=num_tf_data_map_parallel_calls,
  ))
  training_algorithm_class = training_algorithms.get_training_algorithm(
      training_algorithm_name
  )

  merged_hps = hyperparameters.build_hparams(
      model_name=model_name,
      initializer_name=initializer_name,
      dataset_name=dataset_name,
      hparam_file=hparam_file,
      hparam_overrides=hparam_overrides,
      input_pipeline_hps=input_pipeline_hps,
      allowed_unrecognized_hparams=allowed_unrecognized_hparams)

  # Note that one should never tune an RNG seed!!! The seed is only included in
  # the hparams for convenience of running hparam trials with multiple seeds per
  # point.
  rng_seed = merged_hps.rng_seed
  if merged_hps.rng_seed < 0:
    rng_seed = _create_synchronized_rng_seed()
  xm_experiment = None
  xm_work_unit = None
  if jax.process_index() == 0:
    logging.info('Running with seed %d', rng_seed)
  rng = jax.random.PRNGKey(rng_seed)

  # Build the loss_fn, metrics_bundle, and flax_module.
  model = model_cls(
      merged_hps,
      dataset_meta_data,
      loss_name,
      metrics_name,
      compile_init_on_cpu=compile_init_on_cpu,
  )
  trial_dir = os.path.join(experiment_dir, str(worker_id))
  meta_data_path = os.path.join(trial_dir, 'meta_data.json')
  meta_data = {'worker_id': worker_id, 'status': 'incomplete'}
  if jax.process_index() == 0:
    logging.info('rng: %s', rng)
    makedirs(trial_dir, mode=0o775)
    # Set up the metric loggers for host 0.
    metrics_logger, init_logger = utils.set_up_loggers(trial_dir, xm_work_unit)
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
        trainer_cls(
            trial_dir,
            model,
            dataset_builder,
            initializer,
            num_train_steps,
            merged_hps,
            rng,
            eval_batch_size,
            eval_use_ema,
            eval_num_batches,
            test_num_batches,
            eval_train_num_batches,
            eval_frequency,
            checkpoint_steps,
            early_stopping_target_name,
            early_stopping_target_value,
            early_stopping_mode,
            early_stopping_min_steps,
            eval_steps,
            metrics_logger,
            init_logger,
            training_metrics_config=training_metrics_config,
            callback_configs=callback_configs,
            external_checkpoint_path=external_checkpoint_path,
            data_selector=data_selector,
            training_algorithm_class=training_algorithm_class,
        ).train()
    )
    logging.info(epoch_reports)
    meta_data['status'] = 'done'
  except utils.TrainingDivergedError as err:
    meta_data['status'] = 'diverged'
    raise err
  finally:
    if jax.process_index() == 0:
      _write_trial_meta_data(meta_data_path, meta_data)


def main(unused_argv):
  # Don't let TF see the GPU, because all we use it for is tf.data loading.
  tf.config.set_visible_devices([], 'GPU')

  # TODO(gdahl) Figure out a better way to handle passing more complicated
  # flags to the binary.
  training_metrics_config = None
  if FLAGS.training_metrics_config:
    training_metrics_config = json.loads(FLAGS.training_metrics_config)
  if FLAGS.callback_configs:
    callback_configs = json.loads(FLAGS.callback_configs)
  else:
    callback_configs = []

  checkpoint_steps = [int(s.strip()) for s in FLAGS.checkpoint_steps]
  eval_steps = [int(s.strip()) for s in FLAGS.eval_steps]
  if jax.process_index() == 0:
    makedirs(experiment_dir, mode=0o775)
  log_dir = os.path.join(experiment_dir, 'r=3/')
  makedirs(log_dir, mode=0o775)
  log_path = os.path.join(
      log_dir, 'worker{}_{}.log'.format(FLAGS.worker_id, jax.process_index()))
  with gfile.GFile(log_path, 'a') as logfile:
    utils.add_log_file(logfile)
    if jax.process_index() == 0:
      logging.info('argv:\n%s', ' '.join(sys.argv))
      logging.info('device_count: %d', jax.device_count())
      logging.info('num_hosts : %d', jax.process_count())
      logging.info('host_id : %d', jax.process_index())
      logging.info('checkpoint_steps: %r', checkpoint_steps)
      logging.info('eval_steps: %r', eval_steps)

    trainer_cls = trainers.get_trainer_cls(FLAGS.trainer)
    _run(
        trainer_cls=trainer_cls,
        dataset_name=FLAGS.dataset,
        data_selector_name=FLAGS.data_selector,
        eval_batch_size=FLAGS.eval_batch_size,
        eval_use_ema=FLAGS.eval_use_ema,
        eval_num_batches=FLAGS.eval_num_batches,
        test_num_batches=FLAGS.test_num_batches,
        eval_train_num_batches=FLAGS.eval_train_num_batches,
        eval_frequency=FLAGS.eval_frequency,
        checkpoint_steps=checkpoint_steps,
        num_tf_data_prefetches=FLAGS.num_tf_data_prefetches,
        num_device_prefetches=FLAGS.num_device_prefetches,
        num_tf_data_map_parallel_calls=FLAGS.num_tf_data_map_parallel_calls,
        early_stopping_target_name=FLAGS.early_stopping_target_name,
        early_stopping_target_value=FLAGS.early_stopping_target_value,
        early_stopping_mode=FLAGS.early_stopping_mode,
        early_stopping_min_steps=FLAGS.early_stopping_min_steps,
        eval_steps=eval_steps,
        hparam_file=FLAGS.hparam_file,
        allowed_unrecognized_hparams=FLAGS.allowed_unrecognized_hparams,
        hparam_overrides=FLAGS.hparam_overrides,
        initializer_name=FLAGS.initializer,
        model_name=FLAGS.model,
        loss_name=FLAGS.loss,
        metrics_name=FLAGS.metrics,
        num_train_steps=FLAGS.num_train_steps,
        experiment_dir=experiment_dir,
        worker_id=FLAGS.worker_id,
        training_metrics_config=training_metrics_config,
        callback_configs=callback_configs,
        external_checkpoint_path=FLAGS.external_checkpoint_path,
        training_algorithm_name=FLAGS.training_algorithm,
        compile_init_on_cpu=FLAGS.compile_init_on_cpu,
    )


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_dir')
  jax.config.config_with_absl()
  app.run(main)
