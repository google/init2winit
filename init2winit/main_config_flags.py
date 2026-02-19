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

r"""Main file for the init2winit project, updated to use config_flags.

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
from jax.experimental import multihost_utils
from ml_collections import config_flags
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf
from vizier import pyvizier


gfile = tf.io.gfile

# For internal compatibility reasons, we need to pull this function out.
makedirs = tf.io.gfile.makedirs

# Allow caching for any size executables since we had a small executable that
# took 50 minutes to compile.
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
# Since we don't restrict caching by executable size, set a slightly higher
# minimum compile time threshold than the default of 1 second.
jax.config.update('jax_persistent_cache_min_compile_time_secs', 10)
jax.config.update('jax_log_compiles', True)
# Setting jax default prng implementation to protect against jax defaults
# change.
jax.config.update('jax_default_prng_impl', 'threefry2x32')
jax.config.update('jax_threefry_partitionable', True)

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path of the configuration file.',
    lock_config=True,
)


# Special, has no direct equivalent in config.
flags.DEFINE_string('experiment_dir', None,
                    'Path to save weights and other results. Each trial '
                    'directory will have path experiment_dir/worker_id/.')


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
  seeds = multihost_utils.process_allgather(rng_seed)
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
    checkpoint_ttl,
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
            dataset_meta_data=dataset_meta_data,
            loss_name=loss_name,
            metrics_name=metrics_name,
            data_selector=data_selector,
            training_algorithm_class=training_algorithm_class,
            checkpoint_ttl=checkpoint_ttl,
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
  logging.info(
      'jax_compilation_cache_dir=%s', jax.config.jax_compilation_cache_dir
  )
  logging.info(
      'jax_enable_compilation_cache=%s', jax.config.jax_enable_compilation_cache
  )
  # Don't let TF see the GPU, because all we use it for is tf.data loading.
  tf.config.set_visible_devices([], 'GPU')

  config = FLAGS.config
  logging.info('config: %s', config)
  if config.hparam_overrides_json:
    with config.ignore_type():
      config.hparam_overrides = config.hparam_overrides_json


  if config.callback_configs:
    callback_configs = config.callback_configs
  else:  # If config.callback_configs is None, convert to empty list.
    callback_configs = []

  checkpoint_steps = [int(s.strip()) for s in config.checkpoint_steps]
  eval_steps = [int(s.strip()) for s in config.eval_steps]
  if jax.process_index() == 0:
    makedirs(experiment_dir, mode=0o775)
  log_encoding = 'r=3'
  log_dir = os.path.join(experiment_dir, log_encoding)
  makedirs(log_dir, mode=0o775)
  log_path = os.path.join(
      log_dir, 'worker{}_{}.log'.format(worker_id, jax.process_index()))
  with gfile.GFile(log_path, 'a') as logfile:
    utils.add_log_file(logfile)
    if jax.process_index() == 0:
      logging.info('argv:\n%s', ' '.join(sys.argv))
      logging.info('config:\n%s', config)
      logging.info('device_count: %d', jax.device_count())
      logging.info('num_hosts : %d', jax.process_count())
      logging.info('host_id : %d', jax.process_index())
      logging.info('checkpoint_steps: %r', checkpoint_steps)
      logging.info('eval_steps: %r', eval_steps)

    trainer_cls = trainers.get_trainer_cls(config.trainer)
    _run(
        trainer_cls=trainer_cls,
        dataset_name=config.dataset,
        data_selector_name=config.data_selector,
        eval_batch_size=config.eval_batch_size,
        eval_use_ema=config.eval_use_ema,
        eval_num_batches=config.eval_num_batches,
        test_num_batches=config.test_num_batches,
        eval_train_num_batches=config.eval_train_num_batches,
        eval_frequency=config.eval_frequency,
        checkpoint_steps=checkpoint_steps,
        num_tf_data_prefetches=config.num_tf_data_prefetches,
        num_device_prefetches=config.num_device_prefetches,
        num_tf_data_map_parallel_calls=config.num_tf_data_map_parallel_calls,
        early_stopping_target_name=config.early_stopping_target_name,
        early_stopping_target_value=config.early_stopping_target_value,
        early_stopping_mode=config.early_stopping_mode,
        early_stopping_min_steps=config.early_stopping_min_steps,
        eval_steps=eval_steps,
        hparam_file=None,  # Deprecated. TODO(gdahl): Needs to be removed.
        allowed_unrecognized_hparams=config.allowed_unrecognized_hparams,
        hparam_overrides=config.hparam_overrides,
        initializer_name=config.initializer,
        model_name=config.model,
        loss_name=config.loss,
        metrics_name=config.metrics,
        num_train_steps=config.num_train_steps,
        experiment_dir=experiment_dir,
        worker_id=worker_id,
        training_metrics_config=config.training_metrics_config,
        callback_configs=callback_configs,
        external_checkpoint_path=config.external_checkpoint_path,
        training_algorithm_name=config.training_algorithm,
        checkpoint_ttl=config.ttl,
        compile_init_on_cpu=config.compile_init_on_cpu,
    )


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_dir')
  jax.config.config_with_absl()
  app.run(main)
