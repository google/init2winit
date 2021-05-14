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

r"""Main file for the init2winit project.

"""

import json
import os
import sys

from absl import app
from absl import flags
from absl import logging

from init2winit import trainer
import utils as utils  # local file import
import jax
import tensorflow as tf

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_boolean(
    'use_deprecated_checkpointing',
    True,
    'Whether or not to use deprecated checkpointing.')
flags.DEFINE_string('model', 'fully_connected', 'Name of the model to train')
flags.DEFINE_string('loss', 'cross_entropy', 'Loss function')
flags.DEFINE_string('metrics', 'classification_metrics',
                    'Metrics to be used for evaluation')
flags.DEFINE_string('initializer', 'noop', 'Must be in [noop, meta_init]')
flags.DEFINE_string('experiment_dir', None,
                    'Path to save weights and other results. Each trial '
                    'directory will have path experiment_dir/worker_id/')
flags.DEFINE_string('dataset', 'mnist',
                    'Which dataset to train on')
flags.DEFINE_integer('num_train_steps', None, 'The number of steps to train')
flags.DEFINE_integer('eval_batch_size', None, 'Batch size for evaluation')
flags.DEFINE_integer('eval_num_batches', None,
                     'Number of batches for evaluation. Leave None to evaluate '
                     'on the entire validation and test set.')
flags.DEFINE_integer('eval_train_num_batches', 0,
                     'Number of batches when evaluating on the training set')
flags.DEFINE_integer('eval_frequency', 1000, 'Evaluate every k steps.')
flags.DEFINE_string(
    'hparam_overrides', '', 'json representation of a flattened dict of hparam '
    'overrides. For nested dictionaries, the override key '
    'should be specified as lr_hparams.initial_value.')
flags.DEFINE_list(
    'checkpoint_steps', [], 'List of steps to checkpoint the'
    ' model. The checkpoints will be saved in a separate'
    'directory train_dir/checkpoints. Note these checkpoints'
    'will be in addition to the normal checkpointing that'
    'occurs during training for preemption purposes.')
flags.DEFINE_list(
    'eval_steps', [],
    'List of steps to evaluate the model. Evaluating implies saving a '
    'checkpoint for preemption recovery.')
flags.DEFINE_string(
    'hparam_file', None, 'Optional path to hparam json file for overriding '
    'hyperparameters. Hyperparameters are loaded before '
    'applying --hparam_overrides.')
flags.DEFINE_string(
    'training_metrics_config', '',
    'JSON representation of the training metrics config.')

flags.DEFINE_integer('worker_id', 1,
                     'Client id for hparam sweeps and tuning studies.')

FLAGS = flags.FLAGS


def main(unused_argv):

  # TODO(gdahl) Figure out a better way to handle passing more complicated
  # flags to the binary.
  training_metrics_config = None
  if FLAGS.training_metrics_config:
    training_metrics_config = json.loads(FLAGS.training_metrics_config)

  checkpoint_steps = [int(s.strip()) for s in FLAGS.checkpoint_steps]
  eval_steps = [int(s.strip()) for s in FLAGS.eval_steps]
  if jax.host_id() == 0:
    tf.io.gfile.makedirs(FLAGS.experiment_dir)
  log_dir = os.path.join(FLAGS.experiment_dir, 'r=3/')
  tf.io.gfile.makedirs(log_dir)
  log_path = os.path.join(
      log_dir, 'worker{}_{}.log'.format(FLAGS.worker_id, jax.host_id()))
  with tf.io.gfile.GFile(log_path, 'a') as logfile:
    utils.add_log_file(logfile)
    if jax.host_id() == 0:
      logging.info('argv:\n%s', ' '.join(sys.argv))
      logging.info('device_count: %d', jax.device_count())
      logging.info('num_hosts : %d', jax.host_count())
      logging.info('host_id : %d', jax.host_id())
      logging.info('checkpoint_steps: %r', checkpoint_steps)
      logging.info('eval_steps: %r', eval_steps)

    trainer.run(
        dataset_name=FLAGS.dataset,
        eval_batch_size=FLAGS.eval_batch_size,
        eval_num_batches=FLAGS.eval_num_batches,
        eval_train_num_batches=FLAGS.eval_train_num_batches,
        eval_frequency=FLAGS.eval_frequency,
        checkpoint_steps=checkpoint_steps,
        eval_steps=eval_steps,
        hparam_file=FLAGS.hparam_file,
        hparam_overrides=FLAGS.hparam_overrides,
        initializer_name=FLAGS.initializer,
        model_name=FLAGS.model,
        loss_name=FLAGS.loss,
        metrics_name=FLAGS.metrics,
        num_train_steps=FLAGS.num_train_steps,
        experiment_dir=FLAGS.experiment_dir,
        worker_id=FLAGS.worker_id,
        training_metrics_config=training_metrics_config,
        use_deprecated_checkpointing=FLAGS.use_deprecated_checkpointing)


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_dir')
  app.run(main)
