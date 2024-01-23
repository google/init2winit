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

r"""Used to evaluate a the Hessian eigenspectrum of a given checkpoint.

"""

import json
import os
import sys

from absl import app
from absl import flags
from absl import logging
from init2winit.dataset_lib import datasets
from init2winit.hessian import hessian_eval
from init2winit.hessian import  run_lanczos
from init2winit.model_lib import models
import jax
from ml_collections.config_dict import config_dict
import tensorflow.compat.v2 as tf

gfile = tf.io.gfile

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_integer('eval_num_batches', 10, 'How many batches to evaluate.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Path to the checkpoint to evaluate.')
flags.DEFINE_integer('seed', 0, 'seed used to initialize the computation.')
flags.DEFINE_integer('worker_id', 1,
                     'Client id for hparam sweeps and tuning studies.')
flags.DEFINE_string('experiment_config_filename', None,
                    'Path to the config.json file for this experiment.')
flags.DEFINE_string('model', '', 'Name of the model used to evaluate (not'
                    'needed if experiment_config_filenmae is provided).')
flags.DEFINE_string('dataset', '', 'Name of the dataset used to evaluate (not'
                    'needed if experiment_config_filenmae is provided).')
flags.DEFINE_string(
    'hparam_overrides', '', 'json representation of a flattened dict of hparam '
    'overrides. For nested dictionaries, the override key '
    'should be specified as lr_hparams.base_lr.')
flags.DEFINE_integer('min_global_step', None, 'Lower bound of the step'
                     'interval to evaluate. None indicates all checkpoints.')
flags.DEFINE_integer('max_global_step', None, 'Upper bound of the step'
                     'interval to evaluate [min_global_step, max_global_step).')
flags.DEFINE_string(
    'trial_hparams_filename', None,
    'Path to the hparams.json file for the trial we want to run inference on.')
# TODO(gilmer): Find a better way to handle passing dictionaries.
flags.DEFINE_string(
    'hessian_eval_config', '',
    'Json representation of the hessian evaluation config.')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Necessary to use the tfds imagenet loader.
  tf.enable_v2_behavior()


  rng = jax.random.PRNGKey(FLAGS.seed)

  if FLAGS.hessian_eval_config:
    hessian_eval_config = json.loads(FLAGS.hessian_eval_config)
  else:
    hessian_eval_config = hessian_eval.DEFAULT_EVAL_CONFIG

  if FLAGS.experiment_config_filename:
    with gfile.GFile(FLAGS.experiment_config_filename, 'r') as f:
      experiment_config = json.load(f)
    if jax.process_index() == 0:
      logging.info('experiment_config: %r', experiment_config)
    dataset_name = experiment_config['dataset']
    model_name = experiment_config['model']
  else:
    assert FLAGS.dataset and FLAGS.model
    dataset_name = FLAGS.dataset
    model_name = FLAGS.model

  if jax.process_index() == 0:
    logging.info('argv:\n%s', ' '.join(sys.argv))
    logging.info('device_count: %d', jax.device_count())
    logging.info('num_hosts : %d', jax.process_count())
    logging.info('host_id : %d', jax.process_index())

  model = models.get_model(model_name)
  dataset_builder = datasets.get_dataset(dataset_name)
  dataset_meta_data = datasets.get_dataset_meta_data(dataset_name)

  with gfile.GFile(FLAGS.trial_hparams_filename, 'r') as f:
    hps = config_dict.ConfigDict(json.load(f))

  if FLAGS.hparam_overrides:
    if isinstance(FLAGS.hparam_overrides, str):
      hparam_overrides = json.loads(FLAGS.hparam_overrides)
    hps.update_from_flattened_dict(hparam_overrides)
  run_lanczos.eval_checkpoints(
      FLAGS.checkpoint_dir,
      hps,
      rng,
      FLAGS.eval_num_batches,
      model,
      dataset_builder,
      dataset_meta_data,
      hessian_eval_config,
      FLAGS.min_global_step,
      FLAGS.max_global_step)


if __name__ == '__main__':
  app.run(main)
