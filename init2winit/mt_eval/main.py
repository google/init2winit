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

r"""Used to evaluate MT model (BLEU/cross_entropy_loss/log_perplexity).

"""

import json
import os
import sys

from absl import app
from absl import flags
from absl import logging
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
from init2winit.model_lib import models
from init2winit.mt_eval import bleu_evaluator
import jax
import tensorflow.compat.v2 as tf




# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_string('checkpoint_dir', '', 'Path to the checkpoint to evaluate.')
flags.DEFINE_integer('seed', 0, 'seed used to initialize the computation.')
flags.DEFINE_integer('worker_id', 1,
                     'Client id for hparam sweeps and tuning studies.')
flags.DEFINE_string('experiment_config_filename', None,
                    'Path to the config.json file for this experiment.')
flags.DEFINE_string(
    'model', '', 'Name of the model used to evaluate (not'
    'needed if experiment_config_filenmae is provided).')
flags.DEFINE_string(
    'dataset', '', 'Name of the dataset used to evaluate (not'
    'needed if experiment_config_filenmae is provided).')
flags.DEFINE_string(
    'hparam_overrides', '', 'json representation of a flattened dict of hparam '
    'overrides. For nested dictionaries, the override key '
    'should be specified as lr_hparams.initial_value.')
flags.DEFINE_string(
    'trial_hparams_filename', None,
    'Path to the hparams.json file for the trial we want to run inference on.')
flags.DEFINE_string('mt_eval_config', '',
                    'Json representation of the mt evaluation config.')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Necessary to use the tfds loader.
  tf.enable_v2_behavior()

  if jax.process_count() > 1:
    # TODO(ankugarg): Add support for multihost inference.
    raise NotImplementedError('BLEU eval does not support multihost inference.')

  rng = jax.random.PRNGKey(FLAGS.seed)

  mt_eval_config = json.loads(FLAGS.mt_eval_config)

  if FLAGS.experiment_config_filename:
    with  tf.io.gfile.GFile(FLAGS.experiment_config_filename) as f:
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
    logging.info('num_hosts : %d', jax.host_count())
    logging.info('host_id : %d', jax.host_id())

  model_class = models.get_model(model_name)
  dataset_builder = datasets.get_dataset(dataset_name)
  dataset_meta_data = datasets.get_dataset_meta_data(dataset_name)

  hparam_overrides = None
  if FLAGS.hparam_overrides:
    if isinstance(FLAGS.hparam_overrides, str):
      hparam_overrides = json.loads(FLAGS.hparam_overrides)

  merged_hps = hyperparameters.build_hparams(
      model_name=model_name,
      initializer_name=experiment_config['initializer'],
      dataset_name=dataset_name,
      hparam_file=FLAGS.trial_hparams_filename,
      hparam_overrides=hparam_overrides)

  if jax.process_index() == 0:
    logging.info('Merged hps are: %s', json.dumps(merged_hps.to_json()))

  evaluator = bleu_evaluator.BLEUEvaluator(FLAGS.checkpoint_dir, merged_hps,
                                           rng,
                                           model_class, dataset_builder,
                                           dataset_meta_data,
                                           mt_eval_config)
  evaluator.translate_and_calculate_bleu()


if __name__ == '__main__':
  app.run(main)
