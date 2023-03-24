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

r"""Main file for the init2winit project.

"""

import os
import sys

from absl import app
from absl import flags
from absl import logging
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
import jax
import tensorflow as tf

# Don't let TF see the GPU, because all we use it for is tf.data loading.
tf.config.experimental.set_visible_devices([], 'GPU')

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

flags.DEFINE_string('dataset', None, 'Which dataset to inspect')
flags.DEFINE_string('model', None, 'Which model to use')
flags.DEFINE_integer('batch_size', None,
                     'Number of examples to retrieve in 1 batch')
flags.DEFINE_integer('num_batches', None, 'Number of batches to retrieve')

FLAGS = flags.FLAGS


def main(unused_argv):
  if jax.process_index() == 0:
    logging.info('argv:\n%s', ' '.join(sys.argv))
    logging.info('device_count: %d', jax.device_count())
    logging.info('num_hosts : %d', jax.process_count())
    logging.info('host_id : %d', jax.process_index())

    if FLAGS.batch_size is None or FLAGS.batch_size <= 0:
      raise ValueError("""FLAGS.batch_size value is invalid,
          expected a positive non-zero integer.""")

    if FLAGS.dataset is None:
      raise ValueError("""FLAGS.dataset value is invalid,
          expected a non-empty string describing dataset name.""")

    batch_size = FLAGS.batch_size
    num_batches = FLAGS.num_batches
    dataset_name = FLAGS.dataset
    model_name = FLAGS.model
    initializer_name = 'noop'

    hparam_overrides = {
        'batch_size': batch_size,
    }

    hps = hyperparameters.build_hparams(
        model_name=model_name,
        initializer_name=initializer_name,
        dataset_name=dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides)

    rng = jax.random.PRNGKey(0)
    rng, data_rng = jax.random.split(rng)

    dataset = datasets.get_dataset(FLAGS.dataset)(data_rng, batch_size,
                                                  batch_size, hps)
    train_iter = dataset.train_iterator_fn()

    for i in range(num_batches):
      batch = next(train_iter)
      logging.info('train batch_num = %d, batch = %r', i, batch)

    for batch in dataset.valid_epoch(num_batches):
      logging.info('validation batch = %r', batch)


if __name__ == '__main__':
  app.run(main)
