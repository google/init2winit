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

r"""Tool to preprocess OGBG MolPCBA dataset.

This script loads the OGBG MolPCBA dataset using the existing pipeline (which
does dynamic batching and padding), unrolls it for a specified number of steps,
and saves the resulting dataset to disk using tf.data.Dataset.save.

This allows for much faster loading during training by avoiding the expensive
graph processing and dynamic batching at runtime.
"""

import logging
from absl import app
from absl import flags
from init2winit.dataset_lib import ogbg_molpcba
from init2winit.dataset_lib import ogbg_molpcba_preprocessed
from ml_collections import config_dict
import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s', force=True
)

# Setup flags
flags.DEFINE_string('output_path', None, 'Path to save the processed dataset.')
flags.DEFINE_integer('num_steps', 10000000, 'Number of steps to generate.')
flags.DEFINE_integer(
    'batch_size',
    256,
    'Batch size (per-host if sharded later, but here it is just the batch size'
    ' produced).',
)
flags.DEFINE_integer(
    'target_num_processes', None, 'Target number of processes.'
)
flags.DEFINE_float('batch_nodes_multiplier', 1.0, 'Multiplier for max nodes.')
flags.DEFINE_float('batch_edges_multiplier', 2.0, 'Multiplier for max edges.')
flags.DEFINE_float('avg_nodes_per_graph', 26.0, 'Average nodes per graph.')
flags.DEFINE_float('avg_edges_per_graph', 28.0, 'Average edges per graph.')
flags.DEFINE_bool('add_bidirectional_edges', False, 'Add bidirectional edges.')
flags.DEFINE_bool('add_virtual_node', False, 'Add virtual node.')
flags.DEFINE_bool('add_self_loops', False, 'Add self loops.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.output_path:
    raise ValueError('Must specify --output_path')

  # Construct HParams
  # We use the defaults from ogbg_molpcba and override with flags
  hps = config_dict.ConfigDict(ogbg_molpcba.DEFAULT_HPARAMS)
  hps.batch_nodes_multiplier = FLAGS.batch_nodes_multiplier
  hps.batch_edges_multiplier = FLAGS.batch_edges_multiplier
  hps.avg_nodes_per_graph = int(FLAGS.avg_nodes_per_graph)
  hps.avg_edges_per_graph = int(FLAGS.avg_edges_per_graph)
  hps.add_bidirectional_edges = FLAGS.add_bidirectional_edges
  hps.add_virtual_node = FLAGS.add_virtual_node
  hps.add_self_loops = FLAGS.add_self_loops

  logging.info(
      'Generating dataset with batch_size=%d for %d devices for %d steps...',
      FLAGS.batch_size,
      FLAGS.target_num_processes,
      FLAGS.num_steps,
  )
  logging.info('Saving to %s', FLAGS.output_path)

  ogbg_molpcba_preprocessed.generate_and_save_dataset(
      hps=hps,
      output_path=FLAGS.output_path,
      num_steps=FLAGS.num_steps,
      batch_size=FLAGS.batch_size,
      seed=FLAGS.seed,
      progress_bar_fn=lambda x: tqdm.tqdm(
          x, desc='Generating batches', unit='batch'
      ),
      target_num_processes=FLAGS.target_num_processes,
  )

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
