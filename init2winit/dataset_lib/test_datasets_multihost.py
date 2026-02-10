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

"""Tests for datasets in multihost setting.

"""

import functools
from unittest import mock

from absl import logging
from absl.testing import parameterized
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import datasets
import dataset_lib.fineweb_edu_10b_input_pipeline  # pylint: disable=unused-import  # local file import
import jax
from jax.experimental import mesh_utils
import tensorflow as tf




def ogbg_molpcba_mock(*args, **kwargs):
  del args, kwargs
  num_examples = 10
  num_nodes = 8
  num_edges = 8
  node_dim = 3
  edge_dim = 2
  return tf.data.Dataset.from_generator(
      lambda: (
          {
              'edge_feat': tf.ones(
                  shape=(num_edges, edge_dim), dtype=tf.float32
              ),
              'edge_index': tf.ones(shape=(num_edges, 2), dtype=tf.int64),
              'labels': tf.ones(shape=(128,), dtype=tf.int64),
              'node_feat': tf.ones(
                  shape=(num_nodes, node_dim), dtype=tf.float32
              ),
              'num_nodes': tf.constant([num_nodes], dtype=tf.int64),
              'num_edges': tf.constant([num_edges], dtype=tf.int64),
          }
          for _ in range(num_examples)
      ),
      output_types={
          'edge_feat': tf.float32,
          'edge_index': tf.int64,
          'labels': tf.int64,
          'node_feat': tf.float32,
          'num_nodes': tf.int64,
          'num_edges': tf.int64,
      },
      output_shapes={
          'edge_feat': tf.TensorShape([num_edges, edge_dim]),
          'edge_index': tf.TensorShape([num_edges, 2]),
          'labels': tf.TensorShape([128]),
          'node_feat': tf.TensorShape([num_nodes, node_dim]),
          'num_nodes': tf.TensorShape([1]),
          'num_edges': tf.TensorShape([1]),
      },
  )


def fineweb_edu_10b_mock(*args, **kwargs):  # pylint: disable=unused-argument
  del args, kwargs
  num_examples = 10
  return tf.data.Dataset.from_generator(
      lambda: ({'input_ids': tf.ones(shape=(2049,), dtype=tf.int64)}
               for _ in range(num_examples)),
      output_types={'input_ids': tf.int64},
      output_shapes={'input_ids': tf.TensorShape([2049])},
  )


# Set up a dictionary of dataset names to (mock_fn, patch_target) tuples.
all_dataset_names = {
    'fineweb_edu_10B': (
        fineweb_edu_10b_mock,
        'init2winit.dataset_lib.fineweb_edu_10b_input_pipeline.tf.data.Dataset.load',
    ),
    'ogbg_molpcba': (
        ogbg_molpcba_mock,
        'init2winit.dataset_lib.ogbg_molpcba.tfds.load',
    ),
}

parameterized_tests = [
    ('test_{}'.format(dataset_name), dataset_name, mock_fn, patch_target)
    for dataset_name, (mock_fn, patch_target) in all_dataset_names.items()
]


class DatasetsTest(parameterized.TestCase):
  """Tests for losses.py."""

  def setUp(self):
    super().setUp()
    self.mesh_shape = (jax.device_count(),)
    self.mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(self.mesh_shape, devices=jax.devices()),
        axis_names=('devices',),
    )

  @parameterized.named_parameters(*parameterized_tests)
  def test_multihost_datasets(self, dataset_name, mock_fn, patch_target):
    """Tests batching in multihost setup."""
    with mock.patch(patch_target, side_effect=mock_fn):
      rng = jax.random.PRNGKey(42)
      host_rng = jax.random.fold_in(rng, jax.process_index())

      num_processes = jax.process_count()
      assert num_processes > 1
      batch_size = num_processes * jax.local_device_count()
      logging.info('batch size = %d', batch_size)
      dataset_builder = datasets.get_dataset(dataset_name)
      hps = datasets.get_dataset_hparams(dataset_name)

      dataset = dataset_builder(
          host_rng, batch_size, eval_batch_size=batch_size, hps=hps
      )
      train_iter = dataset.train_iterator_fn()
      batch = next(train_iter)

      make_global_array_fn = functools.partial(
          data_utils.make_global_array, mesh=self.mesh
      )

      # Combine host-local batches into global batch
      global_batch = jax.tree_util.tree_map(make_global_array_fn, batch)
      global_batch_shape_pytree = data_utils.get_batch_size_pytree(
          global_batch
      )

      # Check that the global batch shape is the same as the hps batch size.
      self.assertTrue(
          all(
              global_batch_size == batch_size
              for global_batch_size in jax.tree_util.tree_leaves(
                  global_batch_shape_pytree
              )
          )
      )

if __name__ == '__main__':
  multiprocess_test_util.main()
