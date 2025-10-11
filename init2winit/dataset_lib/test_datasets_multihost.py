# coding=utf-8
# Copyright 2025 The init2winit Authors.
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
import jax.experimental.multihost_utils
import tensorflow as tf



def get_shape(pytree):
  """Returns a pytree of shapes."""
  return jax.tree_util.tree_map(lambda x: x.shape, pytree)


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
        'init2winit.dataset_lib.fineweb_edu_10b_input_pipeline.tf.data.Dataset.load'
    )
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

      batch_size = jax.process_count() * jax.local_device_count()
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
      sharded_batch = jax.tree_util.tree_map(make_global_array_fn, batch)
      sharded_batch_shape = get_shape(sharded_batch)  # global shape

      for shape in sharded_batch_shape.values():
        self.assertEqual(shape[0], batch_size)

if __name__ == '__main__':
  multiprocess_test_util.main()
