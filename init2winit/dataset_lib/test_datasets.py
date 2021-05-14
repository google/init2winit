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

"""Unit tests for datasets.py."""

import itertools

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from init2winit.dataset_lib.datasets import get_dataset
from init2winit.dataset_lib.datasets import get_dataset_hparams
import jax.numpy as jnp
import jax.random
import numpy as np
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS


# NOTE: Imagenet will not satisfy data determinism, so we will not test for it.
small_image_ds = ['mnist', 'cifar10', 'fashion_mnist']
determinism_test_keys = [('test_{}'.format(m), m) for m in small_image_ds]


class DatasetTest(parameterized.TestCase):
  """Unit tests for datasets.py."""

  @parameterized.named_parameters(*determinism_test_keys)
  def test_determinism(self, ds):
    """Test that shuffle_rng and epoch correctly determine the order of data."""
    batch_size = 32
    eval_batch_size = 16
    np.random.seed(0)  # set the seed so the mock data is deterministic.

    # This will override the tfds.load(mnist) call to return 100 fake samples.
    with tfds.testing.mock_data(num_examples=128):
      dataset_builder = get_dataset(ds)
      hps = get_dataset_hparams(ds)
      hps.train_size = 80
      hps.valid_size = 48
      hps.test_size = 40
      dataset = dataset_builder(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=batch_size,
          eval_batch_size=eval_batch_size,
          hps=hps)
      dataset_copy = dataset_builder(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=batch_size,
          eval_batch_size=eval_batch_size,
          hps=hps)
    batch_idx_to_test = 1

    saved_batch = next(itertools.islice(
        dataset.train_iterator_fn(), batch_idx_to_test,
        batch_idx_to_test + 1))
    saved_batch_same_epoch = next(itertools.islice(
        dataset_copy.train_iterator_fn(), batch_idx_to_test,
        batch_idx_to_test + 1))
    saved_batch_diff_epoch = next(itertools.islice(
        dataset.train_iterator_fn(), batch_idx_to_test + 3,
        batch_idx_to_test + 4))

    saved_batch_eval = next(itertools.islice(
        dataset.valid_epoch(), batch_idx_to_test,
        batch_idx_to_test + 1))
    saved_batch_eval_same_epoch = next(
        itertools.islice(dataset_copy.valid_epoch(), batch_idx_to_test,
                         batch_idx_to_test + 1))

    self.assertTrue(
        jnp.array_equal(saved_batch['inputs'],
                        saved_batch_same_epoch['inputs']))
    self.assertTrue(
        jnp.array_equal(saved_batch['targets'],
                        saved_batch_same_epoch['targets']))
    self.assertFalse(
        jnp.array_equal(saved_batch['inputs'],
                        saved_batch_diff_epoch['inputs']))
    self.assertFalse(
        jnp.array_equal(saved_batch['targets'],
                        saved_batch_diff_epoch['targets']))
    self.assertTrue(
        jnp.array_equal(saved_batch_eval['inputs'],
                        saved_batch_eval_same_epoch['inputs']))

    # Check shapes
    expected_shape = jnp.array([
        batch_size, hps.input_shape[0], hps.input_shape[1], hps.input_shape[2]
    ])
    expected_shape_eval = jnp.array([
        eval_batch_size, hps.input_shape[0],
        hps.input_shape[1], hps.input_shape[2],
    ])
    self.assertTrue(
        jnp.array_equal(saved_batch['inputs'].shape, expected_shape))
    self.assertTrue(
        jnp.array_equal(saved_batch_eval['inputs'].shape, expected_shape_eval))

    expected_target_shape = jnp.array(
        [batch_size, get_dataset_hparams(ds)['output_shape'][-1]])
    self.assertTrue(jnp.array_equal(saved_batch['targets'].shape,
                                    expected_target_shape))

    # Check that the training gen drops the last partial batch.
    drop_partial_batches = list(
        itertools.islice(dataset.train_iterator_fn(), 0, 2))

    # Check that the validation set correctly pads the final partial batch.
    no_drop_partial_batches = list(dataset.test_epoch(num_batches=3))
    self.assertLen(drop_partial_batches, 2)
    self.assertLen(no_drop_partial_batches, 3)
    expected_shape = jnp.array([
        80 % batch_size, hps.input_shape[0],
        hps.input_shape[1], hps.input_shape[2],
    ])
    self.assertTrue(
        jnp.array_equal(no_drop_partial_batches[2]['inputs'].shape,
                        expected_shape))

    # We expect the partial batch to have 40 % 16 = 8 non padded inputs.
    self.assertEqual(no_drop_partial_batches[2]['weights'].sum(), 8)

    # Test number of batches
    num_batches = 1
    num_generated = len(
        [
            b for b in itertools.islice(
                dataset.train_iterator_fn(), 0, num_batches)
        ])
    self.assertEqual(num_batches, num_generated)


if __name__ == '__main__':
  absltest.main()
