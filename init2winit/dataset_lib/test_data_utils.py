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

"""Unit tests for datasets.py."""

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.dataset_lib import data_utils
import numpy as np

desired_batch_size = 23
batch_axes = [0, 0, 3, 2]
test_names = ['default', 'NHWC', 'HWCN', 'HWNC']
image_formats = [None, 'NHWC', 'HWCN', 'HWNC']
batch_size = 13
width = 11
num_channels = 3
input_shapes = [
    (batch_size, width, width, num_channels),
    (batch_size, width, width, num_channels),
    (width, width, num_channels, batch_size),
    (width, width, batch_size, num_channels),
]
test_parameters = zip(test_names, image_formats, batch_axes, input_shapes)


class DataUtilsTest(parameterized.TestCase):
  """Unit tests for datasets.py."""

  @parameterized.named_parameters(*test_parameters)
  def test_padding(self, image_format, batch_axis, input_shape):
    """Test that the shape is the expected padded shape."""
    batch = {'inputs': np.ones(input_shape)}
    padded_batch = data_utils.maybe_pad_batch(
        batch, desired_batch_size, image_format)
    expected_shapes = list(input_shape)
    expected_shapes[batch_axis] = desired_batch_size
    self.assertEqual(padded_batch['inputs'].shape, tuple(expected_shapes))
    self.assertEqual(padded_batch['weights'].shape, (desired_batch_size,))

  def test_padding_seq2seq(self):
    """Test padding for sequence-to-sequence models."""
    input_len_max = 25
    input_len_true = 22  # true input_seq_length for each example in batch.
    target_len_max = 25
    target_len_true = 21  # true target_seq_length for each example in batch.

    inputs_shape = (batch_size, input_len_max)
    targets_shape = (batch_size, target_len_max)
    batch = {'inputs': np.ones(inputs_shape), 'targets': np.ones(targets_shape)}
    batch['inputs'][:, input_len_true:] = 0  # zero-pad extra inputs tokens
    batch['targets'][:, target_len_true:] = 0  # zero-pad extra targets tokens
    expected_inputs_shape = (desired_batch_size, input_len_max)
    expected_targets_shape = (desired_batch_size, target_len_max)
    expected_weights_shape = (desired_batch_size, target_len_max)
    padded_batch = data_utils.maybe_pad_batch(
        batch, desired_batch_size, data_format=None, mask_key='targets')
    self.assertEqual(padded_batch['inputs'].shape, expected_inputs_shape)
    self.assertEqual(padded_batch['targets'].shape, expected_targets_shape)
    self.assertEqual(padded_batch['weights'].shape, expected_weights_shape)

    batch_pad = desired_batch_size - batch_size
    expected_weights_array = np.ones((desired_batch_size, target_len_max))
    # pad at batch axis
    expected_weights_array[-batch_pad:] = 0
    # # pad at sequence_len axis
    expected_weights_array[:, target_len_true:] = 0
    self.assertTrue(
        np.array_equal(padded_batch['weights'], expected_weights_array))


if __name__ == '__main__':
  absltest.main()
