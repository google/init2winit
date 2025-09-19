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

"""Tests for init2winit.dataset_lib.fineweb_edu_10b_input_pipeline."""

from absl.testing import absltest
from init2winit.dataset_lib import fineweb_edu_10b_input_pipeline
import numpy as np
import tensorflow as tf


class FinewebEdu10bInputPipelineTest(absltest.TestCase):
  """Unit tests."""

  def test_batch_with_padding(self):
    """Test batching with padding."""
    arr = np.arange(18, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices(arr)
    ds = ds.batch(
        6,
    )  # sequences of length 6
    ds = fineweb_edu_10b_input_pipeline.batch_with_padding(
        ds, 4, padded_shapes=(4, None), padding_id=-1
    )

    padded_batch = list(ds.as_numpy_iterator())[0]

    padded_batch_expected = np.array([
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [-1, -1, -1, -1, -1, -1],
    ])

    self.assertTrue(np.array_equal(padded_batch, padded_batch_expected))

if __name__ == '__main__':
  absltest.main()
