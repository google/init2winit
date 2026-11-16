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

"""Tests for init2winit.dataset_lib.librispeech_input_pipeline."""

from absl.testing import absltest
from init2winit.dataset_lib import librispeech_input_pipeline
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf


class LibrispeechInputPipelineTest(absltest.TestCase):
  """Tests for the LibriSpeech input pipeline."""

  def test_preprocess_data_paddings(self):
    """Verifies padded shapes, lengths, and masks in preprocess_data."""
    examples = [
        {'inputs': np.array([10, 20], np.int32), 'targets': b'A'},
        {'inputs': np.array([1, 2, 3, 4, 5], np.int32), 'targets': b'ABC'},
        {'inputs': np.array([1, 2, 3, 4, 5, 6], np.int32), 'targets': b'A'},
        {'inputs': np.array([7, 8], np.int32), 'targets': b'ABCD'},
        {'inputs': np.array([30, 40, 50], np.int32), 'targets': b'AB'},
    ]
    sig = {
        'inputs': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'targets': tf.TensorSpec(shape=(), dtype=tf.string),
    }
    raw_ds = tf.data.Dataset.from_generator(
        lambda: iter(examples), output_signature=sig
    )
    hps = config_dict.ConfigDict({
        'tokenizer_type': 'RAW',
        'max_input_length': 5,
        'max_target_length': 5,
    })
    ds = librispeech_input_pipeline.preprocess_data(
        raw_ds,
        train=False,
        hps=hps,
        batch_size=2,
        drop_remainder=False,
        prefetch_size=0,
    )
    batches = list(ds.as_numpy_iterator())
    self.assertLen(batches, 2)
    self.assertSetEqual(
        set(batches[0].keys()),
        {'inputs', 'targets', 'input_paddings', 'target_paddings'},
    )
    self.assertEqual(batches[0]['input_paddings'].dtype, np.float32)
    np.testing.assert_array_equal(
        batches[0]['inputs'], [[10, 20, 0, 0, 0], [1, 2, 3, 4, 5]]
    )
    np.testing.assert_array_equal(
        batches[0]['input_paddings'],
        [[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        batches[0]['targets'], [[1, 6, 2, 0, 0], [1, 6, 7, 8, 2]]
    )
    np.testing.assert_array_equal(
        batches[0]['target_paddings'],
        [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(batches[1]['inputs'], [[30, 40, 50, 0, 0]])
    np.testing.assert_array_equal(
        batches[1]['input_paddings'], [[0.0, 0.0, 0.0, 1.0, 1.0]]
    )
    np.testing.assert_array_equal(batches[1]['targets'], [[1, 6, 7, 2, 0]])
    np.testing.assert_array_equal(
        batches[1]['target_paddings'], [[0.0, 0.0, 0.0, 0.0, 1.0]]
    )


if __name__ == '__main__':
  absltest.main()
