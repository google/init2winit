# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""Tests for init2winit.dataset_lib.small_image_datasets."""

import itertools

from absl import flags
from absl.testing import absltest
from init2winit.dataset_lib import small_image_datasets
from jax import random
from ml_collections.config_dict import config_dict

_TFRECORD_SUFFIX = 'tfrecord'
_ARRAY_RECORD_SUFFIX = 'array_record'


class SmallImageDatasetsTest(absltest.TestCase):
  """Unit tests for small_image_datasets.py."""

  def test_cifar10(self):
    """Test example generation in CIFAR10 is reproducible."""
    dataset = small_image_datasets.get_cifar10(
        random.PRNGKey(0), 1, 1,
        config_dict.ConfigDict(
            dict(
                flip_probability=0.5,
                alpha=1.0,
                crop_num_pixels=4,
                use_mixup=True,
                train_size=45000,
                valid_size=5000,
                test_size=10000,
                include_example_keys=True,
                input_shape=(32, 32, 3),
                output_shape=(10,))))

    examples = itertools.islice(dataset.valid_epoch(), 10)
    example_keys = [
        example['example_key'][0].decode('utf-8') for example in examples
    ]
    file_format = (
        _ARRAY_RECORD_SUFFIX
        if flags.FLAGS.array_record_default else _TFRECORD_SUFFIX)
    self.assertEqual(example_keys, [
        f'cifar10-train.{file_format}-00000-of-00001__45000',
        f'cifar10-train.{file_format}-00000-of-00001__45001',
        f'cifar10-train.{file_format}-00000-of-00001__45002',
        f'cifar10-train.{file_format}-00000-of-00001__45003',
        f'cifar10-train.{file_format}-00000-of-00001__45004',
        f'cifar10-train.{file_format}-00000-of-00001__45005',
        f'cifar10-train.{file_format}-00000-of-00001__45006',
        f'cifar10-train.{file_format}-00000-of-00001__45007',
        f'cifar10-train.{file_format}-00000-of-00001__45008',
        f'cifar10-train.{file_format}-00000-of-00001__45009',
    ])


if __name__ == '__main__':
  absltest.main()
