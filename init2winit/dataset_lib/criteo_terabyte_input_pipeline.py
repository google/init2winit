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

"""Data loader for pre-processed Criteo data."""

from typing import List

from absl import logging
from flax import struct
import tensorflow as tf


@struct.dataclass
class CriteoFeatureConfig:
  num_dense_features: int
  vocab_sizes: List[int]
  label_feature_name: str


@struct.dataclass
class CriteoDatasetParams:
  batch_size: int
  dataset_index: int
  dataset_num_shards: int


class CriteoTsvReader(object):
  """Input reader fn for pre-processed Criteo data.

  Raw Criteo data is assumed to be preprocessed in the following way:
  1. Missing values are replaced with zeros.
  2. Negative values are replaced with zeros.
  3. Integer features are transformed by log(x+1) and are hence tf.float32.
  4. Categorical data is bucketized and are hence tf.int32.
  """

  def __init__(self,
               file_path=None,
               feature_config=None,
               is_training=True,
               distributed_eval=False,
               parallelism=1,
               num_datasets=None,
               params=None):
    self._file_path = file_path
    self._feature_config = feature_config
    self._is_training = is_training
    self._distributed_eval = distributed_eval
    self._parallelism = parallelism
    self._params = params

    num_files = len(tf.io.gfile.glob(file_path))
    if num_datasets > num_files:
      raise ValueError('There must be at least one file per dataset.')
    if num_datasets * parallelism > num_files:
      logging.warning(
          'Reducing dataset parallelism due to insufficient number of files.')
      self._parallelism = num_files // num_datasets

  def __call__(self, params):
    batch_size = params.batch_size
    fc = self._feature_config

    @tf.function
    def _parse_example_fn(example):
      """Parser function for pre-processed Criteo TSV records."""
      label_defaults = [[0.0]]
      int_defaults = [[0.0] for _ in range(fc.num_dense_features)]
      categorical_defaults = [[0] for _ in range(len(fc.vocab_sizes))]
      record_defaults = label_defaults + int_defaults + categorical_defaults
      fields = tf.io.decode_csv(
          example, record_defaults, field_delim='\t', na_value='-1')

      num_labels = 1
      num_dense = len(int_defaults)
      features = {}
      features[fc.label_feature_name] = tf.reshape(fields[0], [batch_size, 1])

      int_features = []
      for idx in range(num_dense):
        int_features.append(fields[idx + num_labels])
      features['int-features'] = tf.stack(int_features, axis=1)

      cat_features = []
      for idx in range(len(fc.vocab_sizes)):
        cat_features.append(
            tf.cast(fields[idx + num_dense + num_labels], dtype=tf.int32))
      features['cat-features'] = tf.stack(cat_features, axis=1)
      return features

    filenames = tf.data.Dataset.list_files(self._file_path, shuffle=False)
    filenames = filenames.shard(params.dataset_num_shards, params.dataset_index)

    def make_dataset(ds_index):
      ds = filenames.shard(self._parallelism, ds_index)
      ds = ds.repeat(2)
      ds = ds.interleave(
          tf.data.TextLineDataset,
          cycle_length=16,
          block_length=batch_size // 8,
          num_parallel_calls=8,
          deterministic=False)
      ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.map(_parse_example_fn, num_parallel_calls=16)
      return ds

    ds_indices = tf.data.Dataset.range(self._parallelism)
    ds = ds_indices.interleave(
        make_dataset,
        cycle_length=self._parallelism,
        block_length=1,
        num_parallel_calls=self._parallelism,
        deterministic=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
