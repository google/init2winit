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

"""Efficient ImageNet input pipeline using tf.data.Dataset."""

import os

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import imagenet_preprocessing

import jax
import tensorflow as tf

EVAL_IMAGES = 50000
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def load_split(batch_size,
               split,
               dtype,
               rng,
               data_dir=None,
               image_size=224,
               shuffle_size=16384):
  """Returns the input_fn.

  Args:
    batch_size: the batch size to use for `dataset.batch()`.
    split: a string representing the dataset split to use, either 'train',
      'eval_train', or 'validation'.
    dtype: the dtype of the image in the data pipeline.
    rng: RNG seed that is split into a shuffle seed and a seed that is folded
      into a per-example seed.
    data_dir: an optional string to read the ImageNet TFRecord files from.
    image_size: the size to resize the images to using `tf.image.resize(...,
      method='bicubic')`.
    shuffle_size: the size of the shuffle buffer used in `dataset.shuffler()`.
  Returns: a tf.data.Dataset that is batched and preprocessed, and optionally
    shuffled and repeated, for ImageNet based off the MLPerf codebase. Note that
    for evaluation, the final partial batches are not yet padded to be the same
    shape, so callers should also call `data_utils.maybe_pad_batch(batch,
    eval_host_batch_size)` to pad them.
  """
  if split not in ['train', 'eval_train', 'validation']:
    raise ValueError('Invalid split name {}.'.format(split))

  preprocess_rng, shuffle_rng = jax.random.split(rng, 2)


  def dataset_parser(*value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      *value: a tuple where the last element is always the tfrecord to be
        parsed, and there is an optional first element that is the index into
        the (infinite) dataset of the current example, which is used for
        deterministic training.

    Returns:
      A tuple of (image, dense label) tensors.
    """
    if len(value) > 1:
      [example_index, value] = value
      per_example_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(preprocess_rng, tf.int64), example_index)
    elif split == 'train':
      raise ValueError(
          'Must enumerate() over tf.data.Dataset when training in order to get '
          'a per-example index to fold into a per-example seed.')
    else:
      value = value[0]
      per_example_rng = None
    parsed = tf.io.parse_single_example(
        value, {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64, 0)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1

    if split == 'train':
      image = imagenet_preprocessing.preprocess_for_train(
          image_bytes,
          per_example_rng,
          dtype,
          image_size,
          crop='random',
          random_crop_area_range=(0.05, 1.0),
          use_center_crop_if_random_failed=False,
          random_flip=True)
    else:
      image = imagenet_preprocessing.preprocess_for_eval(
          image_bytes, dtype, image_size)

    return {
        'inputs': image,
        'targets': tf.one_hot(label, imagenet_preprocessing.NUM_CLASSES),
    }

  index = jax.process_index()
  num_hosts = jax.process_count()
  use_training_files = split in ['train', 'eval_train']
  file_pattern = os.path.join(
      data_dir, 'train-*' if use_training_files else 'validation-*')
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  dataset = dataset.shard(num_hosts, index)
  concurrent_files = min(10, 1024 // num_hosts)
  dataset = dataset.interleave(tf.data.TFRecordDataset, concurrent_files, 1,
                               concurrent_files)

  if split == 'train':
    dataset = dataset.cache()  # cache compressed JPEGs instead
    dataset = dataset.shuffle(
        shuffle_size, reshuffle_each_iteration=True,
        seed=data_utils.convert_jax_to_tf_random_seed(shuffle_rng)).repeat()
    dataset = dataset.enumerate().map(dataset_parser, 64)
  else:
    dataset = dataset.map(dataset_parser, 64)
  dataset = dataset.batch(batch_size, drop_remainder=False)

  if split != 'train':
    dataset = dataset.cache()

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  options = tf.data.Options()
  options.experimental_deterministic = True
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_threading.private_threadpool_size = 48
  dataset = dataset.with_options(options)
  return dataset
