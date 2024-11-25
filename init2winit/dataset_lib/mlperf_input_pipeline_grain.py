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

"""Efficient ImageNet input pipeline using tf.data.Dataset."""

import dataclasses
from typing import Any, Sequence

from clu import preprocess_spec
from grain._src.tensorflow import transforms
import grain.tensorflow as grain
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

EVAL_IMAGES = 50000
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
NUM_CLASSES = 1000

FlatFeatures = preprocess_spec.FlatFeatures


def transpose_and_normalize_image(image):
  mean = tf.constant([[MEAN_RGB]], dtype=image.dtype)
  stddev = tf.constant([[STDDEV_RGB]], dtype=image.dtype)
  image -= mean
  image /= stddev
  return image


# Note that this will run before batching.
@dataclasses.dataclass(frozen=True)
class NormalizeAndOneHot(preprocess_spec.MapTransform):

  def _transform(self, features: FlatFeatures) -> FlatFeatures:
    features['inputs'] = transpose_and_normalize_image(features['image'])
    features['targets'] = tf.one_hot(features['label'], NUM_CLASSES)
    del features['label']
    del features['image']
    return features


@dataclasses.dataclass(frozen=True)
class DecodeRandomCropAndResize(preprocess_spec.RandomMapTransform):
  """Decodes the images and extracts a random crop."""

  resize_size: int

  def _transform(self, features: FlatFeatures, seed: tf.Tensor) -> FlatFeatures:
    image = features['image']
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.stateless_sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image),
        bbox,
        seed=seed,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)

    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    image = tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)
    image = tf.image.resize([image], [self.resize_size, self.resize_size],
                            method='bicubic')[0]
    features['image'] = image
    return features


@dataclasses.dataclass(frozen=True)
class CentralCropAndResize(preprocess_spec.MapTransform):
  """Makes a central crop of a given size."""

  resize_size: int

  def _transform(self, features: FlatFeatures) -> FlatFeatures:
    image = features['image']
    shape = tf.image.extract_jpeg_shape(image)
    crop_size = tf.cast(
        ((self.resize_size / (self.resize_size + CROP_PADDING)) *
         tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)), tf.int32)
    offset_y, offset_x = [((shape[i] - crop_size) + 1) // 2 for i in range(2)]
    crop_window = tf.stack([offset_y, offset_x, crop_size, crop_size])
    image = tf.image.decode_and_crop_jpeg(image, crop_window, channels=3)
    image = tf.image.resize([image], [self.resize_size, self.resize_size],
                            method='bicubic')[0]
    features['image'] = image
    return features


@dataclasses.dataclass(frozen=True)
class RandomFlipLeftRight(preprocess_spec.RandomMapTransform):

  def _transform(self, features: FlatFeatures, seed: tf.Tensor) -> FlatFeatures:
    features['image'] = tf.image.stateless_random_flip_left_right(
        features['image'], seed)
    return features


@dataclasses.dataclass(frozen=True)
class ReshapeAndConvertDtype(preprocess_spec.MapTransform):
  """Reshapes and converts to a given dtype."""

  resize_size: int
  dtype: Any

  def _transform(self, features: FlatFeatures) -> FlatFeatures:
    image = features['image']
    image = tf.reshape(image, [self.resize_size, self.resize_size, 3])
    image = tf.image.convert_image_dtype(image, self.dtype)
    features['image'] = image
    return features


@dataclasses.dataclass(frozen=True)
class DropFeatures(preprocess_spec.MapTransform):

  feature_names: Sequence[str]

  def _transform(self, features: FlatFeatures) -> FlatFeatures:
    return {k: v for k, v in features.items() if k not in self.feature_names}


def load_split(batch_size,
               split,
               dtype,
               rng=None,
               data_dir=None,
               image_size=224,
               preprocess_transform=None,
               is_train=False):
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
    preprocess_transform: a function that takes two arguments (a single image
      and a single label) and returns a dict with keys 'inputs' and 'labels'.
    is_train: True if loading the train dataset which will be infinite,
      so it can't be cached.
  Returns: a tf.data.Dataset that is batched and preprocessed, and optionally
    shuffled and repeated, for ImageNet based off the MLPerf codebase. Note that
    for evaluation, the final partial batches are not yet padded to be the same
    shape, so callers should also call `data_utils.maybe_pad_batch(batch,
    eval_host_batch_size)` to pad them.
  """
  if split not in ['train', 'validation', 'test']:
    raise ValueError('Invalid split name {}.'.format(split))

  initial_step = 1

  grain.config.update('tf_interleaved_shuffle', True)

  # The init2winit convention is to specify batch_size per CPU host
  global_batch_size = jax.process_count() * batch_size
  if split == 'train':
    start_index = (initial_step - 1) * global_batch_size + jax.process_index()
  else:
    start_index = jax.process_index()

  if split == 'train':
    # Tell TFDS to not decode the image as we combined it with the random crop.
    decoders = {'image': tfds.decode.SkipDecoding()}
    transformations = [] if is_train else [transforms.CacheTransform()]
    transformations += [
        DecodeRandomCropAndResize(resize_size=image_size),
        RandomFlipLeftRight(),
        ReshapeAndConvertDtype(resize_size=image_size, dtype=dtype),
        DropFeatures(('file_name',))
    ]
  else:
    decoders = {'image': tfds.decode.SkipDecoding()}
    transformations = [
        CentralCropAndResize(224),
        ReshapeAndConvertDtype(resize_size=image_size, dtype=dtype),
        DropFeatures(('file_name',)),
        transforms.CacheTransform()
    ]

  if preprocess_transform is not None:
    transformations.append(preprocess_transform)

  data_dir = tfds.core.constants.ARRAY_RECORD_DATA_DIR

  loader = grain.load_from_tfds(
      name='imagenet2012',
      data_dir=data_dir,
      split=split,
      shuffle=True,
      seed=rng,
      shard_options=grain.ShardByJaxProcess(drop_remainder=split == 'train'),
      decoders=decoders,
      transformations=transformations,
      num_epochs=None if is_train else 1,
      batch_size=batch_size)

  dataset_iter = loader.as_dataset(start_index=start_index)
  return dataset_iter
