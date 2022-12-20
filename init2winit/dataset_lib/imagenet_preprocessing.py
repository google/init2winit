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

"""ImageNet input pipeline with resnet preprocessing."""

from collections import abc

from init2winit.dataset_lib import autoaugment
import tensorflow.compat.v2 as tf

CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                rng_seed,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    rng_seed: `Tensor` random seed.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.

  Returns:
    cropped image `Tensor`
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.stateless_sample_distorted_bounding_box(
      shape,
      seed=rng_seed,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size, rng_seed):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      rng_seed=rng_seed,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad, lambda: _decode_and_center_crop(image_bytes, image_size),
                  lambda: _resize(image, image_size))

  return image


def maybe_repeat(arg, n_reps):
  if not isinstance(arg, abc.Sequence):
    arg = (arg,) * n_reps
  return arg


def resize(image, size, method='bilinear'):
  """Resizes image to a given size.

  Args:
    image:
    size: either an integer H, where H is both the new height and width of the
      resized image, or a list or tuple [H, W] of integers, where H and W are
      new image"s height and width respectively.
    method: rezied method, see tf.image.resize docs for options.

  Returns:
    A function for resizing an image.
  """
  size = maybe_repeat(size, 2)

  # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
  # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
  # In particular it was not equivariant with rotation and lead to the network
  # to learn a shortcut in self-supervised rotation task, if rotation was
  # applied after resize.
  dtype = image.dtype
  image = tf.image.resize(image, size, method)
  return tf.cast(image, dtype)


def _decode_and_inception_crop(image_data,
                               size,
                               area_min=5,
                               area_max=100,
                               method='bilinear'):
  """Decode jpeg and add inception crop.

  Args:
    image_data:
    size:
    area_min:
    area_max:
    method:

  Returns:
    tf.Tensor
  """
  shape = tf.image.extract_jpeg_shape(image_data)
  begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
      shape,
      tf.zeros([0, 0, 4], tf.float32),
      area_range=(area_min / 100, area_max / 100),
      min_object_covered=0,  # Don't enforce a minimum area.
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(begin)
  target_height, target_width, _ = tf.unstack(crop_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)
  image = resize(image, size, method)

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(hps,
                         image_bytes,
                         rng_seed,
                         dtype=tf.float32,
                         image_size=224):
  """Preprocesses the given image for training.

  Args:
    hps: ConfigDict of options.
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    rng_seed: `Tensor` random seed.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  crop_rng, flip_rng, randaug_rng = tf.unstack(tf.random.split(rng_seed, 3))
  if hps.crop == 'random':
    image = _decode_and_random_crop(image_bytes, image_size, crop_rng)
  elif hps.crop == 'inception':
    image = _decode_and_inception_crop(image_bytes, image_size)
  elif hps.crop == 'center':
    image = _decode_and_center_crop(image_bytes, image_size)
  else:
    raise ValueError(f'{hps.crop} is not a valid crop strategy')

  image = tf.reshape(image, [image_size, image_size, 3])
  if hps['random_flip']:
    image = tf.image.stateless_random_flip_left_right(image, seed=flip_rng)

  if hps.get('use_randaug'):
    # NOTE(dsuo): autoaugment code expects uint8 image; not sure why we use
    # float32[0, 255], but just making sure pipeline runs.
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = autoaugment.distort_image_with_randaugment(image,
                                                       hps.randaug.num_layers,
                                                       hps.randaug.magnitude,
                                                       randaug_rng)

  image = tf.cast(image, tf.float32)
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=224):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image
