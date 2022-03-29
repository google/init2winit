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
import itertools

from absl import logging
from init2winit.dataset_lib import autoaugment
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import image_preprocessing
import jax
from jax.experimental import multihost_utils
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(224, 224, 3),
    output_shape=(1000,),
    train_size=1281167,
    valid_size=50000,
    use_inception_crop=False,
    use_mixup=False,
    mixup={'alpha': 0.5},
    use_randaug=False,
    randaug={
        'magnitude': 15,
        'num_layers': 2
    }))

    # pylint:disable=raise-missing-from
METADATA = {
    'apply_one_hot_in_loss': False,
}


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
  Returns:
    cropped image `Tensor`
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
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


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
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
    size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
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
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes,
                         key,
                         dtype=tf.float32,
                         image_size=224,
                         hps=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    key: tf random key.
    dtype: data type of the image.
    image_size: image size.
    hps: ConfigDict of options.

  Returns:
    A preprocessed image `Tensor`.
  """
  if hps.get('use_inception_crop'):
    image = _decode_and_inception_crop(image_bytes, image_size)
  else:
    image = _decode_and_random_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.random_flip_left_right(image)

  if hps.get('use_randaug'):
    # NOTE(dsuo): autoaugment code expects uint8 image; not sure why we use
    # float32[0, 255], but just making sure pipeline runs.
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = autoaugment.distort_image_with_randaugment(image,
                                                       hps.randaug.num_layers,
                                                       hps.randaug.magnitude,
                                                       key)
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


# TODO(gilmer,gdahl,znado): Fix eval metrics to include final partial batch!
def load_split(
    per_host_batch_size,
    split,
    hps,
    dtype=tf.float32,
    image_size=224,
    shuffle_rng=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  The dataset returned by this function will repeat forever if split == 'train',
  else it will be finite.
  For the training set, we drop the last partial batch. This is fine to do
  because we additionally shuffle the data randomly each epoch, thus the trainer
  will see all data in expectation. For the validation set, we pad the final
  batch to the desired batch size.

  Args:
    per_host_batch_size: the batch size returned by the data pipeline.
    split: One of ['train', 'eval_train', 'valid'].
    hps: The hparams the experiment is run with. Required fields are train_size
      and valid_size.
    dtype: data type of the image.
    image_size: The target size of the images.
    shuffle_rng: The RNG used to shuffle the split. Only used if
      `split == 'train'`.
  Returns:
    A `tf.data.Dataset`.
  """
  if split not in ['train', 'eval_train', 'valid']:
    raise ValueError('Unrecognized split {}'.format(split))
  if split in ['train']:
    split_size = hps.train_size // jax.process_count()
  else:
    split_size = hps.valid_size // jax.process_count()
  start = jax.process_index() * split_size
  end = start + split_size
  # In order to properly load the full dataset, it is important that we load
  # entirely to the end of it on the last host, because otherwise we will drop
  # the last `{train,valid}_size % split_size` elements.
  if jax.process_index() == jax.process_count() - 1:
    end = -1

  logging.info('Loaded data [%d: %d] from %s', start, end, split)
  if split in ['train', 'eval_train']:
    tfds_split = 'train[{}:{}]'.format(start, end)
  else:  # split == 'valid':
    tfds_split = 'validation[{}:{}]'.format(start, end)

  def decode_example(example_index, example):
    # TODO(znado): make pre-processing deterministic.
    if split == 'train':
      # NOTE(dsuo): using fold_in so as not to disturb shuffle_rng.
      # preprocess_rng is different for each example.
      preprocess_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(shuffle_rng, tf.int64), example_index)
      image = preprocess_for_train(example['image'], preprocess_rng, dtype,
                                   image_size, hps)

    else:
      image = preprocess_for_eval(example['image'], dtype, image_size)
    return {
        'image': image,
        'label': example['label'],
        'example_key': example['tfds_id'],
    }

  # TODO(znado): make shuffling the input files deterministic, as this will
  # yield a different order each time we are pre-empted.
  read_config = tfds.ReadConfig(add_tfds_id=True)
  ds = tfds.load(
      'imagenet2012:5.*.*',
      split=tfds_split,
      shuffle_files=True,
      read_config=read_config,
      decoders={
          'image': tfds.decode.SkipDecoding(),
      })

  ds = ds.cache()

  if split == 'train':
    ds = ds.shuffle(
        16 * per_host_batch_size,
        seed=shuffle_rng[0],
        reshuffle_each_iteration=True)
    ds = ds.repeat()

  ds = ds.enumerate().map(
      decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(per_host_batch_size, drop_remainder=False)

  if split != 'train':
    ds = ds.cache()
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds


def get_imagenet(shuffle_rng,
                 batch_size,
                 eval_batch_size,
                 hps):
  """Data generators for imagenet."""
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  image_size = hps.input_shape[0]
  num_classes = 1000

  # TODO(gilmer) Currently the training data is not determistic.
  logging.info('Loading train split')
  train_ds = load_split(
      per_host_batch_size,
      'train',
      hps=hps,
      image_size=image_size,
      shuffle_rng=shuffle_rng)
  train_ds = tfds.as_numpy(train_ds)
  logging.info('Loading eval_train split')
  eval_train_ds = load_split(
      per_host_eval_batch_size, 'eval_train', hps=hps, image_size=image_size)
  eval_train_ds = tfds.as_numpy(eval_train_ds)
  logging.info('Loading eval split')
  eval_ds = load_split(
      per_host_eval_batch_size, 'valid', hps=hps, image_size=image_size)
  eval_ds = tfds.as_numpy(eval_ds)

  def train_iterator_fn():
    if hps.use_mixup:
      # NOTE(dsuo): using `fold_in` so as not to disturb shuffle_rng.
      mixup_rng = jax.random.fold_in(shuffle_rng, jax.process_index())
      mixup_rng = multihost_utils.broadcast_one_to_all(
          mixup_rng, is_source=jax.process_index() == 0)

    for batch in iter(train_ds):
      image = batch['image']
      targets = np.eye(num_classes)[batch['label']]
      if hps.use_mixup:
        mixup_rng = jax.random.fold_in(mixup_rng, 0)
        (image, targets), _ = image_preprocessing.mixup_general(
            mixup_rng,
            image,
            targets,
            alpha=hps.mixup.alpha,
            n=2)
      yield {
          'inputs': image,
          'targets': targets,
          'weights': np.ones(per_host_batch_size, dtype=image.dtype)
      }

  def eval_train_epoch(num_batches=None):
    # This uses per_host_batch_size and not per_host_eval_batch_size.
    eval_train_iter = iter(eval_train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      batch_dict = {
          'inputs': batch['image'],
          'targets': np.eye(num_classes)[batch['label']],
      }
      if hps.get('include_example_keys'):
        batch_dict['example_key'] = batch['example_key']
      yield data_utils.maybe_pad_batch(batch_dict, per_host_eval_batch_size)

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch in itertools.islice(valid_iter, num_batches):
      batch_dict = {
          'inputs': batch['image'],
          'targets': np.eye(num_classes)[batch['label']],
      }
      if hps.get('include_example_keys'):
        batch_dict['example_key'] = batch['example_key']
      yield data_utils.maybe_pad_batch(batch_dict, per_host_eval_batch_size)

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  return data_utils.Dataset(
      train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
