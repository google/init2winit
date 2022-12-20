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

import itertools

from absl import logging
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import image_preprocessing
from init2winit.dataset_lib import imagenet_preprocessing
import jax
from jax.experimental import multihost_utils
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        input_shape=(224, 224, 3),
        output_shape=(1000,),
        train_size=1281167,
        valid_size=50000,
        test_size=10000,  # ImageNet-v2.
        use_imagenetv2_test=True,
        crop='random',  # options are: {"random", "inception", "center"}
        random_flip=True,
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


# TODO(gilmer,gdahl,znado): Fix eval metrics to include final partial batch!
def load_split(
    per_host_batch_size,
    split,
    hps,
    dtype=tf.float32,
    image_size=224,
    shuffle_rng=None,
    tfds_dataset_name='imagenet2012:5.*.*'):  # pyformat: disable
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
    shuffle_rng: The RNG used to shuffle the split. Only used if `split ==
      'train'`.
    tfds_dataset_name: The name of the dataset to load from TFDS. Used to reuse
      this same logic for imagenet-v2.

  Returns:
    A `tf.data.Dataset`.
  """
  if split not in ['train', 'eval_train', 'valid', 'test']:
    raise ValueError('Unrecognized split {}'.format(split))
  if split in ['train', 'eval_train']:
    split_size = hps.train_size // jax.process_count()
  elif split == 'valid':
    split_size = hps.valid_size // jax.process_count()
  else:
    split_size = hps.test_size // jax.process_count()
  start = jax.process_index() * split_size
  end = start + split_size
  # In order to properly load the full dataset, it is important that we load
  # entirely to the end of it on the last host, because otherwise we will drop
  # the last `{train,valid}_size % split_size` elements.
  if jax.process_index() == jax.process_count() - 1:
    end = ''

  logging.info('Loaded data [%d: %s] from %s', start, str(end), split)
  if split in ['train', 'eval_train']:
    tfds_split = 'train'
  elif split == 'valid':
    tfds_split = 'validation'
  else:
    tfds_split = 'test'
  tfds_split += '[{}:{}]'.format(start, end)

  def decode_example(example_index, example):
    if split == 'train':
      # NOTE(dsuo): using fold_in so as not to disturb shuffle_rng.
      # preprocess_rng is different for each example.
      preprocess_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(shuffle_rng, tf.int64), example_index)
      image = imagenet_preprocessing.preprocess_for_train(
          example['image'],
          preprocess_rng,
          dtype,
          image_size,
          crop=hps.crop,
          random_flip=hps.random_flip,
          use_randaug=hps.use_randaug,
          randaug_magnitude=hps.randaug.magnitude,
          randaug_num_layers=hps.randaug.num_layers)
    else:
      image = imagenet_preprocessing.preprocess_for_eval(
          example['image'], dtype, image_size)

    example_dict = {
        'inputs':
            image,
        'targets':
            tf.one_hot(example['label'], imagenet_preprocessing.NUM_CLASSES)
    }

    if split == 'train':
      example_dict['weights'] = 1
    elif hps.get('include_example_keys'):
      example_dict['example_key'] = example['tfds_id']

    return example_dict

  read_config = tfds.ReadConfig(add_tfds_id=True)
  ds = tfds.load(
      tfds_dataset_name,
      split=tfds_split,
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
      decode_example, num_parallel_calls=hps.num_tf_data_map_parallel_calls)
  ds = ds.batch(per_host_batch_size, drop_remainder=False)

  if split == 'train' and hps.use_mixup:
    mixup_rng = tf.convert_to_tensor(shuffle_rng, dtype=tf.int32)
    mixup_rng = tf.random.experimental.stateless_fold_in(mixup_rng, 0)
    mixup_rng = multihost_utils.broadcast_one_to_all(
        mixup_rng, is_source=jax.process_index() == 0)

    def mixup_batch(batch_index, batch):
      per_batch_mixup_rng = tf.random.experimental.stateless_fold_in(
          mixup_rng, batch_index)
      (inputs, targets) = image_preprocessing.mixup_tf(
          per_batch_mixup_rng,
          batch['inputs'],
          batch['targets'],
          alpha=hps.mixup.alpha,
      )
      batch['inputs'] = inputs
      batch['targets'] = targets
      return batch

    ds = ds.enumerate().map(
        mixup_batch, num_parallel_calls=hps.num_tf_data_map_parallel_calls)

  if split != 'train':
    ds = ds.cache()
  ds = ds.prefetch(hps.num_tf_data_prefetches)
  return ds


def get_imagenet(shuffle_rng, batch_size, eval_batch_size, hps):
  """Data generators for imagenet."""
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  image_size = hps.input_shape[0]

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
  validation_ds = load_split(
      per_host_eval_batch_size, 'valid', hps=hps, image_size=image_size)
  validation_ds = tfds.as_numpy(validation_ds)

  test_ds = None
  if hps.use_imagenetv2_test:
    test_ds = load_split(
        per_host_eval_batch_size,
        'test',
        hps=hps,
        image_size=image_size,
        tfds_dataset_name='imagenet_v2/matched-frequency')
    test_ds = tfds.as_numpy(test_ds)

  def train_iterator_fn():
    return train_ds

  def eval_train_epoch(num_batches=None):
    # This uses per_host_batch_size and not per_host_eval_batch_size.
    for batch in itertools.islice(eval_train_ds, num_batches):
      yield data_utils.maybe_pad_batch(batch, per_host_eval_batch_size)

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(validation_ds, num_batches):
      yield data_utils.maybe_pad_batch(batch, per_host_eval_batch_size)

  def test_epoch(num_batches=None):
    if test_ds:
      for batch in itertools.islice(test_ds, num_batches):
        yield data_utils.maybe_pad_batch(batch, per_host_eval_batch_size)
    else:
      yield from ()

  return data_utils.Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                            test_epoch)


def get_fake_batch(hps):
  return {
      'inputs':
          np.ones((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype),
      'targets':
          np.ones((hps.batch_size, *hps.output_shape), dtype=hps.model_dtype),
      'weights':
          np.ones((hps.batch_size,), dtype=hps.model_dtype),
  }
