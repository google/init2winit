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

"""FastMRI knee singlecoil input pipeline."""

import datetime
import itertools
import os

import h5py
from init2winit.dataset_lib import data_utils
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf
import tensorflow_datasets as tfds


DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    input_shape=(320, 320),
    output_shape=(320, 320),
    data_dir='',
    train_size=34742,
    num_train_h5_files=973,
    train_dir='knee_singlecoil_train',
    valid_size=7135,
    num_valid_h5_files=199,
    val_dir='knee_singlecoil_val',
))


METADATA = {
    'apply_one_hot_in_loss': False,
}


def _process_example(kspace, kspace_shape, target, target_shape, volume_max,
                     seed):
  """Generate a single example (slice from mri image).

  Args:
    kspace: raw mri data.
    kspace_shape: shape of kspace. We pass this in because it is not constant.
    target: target image.
    target_shape: shape of target.
    volume_max: max value over the entire volume that the example slice was
      originally derived from.
    seed: seed for stateless randomness.

  Returns:
    dictionary of processed image/target.
  """

  # sample_mask
  num_cols = kspace_shape[1]
  num_cols_float = tf.cast(num_cols, dtype=tf.float32)

  # choose_acceleration
  center_fraction = tf.convert_to_tensor(0.08, dtype=tf.float32)
  acceleration = tf.convert_to_tensor(4.0, dtype=tf.float32)

  num_low_frequencies = tf.cast(
      num_cols_float * center_fraction, dtype=tf.int32)

  # calculate_center_mask
  mask = tf.zeros(num_cols, dtype=tf.float32)
  pad = (num_cols - num_low_frequencies + 1) // 2
  mask = tf.tensor_scatter_nd_update(
      mask, tf.reshape(tf.range(pad, pad + num_low_frequencies), (-1, 1)),
      tf.ones(num_low_frequencies))

  # reshape_mask
  center_mask = tf.reshape(mask, (1, num_cols))

  # calculate_acceleration_mask
  num_low_frequencies_float = tf.cast(num_low_frequencies, dtype=tf.float32)
  prob = (num_cols_float / acceleration - num_low_frequencies_float) / (
      num_cols_float - num_low_frequencies_float
  )

  mask = tf.cast(
      tf.random.stateless_uniform((num_cols,), seed) < prob,
      dtype=tf.float32)
  acceleration_mask = tf.reshape(mask, (1, num_cols))

  mask = tf.math.maximum(center_mask, acceleration_mask)
  mask = tf.cast(mask, dtype=tf.complex64)

  # apply_mask
  masked_kspace = kspace * mask + 0.0

  # ifft2c
  shifted_kspace = tf.signal.ifftshift(masked_kspace, axes=(0, 1))
  shifted_image = tf.signal.ifft2d(shifted_kspace)
  image = tf.signal.fftshift(shifted_image, axes=(0, 1))
  scaling_norm = tf.cast(
      tf.math.sqrt(
          tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')),
      kspace.dtype)
  image = image * scaling_norm
  image = tf.stack((tf.math.real(image), tf.math.imag(image)), axis=-1)

  # complex_center_crop
  w_from = (kspace_shape[0] - target_shape[0]) // 2
  h_from = (kspace_shape[1] - target_shape[1]) // 2
  w_to = w_from + target_shape[0]
  h_to = h_from + target_shape[1]

  image = image[..., w_from:w_to, h_from:h_to, :]

  # complex_abs
  abs_image = tf.math.sqrt(tf.math.reduce_sum(image ** 2, axis=-1))

  # normalize_instance
  mean = tf.math.reduce_mean(abs_image)
  std = tf.math.reduce_std(abs_image)
  norm_image = (abs_image - mean) / std

  # clip_image
  image = tf.clip_by_value(norm_image, -6, 6)

  # process target
  norm_target = (target - mean) / std
  target = tf.clip_by_value(norm_target, -6, 6)

  return {'inputs': image, 'targets': target, 'volume_max': volume_max}


def _h5_to_examples(path):
  """Yield MRI slices from an hdf5 file containing a single MRI volume."""
  tf.print('fastmri_dataset._h5_to_examples call:', path,
           datetime.datetime.now().strftime('%H:%M:%S:%f'))
  with tf.io.gfile.GFile(path, 'rb') as gf:
    path = gf
  with h5py.File(path, 'r') as hf:
    # NOTE(dsuo): logic taken from reference code
    volume_max = hf.attrs.get('max', 0.0)

    for i in range(hf['kspace'].shape[0]):
      yield hf['kspace'][i], hf['kspace'][i].shape, hf['reconstruction_esc'][
          i], hf['reconstruction_esc'][i].shape, volume_max


def _create_generator(filename):
  signature = (
      tf.TensorSpec(shape=(640, None), dtype=tf.complex64),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(320, 320), dtype=tf.float32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(), dtype=tf.float32),
  )
  return tf.data.Dataset.from_generator(
      _h5_to_examples, args=(filename,), output_signature=signature)


def load_split(per_host_batch_size, split, hps, shuffle_rng=None):
  """Creates a split from the FastMRI dataset using tfds.

  NOTE: only creates knee singlecoil datasets.

  Args:
    per_host_batch_size: the batch size returned by the data pipeline.
    split: One of ['train', 'eval_train', 'val'].
    hps: The hparams the experiment is run with. Required fields are train_size
      and valid_size.
    shuffle_rng: The RNG used to shuffle the split.
  Returns:
    A `tf.data.Dataset`.
  """
  if split not in ['train', 'eval_train', 'val']:
    raise ValueError('Unrecognized split {}'.format(split))

  # NOTE(dsuo): we split on h5 files, but each h5 file has some number of slices
  # that each represent an example. h5 files have approximately the same number
  # of slices, so we just split files equally among hosts.
  if split in ['train']:
    split_size = hps.num_train_h5_files // jax.process_count()
  else:
    split_size = hps.num_valid_h5_files // jax.process_count()
  start = jax.process_index() * split_size
  end = start + split_size
  # In order to properly load the full dataset, it is important that we load
  # entirely to the end of it on the last host, because otherwise we will drop
  # the last `{train,valid}_size % split_size` elements.
  if jax.process_index() == jax.process_count() - 1:
    end = -1

  data_dir = hps.data_dir

  if split in ['train', 'eval_train']:
    data_dir = os.path.join(data_dir, hps.train_dir)
  else:  # split == 'val'
    data_dir = os.path.join(data_dir, hps.val_dir)

  h5_paths = [
      os.path.join(data_dir, path) for path in tf.io.gfile.listdir(data_dir)
  ][start:end]

  ds = tf.data.Dataset.from_tensor_slices(h5_paths)
  ds = ds.interleave(
      _create_generator,
      cycle_length=32,
      block_length=64,
      num_parallel_calls=hps.num_tf_data_map_parallel_calls)
  ds = ds.cache()

  def process_example(example_index, example):
    process_rng = tf.cast(jax.random.fold_in(shuffle_rng, 0), tf.int64)
    if split == 'train':
      # NOTE(dsuo): we use fixed randomness for eval.
      process_rng = tf.random.experimental.stateless_fold_in(
          process_rng, example_index)
    return _process_example(*example, process_rng)

  ds = ds.enumerate().map(
      process_example, num_parallel_calls=hps.num_tf_data_map_parallel_calls)

  if split == 'train':
    ds = ds.shuffle(
        16 * per_host_batch_size,
        seed=shuffle_rng[0],
        reshuffle_each_iteration=True)
    ds = ds.repeat()

  ds = ds.batch(per_host_batch_size, drop_remainder=False)

  if split != 'train':
    ds = ds.cache()
  ds = ds.prefetch(hps.num_tf_data_prefetches)

  return ds


def get_fastmri(shuffle_rng, batch_size, eval_batch_size, hps):
  """FastMRI dataset.

  Args:
    shuffle_rng: rng for shuffling.
    batch_size: batch size.
    eval_batch_size: batch size for eval.
    hps: hyperparameters.

  Returns:
    An init2winit Dataset.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  train_ds = load_split(per_host_batch_size, 'train', hps, shuffle_rng)
  train_ds = tfds.as_numpy(train_ds)

  # NOTE(dsuo): fastMRI has fixed randomness for eval.
  eval_train_ds = load_split(per_host_eval_batch_size, 'eval_train', hps,
                             shuffle_rng)
  eval_train_ds = tfds.as_numpy(eval_train_ds)
  eval_ds = load_split(per_host_eval_batch_size, 'val', hps, shuffle_rng)
  eval_ds = tfds.as_numpy(eval_ds)

  def train_iterator_fn():
    return train_ds

  def eval_train_epoch(num_batches=None):
    for batch in itertools.islice(eval_train_ds, num_batches):
      yield data_utils.maybe_pad_batch(batch, per_host_eval_batch_size)

  def valid_epoch(num_batches=None):
    for batch in itertools.islice(eval_ds, num_batches):
      yield data_utils.maybe_pad_batch(batch, per_host_eval_batch_size)

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  return data_utils.Dataset(train_iterator_fn, eval_train_epoch, valid_epoch,
                            test_epoch)
