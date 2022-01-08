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

"""Data generators for init2winit."""

import functools
import itertools

from absl import logging
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import image_preprocessing
from init2winit.dataset_lib.data_utils import Dataset
import jax.numpy as jnp
import jax.random
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


MNIST_HPARAMS = config_dict.ConfigDict(dict(
    train_size=50000,
    valid_size=10000,
    test_size=10000,
    input_shape=(28, 28, 1),
    output_shape=(10,)))
MNIST_METADATA = {
    'apply_one_hot_in_loss': False,
}

MNIST_AUTOENCODER_HPARAMS = config_dict.ConfigDict(dict(
    train_size=50000,
    valid_size=10000,
    test_size=10000,
    input_shape=(28, 28, 1),
    output_shape=(784,)))
MNIST_AUTOENCODER_METADATA = {
    'apply_one_hot_in_loss': False,
}

FASHION_MNIST_HPARAMS = config_dict.ConfigDict(dict(
    train_size=45000,
    valid_size=5000,
    test_size=10000,
    input_shape=(28, 28, 1),
    output_shape=(10,)))
FASHION_MNIST_METADATA = {
    'apply_one_hot_in_loss': False,
}

CIFAR10_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    flip_probability=0.5,
    alpha=1.0,
    crop_num_pixels=4,
    use_mixup=True,
    train_size=45000,
    valid_size=5000,
    test_size=10000,
    input_shape=(32, 32, 3),
    output_shape=(10,)))
CIFAR10_METADATA = {
    'apply_one_hot_in_loss': False,
}

CIFAR100_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    flip_probability=0.5,
    alpha=1.0,
    crop_num_pixels=4,
    use_mixup=True,
    train_size=45000,
    valid_size=5000,
    test_size=10000,
    input_shape=(32, 32, 3),
    output_shape=(100,)))
CIFAR100_METADATA = {
    'apply_one_hot_in_loss': False,
}

SVHN_NO_EXTRA_DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    flip_probability=0.5,
    alpha=1.0,
    crop_num_pixels=4,
    use_mixup=False,
    train_size=73257 - 7000,
    valid_size=7000,
    test_size=26032,
    input_shape=(32, 32, 3),
    output_shape=(10,)))
SVHN_NO_EXTRA_METADATA = {
    'apply_one_hot_in_loss': False,
}


def _eval_batches(images,
                  labels,
                  per_host_batch_size,
                  num_batches=None,
                  valid_example_keys=None):
  """Produce a stream of batches for a single evaluation epoch."""
  for idx in itertools.islice(
      range(0, images.shape[0], per_host_batch_size), num_batches):
    inputs = jnp.array(images[idx:idx + per_host_batch_size])
    targets = jnp.array(labels[idx:idx + per_host_batch_size])
    data_dict = {
        'inputs': inputs,
        'targets': targets,
        'weights': jnp.ones(per_host_batch_size, dtype=inputs.dtype),
    }
    if valid_example_keys is not None:
      data_dict['example_key'] = valid_example_keys[idx:idx +
                                                    per_host_batch_size]
    yield data_utils.maybe_pad_batch(data_dict, per_host_batch_size)


def _shard_by_host_id(array):
  split_size = len(array) // jax.process_count()
  start = split_size * jax.process_index()
  return array[start: start+split_size]


def _prepare_small_image_datasets(
    data_train,
    data_valid,
    data_test,
    per_host_batch_size,
    per_host_eval_batch_size,
    train_size,
    rescale,
    input_shape,
    output_shape,
    shuffle_rng,
    augment_fn,
    is_one_hot=True,
    autoencoder=False,
    include_example_keys=False):
  """Prepare Dataset using tf.data.Datasets of the different splits."""
  if autoencoder and is_one_hot:
    raise ValueError(
        'One hot encoding cannot be applied to autoencoder datasets.')

  eval_image_iterator = functools.partial(
      data_utils.image_iterator,
      rescale=rescale,
      output_shape=output_shape,
      is_one_hot=is_one_hot,
      autoencoder=autoencoder)

  # Setup the eval_train split as a copy of the training data, in the form of
  # the first `num_train_batches` batches of the data as an np.array.
  eval_train_iterator = eval_image_iterator(data_train)

  num_train_batches = train_size // per_host_batch_size
  eval_train_data = list(
      itertools.islice(eval_train_iterator, 0, num_train_batches))

  eval_train_inputs = jnp.array([batch['inputs'] for batch in eval_train_data])
  eval_train_inputs_shape = (num_train_batches * per_host_batch_size,
                             *input_shape)
  eval_train_inputs = np.reshape(eval_train_inputs, eval_train_inputs_shape)
  eval_train_targets = jnp.array(
      [batch['targets'] for batch in eval_train_data])
  eval_train_output_shape = (num_train_batches * per_host_batch_size,
                             *output_shape)
  eval_train_targets = np.reshape(eval_train_targets, eval_train_output_shape)

  valid_inputs = jnp.array([])
  valid_targets = jnp.array([])
  valid_example_keys = jnp.array([])
  if data_valid:
    valid_data = next(eval_image_iterator(
        data_valid, include_example_keys=include_example_keys))
    valid_inputs = valid_data['inputs']
    valid_targets = valid_data['targets']
    if include_example_keys:
      valid_example_keys = valid_data['tfds_id']

  test_data = next(eval_image_iterator(data_test))
  test_inputs = jnp.array(test_data['inputs'].astype(np.float32))
  test_targets = test_data['targets']

  # Shard the validation and test data by host id.
  valid_inputs = _shard_by_host_id(valid_inputs)
  valid_targets = _shard_by_host_id(valid_targets)
  test_inputs = _shard_by_host_id(test_inputs)
  test_targets = _shard_by_host_id(test_targets)

  # TODO(gilmer): The simplest way to do this would be to directly yield from
  # tfds.as_numpy(). However we currently do not know how to properly handle
  # restarts with tfds. We'd like the train shuffle to depend on the epoch
  # number, so everytime we generate epoch 10 it yields the same pseudorandom
  # order.

  train_iterator_fn = functools.partial(
      data_utils.image_iterator,
      data_train,
      rescale=rescale,
      output_shape=output_shape,
      is_one_hot=is_one_hot,
      autoencoder=autoencoder,
      shuffle_rng=shuffle_rng,
      augment_fn=augment_fn)

  eval_train_epoch = functools.partial(
      _eval_batches,
      eval_train_inputs,
      eval_train_targets,
      per_host_eval_batch_size)
  valid_epoch = functools.partial(
      _eval_batches,
      valid_inputs,
      valid_targets,
      per_host_eval_batch_size,
      valid_example_keys=valid_example_keys)
  test_epoch = functools.partial(
      _eval_batches,
      test_inputs,
      test_targets,
      per_host_eval_batch_size)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)


def _process_small_tfds_image_ds(
    dataset_name,
    per_host_batch_size,
    per_host_eval_batch_size,
    train_size,
    valid_size,
    test_size,
    rescale,
    input_shape,
    output_shape,
    shuffle_rng,
    augment_fn=image_preprocessing.identity_augment,
    is_one_hot=True,
    autoencoder=False):
  """Helper wrapper around tfds which converts the data into the init2winit API.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen():
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory. By default, we will drop the last partial batch if the
  batch_size does not evenly divide the data set. The validation set is
  chosen to be the last valid_size images of the tfds train set.
  Note this has already been shuffled, so this will not necessarily correspond
  to the standard data validation set.

  This function makes the following assumptions: The tfds version only has a
  train and test split. The labels provided by tfds are integers (representing
  a classification problem). The entire dataset fits into memory. We assume that
  the data shape has rank 4, as in the case of image data. Additionally, we
  assume that the size of the data is an integer multiple of n_devices.
  TODO (gilmer): Remove this assumption.

  Args:
    dataset_name: (str) Currently tested mnist, fashion_mnist, cifar10.
    per_host_batch_size: The global train batch size, used to determine the
      batch size yielded from train_gen().
    per_host_eval_batch_size: The global eval batch size, used to determine the
      batch size yielded from valid_epoch() and test_epoch().
    train_size: (int) Number of training samples to use.
    valid_size: (int) Number of validation samples.
    test_size: (int) Number of test samples.
    rescale: Function to preprocess an input batch.
    input_shape: (tuple) Used to check that the data is of the correct shape.
    output_shape: (tuple) Shape of network output.
    shuffle_rng: jax.random.PRNGKey
    augment_fn: Function with API (rng, images, labels) -> images, labels.
      This function will be applied to every training batch.
    is_one_hot: (bool) If true, targets are one hot encoded.
    autoencoder: (bool) If true, targets are set to input images.

  Returns:
    train_iterator_fn, valid_epoch, test_epoch: three generators.
  """
  augment_fn = jax.jit(augment_fn)

  train_split = tfds.core.ReadInstruction('train', to=train_size, unit='abs')
  data_train = tfds.load(name=dataset_name, split=train_split,
                         as_dataset_kwargs={'shuffle_files': False})
  # Ensure a different shuffle of the training data on each host.
  shuffle_rng = jax.random.fold_in(shuffle_rng, jax.process_index())
  data_train = data_train.shuffle(
      train_size,
      reshuffle_each_iteration=True,
      seed=shuffle_rng[0])
  data_train = data_train.repeat()
  data_train = data_train.batch(per_host_batch_size)

  if valid_size > 0:
    valid_split = tfds.core.ReadInstruction(
        'train', from_=-valid_size, unit='abs')
    data_valid = tfds.load(name=dataset_name, split=valid_split,
                           as_dataset_kwargs={'shuffle_files': False})
    data_valid = data_valid.batch(valid_size)
  else:
    data_valid = None

  data_test = tfds.load(name=dataset_name, split='test',
                        as_dataset_kwargs={'shuffle_files': False})
  data_test = data_test.batch(test_size)

  return _prepare_small_image_datasets(data_train, data_valid, data_test,
                                       per_host_batch_size,
                                       per_host_eval_batch_size, train_size,
                                       rescale, input_shape, output_shape,
                                       shuffle_rng, augment_fn,
                                       is_one_hot, autoencoder)


def _load_deterministic_with_custom_validation(
    dataset_name,
    per_host_batch_size,
    per_host_eval_batch_size,
    train_size,
    valid_size,
    test_size,
    rescale,
    input_shape,
    output_shape,
    shuffle_rng,
    augment_fn=image_preprocessing.identity_augment,
    include_example_keys=False):
  """Load a small image dataset with a deterministic validation set.

  This allows users to deterministically load a validation set that is sampled
  from the underlying dataset's train set.

  Args:
    dataset_name: (str) Currently tested mnist, fashion_mnist, cifar10.
    per_host_batch_size: The global train batch size, used to determine the
      batch size yielded from train_gen().
    per_host_eval_batch_size: The global eval batch size, used to determine the
      batch size yielded from valid_epoch() and test_epoch().
    train_size: (int) Number of training samples to use.
    valid_size: (int) Number of validation samples.
    test_size: (int) Number of test samples.
    rescale: Function to preprocess an input batch.
    input_shape: (tuple) Used to check that the data is of the correct shape.
    output_shape: (tuple) Shape of network output.
    shuffle_rng: jax.random.PRNGKey
    augment_fn: Function with API (rng, images, labels) -> images, labels. This
      function will be applied to every training batch.
    include_example_keys: (bool) If True, then the tfds_id will be exposed in
      each batch dict of the validation set under the key `example_key`.

  Returns:
    train_iterator_fn, valid_epoch, test_epoch: three generators.
  """
  augment_fn = jax.jit(augment_fn)

  if include_example_keys:
    read_config = tfds.ReadConfig(add_tfds_id=True)
  else:
    read_config = None

  train_ds = tfds.load(
      dataset_name,
      split=tfds.core.ReadInstruction(
          'train', from_=0, to=train_size, unit='abs'))

  data_train = train_ds.shuffle(
      train_size,
      reshuffle_each_iteration=True,
      seed=int(jax.random.randint(shuffle_rng, [1], 0, 1000)))
  data_train = data_train.repeat()
  data_train = data_train.batch(per_host_batch_size)

  if valid_size > 0:
    valid_ds = tfds.load(
        dataset_name,
        read_config=read_config,
        split=tfds.core.ReadInstruction(
            'train', from_=train_size, to=train_size + valid_size, unit='abs'))
    data_valid = valid_ds.batch(valid_size)
  else:
    valid_ds = tf.data.Dataset.from_tensor_slices([])
    data_valid = []

  data_test = tfds.load(
      name=dataset_name,
      split='test',
      as_dataset_kwargs={'shuffle_files': False})
  data_test = data_test.batch(test_size)

  return _prepare_small_image_datasets(
      data_train,
      data_valid,
      data_test,
      per_host_batch_size,
      per_host_eval_batch_size,
      train_size,
      rescale,
      input_shape,
      output_shape,
      shuffle_rng,
      augment_fn,
      include_example_keys=include_example_keys)


def get_mnist(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Returns generators for the MNIST train, validation, and test set.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory (70000 images). The last batch may be smaller than
  batch_size if batch_size does not evenly divide the size of the data. The
  validation set is chosen to be the last 10000 images of the tfds train set.
  Note this has already been shuffled, so this will not necessarily correspond
  to the standard MNIST validation set.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits.
  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  rescale = lambda x: x / 255.0
  return _process_small_tfds_image_ds('mnist',
                                      per_host_batch_size,
                                      per_host_eval_batch_size,
                                      hps.train_size, hps.valid_size,
                                      hps.test_size, rescale,
                                      hps.input_shape,
                                      hps.output_shape, shuffle_rng)


def get_mnist_autoencoder(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """generators for the MNIST autoencoder train, validation, and test set.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory (70000 images). The last batch may be smaller than
  batch_size if batch_size does not evenly divide the size of the data. The
  validation set is chosen to be the last 10000 images of the tfds train set.
  Note this has already been shuffled, so this will not necessarily correspond
  to the standard MNIST validation set.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits.

  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  rescale = lambda x: x / 255.0
  return _process_small_tfds_image_ds('mnist',
                                      per_host_batch_size,
                                      per_host_eval_batch_size, hps.train_size,
                                      hps.valid_size, hps.test_size, rescale,
                                      hps.input_shape, hps.output_shape,
                                      shuffle_rng,
                                      is_one_hot=False, autoencoder=True)


def get_fashion_mnist(shuffle_rng,
                      batch_size,
                      eval_batch_size,
                      hps=None):
  """Returns generators for the Fashion MNIST train, validation, and test set.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory (70000 images). The last batch may be smaller than
  batch_size if batch_size does not evenly divide the size of the data. The
  validation set is chosen to be the last 10000 images of the tfds train set.
  Note this has already been shuffled, so the validation set depends on the
  shuffling.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits.
  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  rescale = lambda x: x / 255.0
  return _process_small_tfds_image_ds('fashion_mnist',
                                      per_host_batch_size,
                                      per_host_eval_batch_size,
                                      hps.train_size, hps.valid_size,
                                      hps.test_size, rescale,
                                      hps.input_shape,
                                      hps.output_shape, shuffle_rng)


def get_cifar10(shuffle_rng,
                batch_size,
                eval_batch_size,
                hps=None):
  """Returns generators for the CIFAR10 train, validation, and test set.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch.

  We load the entire dataset into memory (60000 images). The hps dictionary
  allows users to specify the train_size, valid_size and test_size. The original
  dataset doesn't prescribe a validation set. So the validation set will be
  drawn from the train set. This will be done deterministically using a hash
  function.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits. See image_preprocessing for
      hparams that control the data augmentation.

  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  logging.info('per_host_batch_size: %s', per_host_batch_size)
  logging.info('per_host_eval_batch_size: %s', per_host_eval_batch_size)

  mean = jnp.array([125.3, 123.0, 113.9])[None, None, None, :]
  std = jnp.array([63.0, 62.1, 66.7])[None, None, None, :]
  rescale = lambda x: (x - mean) / std
  augment_fn = functools.partial(image_preprocessing.augment_cifar10, hps=hps)

  if hps.train_size + hps.valid_size > 50000:
    raise ValueError('The sum of train_size and valid_size should not exceed '
                     '50k.')
  if hps.test_size > 10000:
    raise ValueError('test_size should not exceed 10k')

  return _load_deterministic_with_custom_validation(
      'cifar10',
      per_host_batch_size,
      per_host_eval_batch_size,
      hps.train_size,
      hps.valid_size,
      hps.test_size,
      rescale,
      hps.input_shape,
      hps.output_shape,
      shuffle_rng,
      augment_fn,
      include_example_keys=hps.get('include_example_keys', False))


def get_cifar100(shuffle_rng,
                 batch_size,
                 eval_batch_size,
                 hps=None):
  """Returns generators for the CIFAR100 train, validation, and test set.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory (60000 images). The last batch may be smaller than
  batch_size if batch_size does not evenly divide the size of the data. The
  validation set is chosen to be the last 10000 images of the tfds train set.
  Note this has already been shuffled, so this will not necessarily correspond
  to any particular validation set.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits. See image_preprocessing for
      hparams that control the data augmentation.
  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()

  mean = jnp.array([129.3041658, 124.06996185, 112.4340492])[None, None,
                                                             None, :]
  std = jnp.array([68.17024395, 65.3918073, 70.41836985])[None, None, None, :]
  rescale = lambda x: (x - mean) / std
  augment_fn = functools.partial(image_preprocessing.augment_cifar10, hps=hps)
  return _process_small_tfds_image_ds('cifar100',
                                      per_host_batch_size,
                                      per_host_eval_batch_size,
                                      hps.train_size, hps.valid_size,
                                      hps.test_size, rescale,
                                      hps.input_shape,
                                      hps.output_shape, shuffle_rng, augment_fn)


# TODO(znado): add version that also has the "extra" training examples.
def get_svhn_no_extra(
    shuffle_rng,
    batch_size,
    eval_batch_size,
    hps=None):
  """Returns generators for the SVHN train, validation, and test set.

  Note that we do not include the "extra" 531131 examples which are sometimes
  used as additional training data, but according to the SVHN website are easier
  examples http://ufldl.stanford.edu/housenumbers/.

  Returns three data generators, train, validation and test. The API of
  a data generator is:
    for batch in train_gen(epoch):
       ...
  suffle_rng is used with the epoch number to shuffle the train epoch every
  epoch. We do not shuffle the validation or test sets. We load the entire
  dataset into memory (99289 images). The last batch may be smaller than
  batch_size if batch_size does not evenly divide the size of the data. The
  validation set is chosen to be the last 10000 images of the tfds train set.
  Note this has already been shuffled, so this will not necessarily correspond
  to a standard validation set.

  The same data augmentation as CIFAR10 is used, except for normalization by the
  dataset mean and stddev.

  Args:
    shuffle_rng: jax.random.PRNGKey
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: The global eval batch size, used to determine the batch
      size yielded from valid_epoch() and test_epoch().
    hps: Hparams object. hps.train_size, hps.valid_size, and hps.test_size will
      specify the sizes of the various data splits. See image_preprocessing for
      hparams that control the data augmentation.
  Returns:
    train_epoch, valid_epoch, test_epoch: three generators.
  """
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  rescale = lambda x: x / 255.0
  augment_fn = functools.partial(image_preprocessing.augment_cifar10, hps=hps)
  return _process_small_tfds_image_ds(
      'svhn_cropped',
      per_host_batch_size,
      per_host_eval_batch_size,
      hps.train_size,
      hps.valid_size,
      hps.test_size,
      rescale,
      hps.input_shape,
      hps.output_shape,
      shuffle_rng,
      augment_fn=augment_fn)
