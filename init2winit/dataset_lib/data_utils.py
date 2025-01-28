# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Common code used by different models."""

import collections

import flax.linen as nn
import jax
from jax.nn import one_hot
from jax.sharding import PartitionSpec as P
import numpy as np


Dataset = collections.namedtuple('Dataset', [
    'train_iterator_fn',
    'eval_train_epoch',
    'valid_epoch',
    'test_epoch',
])


def iterator_as_numpy(iterator):
  for x in iterator:
    yield jax.tree.map(lambda y: y._numpy(), x)  # pylint: disable=protected-access


def image_iterator(data,
                   rescale,
                   output_shape,
                   is_one_hot,
                   autoencoder,
                   shuffle_rng=None,
                   augment_fn=None,
                   include_example_keys=False):
  """Preprocesses the batch data arrays in the data generator.

  Rescales inputs. One hot encode targets if is_one_hot is true.
  Set targets to inputs of output_shape if autoencoder is true.

  Args:
    data: An iterator generating dicts of input and target data arrays.
    rescale: A lambda function preprocessing input data arrays.
    output_shape: Shape of network output.
    is_one_hot: If true, targets are one hot encoded.
    autoencoder: If true, targets are set to inputs.
    shuffle_rng: jax.random.PRNGKey
    augment_fn: The number of classes used for the dataset.
    include_example_keys: If True, then the tfds_id will be exposed in each
      batch dict of the validation set under the key `example_key`.

  Yields:
    A dictionary mapping keys ('image', 'label') to preprocessed data arrays.
  """
  for batch_index, batch in enumerate(iterator_as_numpy(iter(data))):
    inputs = batch['image']
    targets = batch['label']
    if is_one_hot:
      targets = one_hot(batch['label'], output_shape[-1])
    if augment_fn:
      batch_rng = jax.random.fold_in(shuffle_rng, batch_index)
      inputs, targets = augment_fn(batch_rng, inputs, targets)
    inputs = rescale(inputs)
    if autoencoder:
      batch_output_shape = tuple([inputs.shape[0]] + list(output_shape))
      targets = inputs.reshape(batch_output_shape)
    if include_example_keys:
      yield {'inputs': inputs, 'targets': targets, 'tfds_id': batch['tfds_id']}
    else:
      yield {'inputs': inputs, 'targets': targets}


def maybe_pad_batch(batch,
                    desired_batch_size,
                    data_format=None,
                    mask_key=None,
                    padding_value=0.0):
  """Zero pad the batch on the right to desired_batch_size.

  All keys in the batch dictionary will have their corresponding arrays padded.
  Will return a dictionary with the same keys, additionally with the key
  'weights' added, with 1.0 indicating indices which are true data and 0.0
  indicating a padded index.

  Args:
    batch: A dictionary mapping keys to arrays. We assume that inputs is one of
      the keys.
    desired_batch_size: All arrays in the dict will be padded to have first
      dimension equal to desired_batch_size.
    data_format: String data format of batch['inputs'], used to determine which
      dimension to pad. If not provided then it is assumed the first dimension
      is the batch dimension.
    mask_key: Typically used for text datasets, it's either 'inputs' (for
      encoder only models like language models) or 'targets'
      (for encoder-decoder models like seq2seq tasks) to decide weights for
      padded sequence. For Image datasets, this will be (most likely) unused.
    padding_value: value to be used as padding.

  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally we
    add a key representing weights, to indicate how the batch was padded.
  """
  if data_format is None or data_format == 'NHWC':
    batch_axis = 0
  elif data_format == 'HWCN':
    batch_axis = 3
  elif data_format == 'HWNC':
    batch_axis = 2
  else:
    raise ValueError('Unsupported data format {}.'.format(data_format))

  batch_size = batch['inputs'].shape[batch_axis]
  batch_pad = desired_batch_size - batch_size

  if mask_key:  # Typically for text models (LM, MT).
    batch['weights'] = np.where(batch[mask_key] > 0, 1, 0)
  else:
    batch['weights'] = np.ones(batch_size, dtype=np.float32)

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if batch_pad == 0:
    new_batch = jax.tree.map(lambda x: x, batch)
    return new_batch

  def zero_pad(ar, pad_axis):
    pw = [(0, 0)] * ar.ndim
    pw[pad_axis] = (0, batch_pad)
    return np.pad(ar, pw, mode='constant', constant_values=padding_value)

  padded_batch = {'inputs': zero_pad(batch['inputs'], batch_axis)}
  batch_keys = list(batch.keys())
  batch_keys.remove('inputs')
  for key in batch_keys:
    padded_batch[key] = zero_pad(batch[key], 0)
  return padded_batch


def tf_to_numpy(tfds_data):
  # Safe because we won't mutate. Avoids an extra copy from tfds.
  convert_data = lambda x: x._numpy()  # pylint: disable=protected-access
  return jax.tree.map(convert_data, tfds_data)


# See b/280322542 for context.
def convert_jax_to_tf_random_seed(jax_prng_key: jax.random.PRNGKey) -> int:
  tf_seed = jax.random.bits(jax_prng_key)
  return tf_seed


def make_global_array(local_data, mesh):
  """Util to combine per-host batches into a global batch array.

  Args:
    local_data: local data batch on host. 
    mesh: mesh specification to shard the data.

  Returns:
    global_array: global data batch.
  """
  global_shape = (
      local_data.shape[0] * jax.process_count(),
      *local_data.shape[1:],
  )
  sharding = jax.NamedSharding(mesh, P('devices'))

  global_array = jax.make_array_from_process_local_data(
      sharding, local_data, global_shape
  )
  return global_array


def shard_pytree(pytree, mesh, shardings=None):
  if shardings is None:
    shardings = nn.get_sharding(pytree, mesh)
  pytree = jax.tree_util.tree_map(
      lambda arr, sharding: jax.make_array_from_process_local_data(
          sharding, arr
      ),
      pytree,
      shardings,
  )

  return shardings, pytree
