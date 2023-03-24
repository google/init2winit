# coding=utf-8
# Copyright 2023 The init2winit Authors.
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

"""Proteins input pipeline."""

import enum
import functools
import itertools

from absl import logging
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import protein_vocab
from init2winit.dataset_lib.data_utils import Dataset
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

uniref = None

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        max_target_length=1024,
        max_eval_target_length=1024,
        input_shape=(1024,),
        output_shape=(31,),
        train_size=27000000,
        data_name='uniref50/unaligned_encoded',
    ))

METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_inputs': False,
    'causal': False,
    'bert': True,
    'pad_token': 29,
}


class Mode(enum.Enum):
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'predict'
  SAMPLE = 'sample'


def _crop(x, max_length, sample):
  """Select (optionally random) crop from sequence."""
  if sample:
    # Optionally sample random starting position.
    start = tf.random.uniform(
        (),
        dtype=tf.int32,
        maxval=tf.maximum(1,
                          tf.shape(x)[0] - max_length + 1))
  else:
    start = 0

  x = x[start:(start + max_length)]
  return x


def preprocess_masked(inputs, random_tokens, mask_token, pad_token, mask_rate,
                      mask_token_proportion, random_token_proportion, mode,
                      rng):
  """Preprocess inputs for masked language modeling.

  Args:
    inputs: [batch x length] input tokens.
    random_tokens: Set of tokens usable for replacing
    mask_token: Int ID to mask blanks with.
    pad_token: Int ID for PAD token. Positions left unchanged.
    mask_rate: Proportion of tokens to mask out.
    mask_token_proportion: Replace this proportion of chosen positions with
      MASK.
    random_token_proportion: Replace this proportion of chosen positions with
      randomly sampled tokens
    mode: Mode key.
    rng: Jax RNG.

  Returns:
    Tuple of [batch x length] inputs, targets, per position weights. targets
      will have random positions masked out with either a MASK token, or a
      randomly chosen token from the vocabulary.
  """
  total = random_token_proportion + mask_token_proportion
  if total < 0 or total > 1:
    raise ValueError('Sum of random proportion and mask proportion must be'
                     ' in [0, 1] range.')
  targets = inputs

  if mode == Mode.PREDICT:
    weights = jnp.full_like(targets, 1)
    masked_inputs = inputs  # Pass through
  else:
    if rng is None:
      if mode is not Mode.EVAL:
        raise ValueError('Must provide RNG unless in eval mode.')
      # How to keep same eval set across runs?
      # Make each sequences mask invariant to other members
      # of the batch. Right now there is batch size dependence.
      rng = jax.random.PRNGKey(jnp.sum(inputs))

    # Get positions to leave untouched
    is_pad = inputs == pad_token

    # Positions to mask
    rng, subrng = jax.random.split(rng)
    should_mask = jax.random.bernoulli(subrng, p=mask_rate, shape=inputs.shape)
    should_mask = jnp.where(is_pad, 0, should_mask)  # Don't mask out padding.

    # Generate full array of random tokens.
    rng, subrng = jax.random.split(rng)
    random_ids = jax.random.randint(
        subrng, inputs.shape, minval=0, maxval=len(random_tokens))

    fullrandom = random_tokens[random_ids]
    # Full array of MASK tokens
    fullmask = jnp.full_like(inputs, mask_token)

    # Build up masked array by selecting from inputs/fullmask/fullrandom.
    rand = jax.random.uniform(rng, shape=inputs.shape)

    # Remaining probability mass stays original values after MASK and RANDOM.
    # MASK tokens.
    masked_inputs = jnp.where(rand < mask_token_proportion, fullmask,
                              inputs)
    # Random tokens.
    masked_inputs = jnp.where(
        jnp.logical_and(rand >= mask_token_proportion,
                        rand < mask_token_proportion + random_token_proportion),
        fullrandom, masked_inputs)

    # Only replace positions where `should_mask`
    masked_inputs = jnp.where(should_mask, masked_inputs, inputs)
    weights = should_mask

  return masked_inputs, targets, weights


class BertMasker():
  """Construct BERT masker given a vocab."""

  def __init__(self,
               vocab,
               mask_rate=0.15,
               mask_token_proportion=0.1,
               random_token_proportion=0.8):
    self._vocab = vocab
    if vocab.mask is None:
      raise ValueError('Vocabulary must specify a MASK token.')
    special_tokens = [vocab.bos, vocab.eos, vocab.mask, vocab.pad]
    special_tokens = [x for x in special_tokens if x is not None]
    normal_tokens = [x for x in vocab.token_ids if x not in special_tokens]
    self._special_tokens = jnp.array(special_tokens)
    self._normal_tokens = jnp.array(normal_tokens)
    self._mask_rate = mask_rate
    self._mask_token_proportion = mask_token_proportion
    self._random_token_proportion = random_token_proportion

  def __call__(self, inputs, mode, rng):
    inputs, targets, weights = preprocess_masked(
        inputs=inputs,
        mode=mode,
        rng=rng,
        random_tokens=self._normal_tokens,
        mask_token=self._vocab.mask,
        pad_token=self._vocab.pad,
        mask_rate=self._mask_rate,
        mask_token_proportion=self._mask_token_proportion,
        random_token_proportion=self._random_token_proportion)
    return inputs, targets, weights


def shift_right(x, bos_token):
  """Shift the input to the right by padding on axis 1 at train time."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[1] = (1, 0)  # Padding on axis=1
  padded = tf.pad(
      x,
      pad_widths,
      mode='constant',
      constant_values=tf.constant(bos_token, dtype=x.dtype))
  return padded[:, :-1]


def load_protein_tfds(name,
                      split,
                      field='sequence',
                      num_epochs=1,
                      shuffle_buffer=2**15,
                      batch_size=32,
                      max_length=None,
                      sample_length=True,
                      data_dir=None,
                      drop_remainder=True):
  """Load protein tfds by name.

  If split is `train`, shuffle data.

  Args:
    name: Name of dataset, such as `pfam/unaligned_encoded`,
    split: Split name. One of train/valid/test.
    field: Field to return from each example.
    num_epochs: Number of times to repeat dataset. None for infinite.
    shuffle_buffer: Shuffle buffer size. Only applied to train split.
    batch_size: Number of examples per batch.
    max_length: Max length of sequences to return.
    sample_length: If True, sample a random crop if sequence is too long.
    data_dir: Data directory containing tfds files.
    drop_remainder: Whether to drop the last batch when having less than
      batch_size elements.

  Returns:
    Tuple of (batched dataset, protein_vocab.Vocabulary)
  """

  shuffle = split == 'train'
  ds = tfds.load(
      name,
      split=split,
      with_info=False,
      data_dir=data_dir,
      shuffle_files=shuffle)

  # Construct vocab from stored metadata.
  # TODO(ddohan): Regenerate dataset with new vocab.
  vocab = protein_vocab.make_protein_vocab(include_align_tokens=True)

  def _get_field(example):
    return example[field]

  ds = ds.map(_get_field, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # pylint: disable=protected-access
  if max_length:
    ds = ds.map(
        functools.partial(
            _crop, max_length=max_length, sample=sample_length),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # pylint: disable=protected-access

  if shuffle:
    ds = ds.shuffle(buffer_size=shuffle_buffer)

  ds = ds.repeat(num_epochs)

  # TODO(ddohan): Consider adding bucketing by sequence length.
  ds = ds.padded_batch(
      batch_size,
      padded_shapes=max_length,
      padding_values=np.array(vocab.pad, dtype=np.int64),
      drop_remainder=drop_remainder)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, vocab


def load_dataset(data_name,
                 batch_size,
                 eval_batch_size,
                 length=512):
  """Load protein dataset.

  Args:
    data_name: Name of the dataset to load.
    batch_size: Per host batch size.
    eval_batch_size: Per host eval batch size.
    length: Length of sequences.

  Returns:
    Tuple of training dataset, valid dataset, and vocab info.
  """
  logging.info('Loading data_name: %s', data_name)
  train_ds, vocab = load_protein_tfds(
      data_name, 'train', max_length=length, batch_size=batch_size)
  valid_ds, vocab = load_protein_tfds(
      data_name, 'validation', max_length=length, batch_size=eval_batch_size)

  return train_ds, valid_ds, vocab


def _batch_to_dict(batch, masker, mode, rng):
  batch = data_utils.tf_to_numpy(batch)
  inputs, targets, weights = masker(batch, mode, rng)
  return {
      'inputs': inputs,
      'targets': targets,
      'weights': weights
  }


def get_uniref(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Wrapper to conform to the general dataset API."""
  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  return _get_uniref(
      per_host_batch_size,
      per_host_eval_batch_size,
      hps,
      shuffle_rng)


def _get_uniref(
    per_host_batch_size,
    per_host_eval_batch_size,
    hps,
    data_rng):
  """Data generators for Uniref50 clustered protein dataset."""
  # TODO(gilmer) Currently uniref drops the last partial batch on eval.
  logging.warning(
      'Currently the Protein dataset drops the last partial batch on eval')
  if jax.process_count() > 1:
    raise NotImplementedError('Proteins does not support multihost training')

  n_devices = jax.local_device_count()
  if per_host_batch_size % n_devices != 0:
    raise ValueError('n_devices={} must divide per_host_batch_size={}.'.format(
        n_devices, per_host_batch_size))
  if per_host_eval_batch_size % n_devices != 0:
    raise ValueError(
        'n_devices={} must divide per_host_eval_batch_size={}.'.format(
            n_devices, per_host_eval_batch_size))

  train_ds, eval_ds, vocab = load_dataset(
      hps.data_name,
      batch_size=per_host_batch_size,
      eval_batch_size=per_host_eval_batch_size,
      length=hps.max_target_length)

  masker = BertMasker(vocab=vocab)

  def train_iterator_fn():
    for batch_index, batch in enumerate(iter(train_ds)):
      batch_rng = jax.random.fold_in(data_rng, batch_index)
      yield _batch_to_dict(batch, masker, 'train', batch_rng)

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(train_ds)
    for batch_index, batch in enumerate(
        itertools.islice(eval_train_iter, num_batches)):
      batch_rng = jax.random.fold_in(data_rng, batch_index)
      yield _batch_to_dict(batch, masker, 'eval', batch_rng)

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch_index, batch in enumerate(
        itertools.islice(valid_iter, num_batches)):
      batch_rng = jax.random.fold_in(data_rng, batch_index)
      yield _batch_to_dict(batch, masker, 'eval', batch_rng)

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
