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

"""Input pipeline for a WMT dataset."""

import os
from typing import Dict, List, Optional, Union

from clu import deterministic_data
from init2winit.dataset_lib import mt_tokenizer as tokenizer
import jax
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.AUTOTUNE
Features = Dict[str, tf.Tensor]


def get_user_defined_symbols(ds_info: tfds.core.DatasetInfo,
                             reverse_translation: bool):
  """Get language id token."""
  if reverse_translation:
    token, _ = ds_info.supervised_keys
  else:
    _, token = ds_info.supervised_keys
  return f'<2{token}>'


class NormalizeFeatureNamesOp:
  """Normalizes feature names to 'inputs' and 'targets'."""

  def __init__(self, ds_info: tfds.core.DatasetInfo, reverse_translation: bool):
    self.input_lang, self.target_lang = ds_info.supervised_keys
    if reverse_translation:
      self.input_lang, self.target_lang = self.target_lang, self.input_lang

  def __call__(self, features: Features) -> Features:
    features['inputs'] = features.pop(self.input_lang)
    features['targets'] = features.pop(self.target_lang)
    return features


class TaskTokenOp:
  """Adds '2xx' task token to 'inputs'."""

  def __init__(self, ds_info: tfds.core.DatasetInfo,
               reverse_translation: bool):
    self.token_lang = get_user_defined_symbols(ds_info, reverse_translation)

  def __call__(self, features):
    features['inputs'] = self.token_lang + ' ' + features['inputs']
    return features


# TODO(dxin): Rewrite this as a ds.map w/ functools.
class ImportanceSamplingOp:
  """Adds '2xx' task token to 'inputs'."""

  def __init__(self, weight):
    self.weight = weight

  def __call__(self, features):
    features['weights'] = np.array([self.weight], dtype=np.float32)
    return features


def maybe_pad_batch(batch,
                    desired_batch_size,
                    mask_key='targets'):
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
    mask_key: Typically used for text datasets, it's either 'inputs' (for
      encoder only models like language models) or 'targets'
      (for encoder-decoder models like seq2seq tasks) to decide weights for
      padded sequence. For Image datasets, this will be (most likely) unused.

  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally we
    add a key representing weights, to indicate how the batch was padded.
  """
  batch_size = batch['inputs'].shape[0]
  batch_pad = desired_batch_size - batch_size

  if mask_key not in ['targets', 'inputs']:
    raise ValueError(f'Incorrect mask key {mask_key}.')

  if 'weights' in batch:
    batch['weights'] = np.multiply(batch['weights'],
                                   np.where(batch[mask_key] > 0, 1, 0))
  else:
    batch['weights'] = np.where(batch[mask_key] > 0, 1, 0)

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if batch_pad == 0:
    new_batch = jax.tree.map(lambda x: x, batch)
    return new_batch

  def zero_pad(ar, pad_axis):
    pw = [(0, 0)] * ar.ndim
    pw[pad_axis] = (0, batch_pad)
    return np.pad(ar, pw, mode='constant')

  padded_batch = {'inputs': zero_pad(batch['inputs'], 0)}
  batch_keys = list(batch.keys())
  batch_keys.remove('inputs')
  for key in batch_keys:
    padded_batch[key] = zero_pad(batch[key], 0)
  return padded_batch


def get_raw_dataset(dataset_builder: tfds.core.DatasetBuilder,
                    split: str,
                    *,
                    reverse_translation: bool = False,
                    add_language_token: bool = False) -> tf.data.Dataset:
  """Loads a raw WMT dataset and normalizes feature keys.

  Args:
    dataset_builder: TFDS dataset builder that can build `slit`.
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.
    reverse_translation: bool: whether to reverse the translation direction.
      e.g. for 'de-en' this translates from english to german.
    add_language_token: whether to prepend a 2xx language token to the input.

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """
  num_examples = dataset_builder.info.splits[split].num_examples
  per_host_split = deterministic_data.get_read_instruction_for_host(
      split, num_examples, drop_remainder=False)
  ds = dataset_builder.as_dataset(split=per_host_split, shuffle_files=False)
  ds = ds.map(
      NormalizeFeatureNamesOp(
          dataset_builder.info, reverse_translation=reverse_translation),
      num_parallel_calls=AUTOTUNE)
  if add_language_token:
    ds = ds.map(
        TaskTokenOp(dataset_builder.info,
                    reverse_translation=reverse_translation),
        num_parallel_calls=AUTOTUNE)
  return ds


def pack_dataset(dataset: tf.data.Dataset,
                 key2length: Union[int, Dict[str, int]],
                 keys: Optional[List[str]] = None) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Adapted from the mesh-tf implementation.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    key2length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError('Key %s not found in dataset.  Available keys are %s' %
                       (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError('Tensors to be packed must be one-dimensional.')
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  if isinstance(key2length, int):
    key2length = {k: key2length for k in keys}
  for k in keys:
    for suffix in ['_segmentation', '_position']:
      key2length[k + suffix] = key2length[k]

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:key2length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(key2length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, key2length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset: tf.data.Dataset, keys: List[str],
                      key2length: Dict[str, int]) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    key2length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + '_position'] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.
    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
      outputs[k + '_position'] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray

      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:key2length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + '_position'] = tf.concat(
            [partial[k + '_position'],
             tf.range(new_seq_len)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    # For loop over all examples in the batch.
    i, partial, outputs = tf.while_loop(
        cond=lambda *_: True,
        body=body_fn,
        loop_vars=(i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ),
        maximum_iterations=dynamic_batch_size)
    _, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + '_segmentation'] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + '_position'], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed

  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()


# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def get_sampled_dataset(ds_builders,
                        split: str,
                        rates: List[int],
                        reverse_translation: bool,
                        add_language_token: bool,
                        loss_weights: List[float],
                        is_training: bool,
                        sample_seed: int,
                        shuffle_seed: int,
                        shuffle_buffer_size: int = 1024):
  """Create a sampled training dataset."""
  raw_data = []
  for builder in ds_builders:
    raw_data.append(get_raw_dataset(
        builder, split,
        reverse_translation=reverse_translation,
        add_language_token=add_language_token))

  if loss_weights is not None:
    raw_data = [ds.map(ImportanceSamplingOp(weight),
                       num_parallel_calls=AUTOTUNE) for ds, weight
                in zip(raw_data, loss_weights)]

  def _shuffle_repeat(dataset, shuffle_seed: int,
                      shuffle_buffer_size):
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.repeat()
    return dataset

  if is_training:
    raw_data = [_shuffle_repeat(data, shuffle_seed=shuffle_seed,
                                shuffle_buffer_size=shuffle_buffer_size)
                for data in raw_data]

  sampled_raw_data = tf.data.experimental.sample_from_datasets(
      raw_data, rates, seed=sample_seed)
  return sampled_raw_data


def preprocess_wmt_data(dataset,
                        pack_examples: bool = True,
                        max_length: int = 100,
                        batch_size: int = 256,
                        prefetch_size: int = AUTOTUNE):
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):
    # This is after applying sentence piece tokenization.
    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)

    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if pack_examples:
    dataset = pack_dataset(dataset, max_length)
    dataset = dataset.batch(batch_size, drop_remainder=False)
  else:  # simple (static-shape) padded batching
    if 'weights' in dataset.element_spec:
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes={
              'inputs': max_length,
              'targets': max_length,
              'weights': (1,)
          },
          padding_values={
              'inputs': 0,
              'targets': 0,
              'weights': 0.0
          },
          drop_remainder=False)
    else:
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes={
              'inputs': max_length,
              'targets': max_length
          },
          padding_values={
              'inputs': 0,
              'targets': 0
          },
          drop_remainder=False)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_wmt_datasets(config: config_dict.ConfigDict,
                     *,
                     shuffle_seed: int,
                     sample_seed: int,
                     n_devices: int,
                     per_host_batch_size: int,
                     per_host_eval_batch_size: int,
                     vocab_path: Optional[str] = None):
  """Load and return dataset of batched examples for use during training."""
  if per_host_batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (per_host_batch_size, n_devices))
  if per_host_eval_batch_size % n_devices:
    raise ValueError("Eval Batch size %d isn't divided evenly by n_devices %d" %
                     (per_host_eval_batch_size, n_devices))
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  if config.tfds_dataset_key is not None:
    dataset_keys = [config.tfds_dataset_key]
    dataset_rates = None
  else:
    dataset_keys = config.tfds_dataset_keys
    dataset_rates = config.rates
  train_ds_builders = [tfds.builder(tfds_dataset_key) for tfds_dataset_key in
                       dataset_keys]

  eval_ds_builder = tfds.builder(config.tfds_eval_dataset_key)
  if config.tfds_predict_dataset_key:
    predict_ds_builder = tfds.builder(config.tfds_predict_dataset_key)
  else:
    predict_ds_builder = tfds.builder(config.tfds_eval_dataset_key)

  # TODO(dxin): give each task its own reverse translation bool.
  sampled_train_data = get_sampled_dataset(
      train_ds_builders, config.train_split, dataset_rates,
      config.reverse_translation, config.add_language_token,
      loss_weights=config.loss_weights,
      is_training=True, sample_seed=sample_seed,
      shuffle_seed=shuffle_seed)

  eval_data = get_raw_dataset(
      eval_ds_builder,
      config.eval_split,
      reverse_translation=config.reverse_translation,
      add_language_token=config.add_language_token)

  predict_data = get_raw_dataset(
      predict_ds_builder,
      config.predict_split,
      reverse_translation=config.reverse_translation,
      add_language_token=config.add_language_token)

  # Tokenize data.
  user_defined_symbols = []
  if config.add_language_token:
    user_defined_symbols = [
        get_user_defined_symbols(ds_builders.info, True)
        for ds_builders in train_ds_builders]
    user_defined_symbols.extend(
        [get_user_defined_symbols(ds_builders.info, False)
         for ds_builders in train_ds_builders])
  user_defined_symbols = list(set(user_defined_symbols))

  # TODO(dxin): Check in vocab file eventually.
  sp_tokenizer = tokenizer.load_or_train_tokenizer(
      sampled_train_data,
      vocab_path=vocab_path,
      vocab_size=config.vocab_size,
      max_corpus_chars=config.max_corpus_chars,
      character_coverage=config.character_coverage,
      byte_fallback=config.byte_fallback,
      split_digits=config.split_digits,
      user_defined_symbols=user_defined_symbols)
  sampled_train_data = sampled_train_data.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  eval_data = eval_data.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  predict_data = predict_data.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  sampled_train_ds = preprocess_wmt_data(
      sampled_train_data,
      pack_examples=config.pack_examples,
      batch_size=per_host_batch_size,
      max_length=config.max_target_length)

  eval_ds = preprocess_wmt_data(
      eval_data,
      pack_examples=False,
      batch_size=per_host_eval_batch_size,
      max_length=config.max_eval_target_length)

  predict_ds = preprocess_wmt_data(
      predict_data,
      pack_examples=False,
      batch_size=per_host_eval_batch_size,
      max_length=config.max_predict_length)

  return sampled_train_ds, eval_ds, predict_ds
