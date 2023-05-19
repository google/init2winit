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

"""Librispeech dataset input processing pipeline."""
import functools
from typing import Dict

from init2winit.dataset_lib import spm_tokenizer
from init2winit.dataset_lib import wpm_tokenizer
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]

ALLOWED_TOKENIZERS = ['WPM', 'SPM', 'RAW']


class CharacterTokenizer():
  """Tokenizer that uses raw character level vocab."""

  def __init__(self):
    self.vocab = {
        '<unk>': 0,
        '<s>': 1,
        '</s>': 2,
        '_': 3,
        ' ': 4,
        '\'': 5,
        'A': 6,
        'B': 7,
        'C': 8,
        'D': 9,
        'E': 10,
        'F': 11,
        'G': 12,
        'H': 13,
        'I': 14,
        'J': 15,
        'K': 16,
        'L': 17,
        'M': 18,
        'N': 19,
        'O': 20,
        'P': 21,
        'Q': 22,
        'R': 23,
        'S': 24,
        'T': 25,
        'U': 26,
        'V': 27,
        'W': 28,
        'X': 29,
        'Y': 30,
        'Z': 31,
    }

  def string_to_ids(self, sentence):
    """Converts input string to list of character token ids.

    Args:
      sentence: sentence containing transcription of audio in dataset.

    Returns:
      converted list of character token ids.
    """
    result = [1]  # append start token
    for i in range(len(sentence)):
      current_char = sentence[i]
      current_label = self.vocab[current_char]

      result.append(current_label)
    result.append(2)  # append end token

    return result

  def tokenize(self, sentence):
    sentence_content = sentence.numpy().decode('utf-8')
    targets = self.string_to_ids(sentence_content)

    return [targets]


def length_filter(max_input_length, max_target_length):
  """Creates dataset filter functions based on input and target lengths.

  Args:
    max_input_length: max allowed input length for frequency array.
    max_target_length: max allowed sentence length for transcription output.

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """

  def filter_inputs_by_length_fn(x):
    source = x['inputs']
    return tf.less(tf.shape(source)[0], max_input_length + 1)

  def filter_targets_by_length_fn(x):
    source = x['targets']
    return tf.less(tf.shape(source)[0], max_target_length + 1)

  return filter_inputs_by_length_fn, filter_targets_by_length_fn


def _preprocess_output(features, tokenizer, hps):
  """Tokenizes string transcriptions to output int32 token ids.

  Args:
    features: input tf data features
    tokenizer: tokenizer to be used to tokenize the input
      feature's output transcription.
    hps: hyperparameters for the dataset pipeline set upstream, this is used to
      extract flags controlling which tokenizer is used.
      Uses sentence piece tokenizer if hps.use_spm_tokenizer = True.
      Uses word piece tokenizer if hps.use_wpm_tokenizer=True.
      Uses simple character level tokenizer if hps.use_character_tokenizer=True.

  Returns:
    outputs tf data features with tokenized transcripts.
  """
  if hps.tokenizer_type == 'WPM':
    tokenizer_input = features['targets'][tf.newaxis, ...]

    def cpu_tokenizer_fn(*args, **kwargs):
      # Try not to annoy TPUs when tokenizers are depending on TF.
      with tf.device('/cpu:0'):
        return tokenizer.strings_to_ids(*args, **kwargs)

    tokens, token_paddings = tf.numpy_function(
        func=cpu_tokenizer_fn,
        inp=[tokenizer_input],
        Tout=[tf.int32, tf.float32])

    features['targets'] = tokens[0, :]
    features['target_paddings'] = token_paddings[0, :]
  elif hps.tokenizer_type == 'SPM':
    features['targets'] = tokenizer.tokenize(features['targets'])
    features['target_paddings'] = tf.zeros_like(
        features['targets'], dtype=tf.float32)
  elif hps.tokenizer_type == 'RAW':
    features['targets'] = tf.py_function(
        func=tokenizer.tokenize, inp=[features['targets']], Tout=tf.int32)
    features['target_paddings'] = tf.zeros_like(
        features['targets'], dtype=tf.float32)

  return features


def _make_input_paddings(features):
  features['input_paddings'] = tf.zeros_like(
      features['inputs'], dtype=tf.float32)
  return features


def _normalize_feature_names(features):
  features['inputs'] = features.pop('speech')
  features['inputs'] = tf.cast(features['inputs'], dtype=tf.int32)
  features['targets'] = features.pop('text')

  features.pop('id')
  features.pop('chapter_id')
  features.pop('speaker_id')

  return features


def get_raw_dataset(dataset_builder: tfds.core.DatasetBuilder,
                    split: str, shuffle_seed=None) -> tf.data.Dataset:
  """Loads the raw dataset and normalizes feature keys.

  Args:
    dataset_builder: TFDS dataset builder that can build `split`.
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.
    shuffle_seed: seed used to shuffle files across splits.

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """
  per_host_split = tfds.split_for_jax_process(split)

  ds = dataset_builder.as_dataset(
      split=per_host_split,
      shuffle_files=(shuffle_seed is not None),
      read_config=tfds.ReadConfig(shuffle_seed=shuffle_seed))
  ds = ds.map(_normalize_feature_names, num_parallel_calls=AUTOTUNE)

  return ds


def preprocess_data(
    dataset,
    train=True,
    shuffle_buffer_size=512,
    hps=None,
    batch_size=256,
    drop_remainder=True,
    prefetch_size=64,
    shuffle_seed=None,
):
  """Process, shuffle, pad and batch the given dataset."""

  max_target_length = hps.max_target_length
  max_input_length = hps.max_input_length

  if hps.tokenizer_type not in ALLOWED_TOKENIZERS:
    raise ValueError(
        'Passed in tokenizer_type value does not correspond to currently '
        'supported tokenizers, make sure one of WPM, SPM or RAW is set'
        ' as tokenizer_type flag.')

  if hps.tokenizer_type == 'WPM':
    tokenizer = wpm_tokenizer.WpmTokenizer(hps.tokenizer_vocab_path)
  elif hps.tokenizer_type == 'SPM':
    tokenizer = spm_tokenizer.load_tokenizer(hps.tokenizer_vocab_path)
  elif hps.tokenizer_type == 'RAW':
    tokenizer = CharacterTokenizer()

  dataset = dataset.map(
      functools.partial(_preprocess_output, tokenizer=tokenizer, hps=hps),
      num_parallel_calls=10)

  dataset = dataset.map(_make_input_paddings, num_parallel_calls=10)

  # Filter out audio and transcriptions that are longer than given lengths.
  # note that audio filtering is post frequency domain conversion.
  if max_input_length > 0 and max_target_length > 0:
    inputs_length_filter, targets_length_filter = length_filter(
        max_input_length, max_target_length)
    dataset = dataset.filter(inputs_length_filter)
    dataset = dataset.filter(targets_length_filter)

  if train:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.repeat()

  dataset = dataset.padded_batch(
      batch_size,
      drop_remainder=drop_remainder,
      padded_shapes={
          'inputs': max_input_length,
          'targets': max_target_length,
          'input_paddings': max_input_length,
          'target_paddings': max_target_length,
      },
      padding_values={
          'inputs': 0,
          'targets': 0,
          'input_paddings': 1.0,
          'target_paddings': 1.0
      })

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_librispeech_datasets(hps, per_host_batch_size, per_host_eval_batch_size,
                             shuffle_rng):
  """Helper method to get train, eval and test sets for librispeech data."""
  train_ds_builder = tfds.builder('librispeech')

  # TODO(b/280322542): should have here
  # rng1, rng2 = jax.random.split(shuffle_rng)
  train_data = get_raw_dataset(train_ds_builder, hps.train_split,
                               # TODO(b/280322542): use jax.random.bits(rng1)
                               jax.random.key_data(shuffle_rng)[0])
  eval_data = get_raw_dataset(train_ds_builder, hps.eval_split)
  test_data = get_raw_dataset(train_ds_builder, hps.test_split)

  train_ds = preprocess_data(
      train_data,
      train=True,
      batch_size=per_host_batch_size,
      hps=hps,
      # TODO(b/280322542): use jax.random.bits(rng2)
      shuffle_seed=jax.random.key_data(shuffle_rng)[1])

  eval_ds = preprocess_data(
      eval_data,
      train=False,
      batch_size=per_host_eval_batch_size,
      hps=hps,
      drop_remainder=False,
  )

  test_ds = preprocess_data(
      test_data,
      train=False,
      batch_size=per_host_eval_batch_size,
      hps=hps,
      drop_remainder=False)

  return train_ds, eval_ds, test_ds
