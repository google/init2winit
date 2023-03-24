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

"""Contains Tokenizer class for word level tokenization.

Note that the current tokenization workflow is not yet optimized for time and
memory yet.

"""

import tensorflow as tf

EOS_TOKEN = b'<eos>'
UNKNOWN_TOKEN = b'<unk>'


class _Dictionary:
  """Dictionary contains word-to-id mappings and id-to-word mappings.

  Attributes:
    word2idx: dict containing key-values where keys are words and values are
      tokens.
    idx2word: list where the index of each word in the list is the token value.
  """

  def __init__(self):
    self.word2idx = {}
    self.idx2word = []

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      # Start the first token idx at 1, because 0 is reserved for special tokens
      # e.g. for padding and masking
      self.word2idx[word] = len(self.idx2word)
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)


class Tokenizer:
  """Tokenizer object for word level tokenization from words to unique ids.

  Attributes:
    dictionary: Dictionary containing word-to-id and id-to-word mappings
  """

  def __init__(self):
    self.dictionary = _Dictionary()

  def train(self, dataset: tf.data.TextLineDataset):
    """Trains a Tokenizer from a TextLineDataset."""
    # Add words to the dictionary
    for line in dataset:
      words = line.numpy().split() + [EOS_TOKEN]
      for word in words:
        self.dictionary.add_word(word)

  def tokenize(self, dataset: tf.data.TextLineDataset) -> tf.data.Dataset:
    """Tokenizes a TextLineDataset."""
    idss = []
    for line in dataset:
      ids = []
      words = line.numpy().split() + [EOS_TOKEN]
      for word in words:
        try:
          ids.append(self.dictionary.word2idx[word])
        except KeyError:
          ids.append(self.dictionary.word2idx[UNKNOWN_TOKEN])
      idss.append(ids)
    ids = tf.concat(idss, 0)

    tokenized_dataset = tf.data.Dataset.from_tensor_slices(ids)

    return tokenized_dataset
  