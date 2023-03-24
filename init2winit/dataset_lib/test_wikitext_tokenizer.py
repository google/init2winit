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

"""Tests for init2winit.dataset_lib.wikitext103."""

from absl.testing import absltest
from init2winit.dataset_lib import wikitext_tokenizer
import tensorflow as tf



class TestWikitextTokenizer(absltest.TestCase):
  """Unit tests for wikitext103.py."""

  def test_tokenizer_vocab_size(self):
    """Test vocab size.

    Vocab size should be number of unique words in text file + 1 for the <eos>
    token which gets added at the end of each new line.
    """
    # Get number of unique tokens from tokenizer.
    text_dataset = tf.data.TextLineDataset(file_name)

    tokenizer = wikitext_tokenizer.Tokenizer()
    tokenizer.train(text_dataset)

    num_unique_tokens = len(tokenizer.dictionary.idx2word)

    # Get number of unique words from fake data.
    with open(file_name, 'r') as f:
      data = ''
      for line in f:
        # Not removing this would count tokens like '\n\n' and '\n' while the
        # TextLineDataset strips them.
        line = line.strip('\n')
        data = data + line

    words = data.split(' ')
    num_unique_words = len(set(words))

    self.assertEqual(num_unique_tokens, num_unique_words + 1)

if __name__ == '__main__':
  absltest.main()
