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

"""Tests for init2winit.dataset_lib.wikitext103."""

from absl.testing import absltest
from init2winit.dataset_lib import wikitext103
from jax import random


class TestWikitext103(absltest.TestCase):
  """Unit tests for wikitext103.py."""

  def test_vocab_size(self):
    """Test vocab size."""
    wikitext103_hps = wikitext103.DEFAULT_HPARAMS
    dataset = wikitext103.get_wikitext103(
        shuffle_rng=random.PRNGKey(0),
        batch_size=1,
        eval_batch_size=1,
        hps=wikitext103_hps,
    )

    tokens = set()

    for batch in dataset.eval_train_epoch():
      inputs = batch['inputs']
      targets = batch['targets']

      # pylint: disable=g-complex-comprehension
      inputs_flat = [item for sublist in inputs for item in sublist]
      targets_flat = [item for sublist in targets for item in sublist]

      inputs_set = set(inputs_flat)
      targets_set = set(targets_flat)

      tokens = tokens.union(inputs_set, targets_set)

    # Subtract 1 for the padding token
    num_tokens = len(tokens) - 1

    self.assertLen(num_tokens, wikitext103_hps.vocab_size)

if __name__ == '__main__':
  absltest.main()
