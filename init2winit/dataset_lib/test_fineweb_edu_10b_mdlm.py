# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Tests for fineweb_edu_10b_mdlm dataset wrapper.

Verifies that the MDLM variant does NOT shift inputs (inputs == targets)
and that padding is applied correctly.

"""

from unittest import mock

from absl.testing import absltest
from init2winit.dataset_lib import fineweb_edu_10b_input_pipeline as input_pipeline
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf


def _make_fake_token_dataset(num_tokens):
  """Creates a fake dataset mimicking the tokenized FineWeb-Edu format.

  Args:
    num_tokens: Number of tokens in the dataset.

  Returns:
    A tf.data.Dataset where each element is a dict with 'input_ids' as a 1D
    tensor (one document). The pipeline flat_maps these into a stream of
    individual token dicts, then re-batches into sequences.
  """
  # Simulate a single "document" containing all tokens.
  tokens = np.arange(num_tokens, dtype=np.int64)
  ds = tf.data.Dataset.from_tensor_slices({'input_ids': [tokens]})
  return ds


def _make_hps(seq_len=8):
  return config_dict.ConfigDict(
      dict(
          sequence_length=seq_len,
          max_target_length=seq_len,
          max_eval_target_length=seq_len,
          eval_sequence_length=seq_len,
      )
  )


class InputPipelineShiftTest(absltest.TestCase):
  """Tests that shift=True produces shifted data and shift=False does not."""

  def _get_first_batch(self, shift, seq_len=8, num_tokens=100):
    """Runs the pipeline on fake data and returns the first train batch."""
    hps = _make_hps(seq_len)
    fake_ds = _make_fake_token_dataset(num_tokens)

    with mock.patch.object(tf.data.Dataset, 'load', return_value=fake_ds):
      train_ds, _, _ = input_pipeline.get_fineweb_edu_dataset(
          hps,
          train_batch_size=2,
          valid_batch_size=2,
          shuffle_seed=0,
          shift=shift,
      )
    # Take one batch (undo repeat).
    batch = next(iter(train_ds.take(1)))
    inputs = batch['inputs'].numpy()
    targets = batch['targets'].numpy()
    return inputs, targets

  def test_shift_true_targets_offset_by_one(self):
    """With shift=True (AR), targets[i] == inputs[i] + 1."""
    inputs, targets = self._get_first_batch(shift=True)

    # Inputs and targets should differ.
    self.assertFalse(np.array_equal(inputs, targets))

    # For contiguous token ids, targets should be inputs shifted by 1.
    np.testing.assert_array_equal(targets, inputs + 1)

  def test_shift_false_inputs_equal_targets(self):
    """With shift=False (MDLM), inputs == targets exactly."""
    inputs, targets = self._get_first_batch(shift=False)
    np.testing.assert_array_equal(inputs, targets)

  def test_shift_false_sequence_length_correct(self):
    """With shift=False, sequences should be seq_len long, not seq_len+1."""
    seq_len = 8
    inputs, targets = self._get_first_batch(shift=False, seq_len=seq_len)
    self.assertEqual(inputs.shape[-1], seq_len)
    self.assertEqual(targets.shape[-1], seq_len)

  def test_shift_true_sequence_length_correct(self):
    """With shift=True, sequences come from seq_len+1 tokens."""
    seq_len = 8
    inputs, targets = self._get_first_batch(shift=True, seq_len=seq_len)
    self.assertEqual(inputs.shape[-1], seq_len)
    self.assertEqual(targets.shape[-1], seq_len)


class PaddingTest(absltest.TestCase):
  """Tests that eval batches are padded correctly."""

  def test_eval_batch_padding_applied(self):
    """Eval batches should be padded to batch_size with PAD_ID."""
    hps = _make_hps(seq_len=4)
    # 13 tokens -> 3 sequences of length 4 (shift=False, drop_remainder=True).
    # With batch_size=2, we get 1 full batch + 1 partial batch (1 real + 1 pad).
    fake_ds = _make_fake_token_dataset(13)

    with mock.patch.object(tf.data.Dataset, 'load', return_value=fake_ds):
      _, _, valid_ds = input_pipeline.get_fineweb_edu_dataset(
          hps,
          train_batch_size=2,
          valid_batch_size=2,
          shuffle_seed=0,
          shift=False,
      )

    batches = list(valid_ds.as_numpy_iterator())
    # Should have 2 batches: first full, second padded.
    self.assertLen(batches, 2)

    padded_batch = batches[1]
    pad_id = int(input_pipeline.PAD_ID.numpy())

    # The second row of the padded batch should be all PAD_ID.
    np.testing.assert_array_equal(padded_batch['inputs'][1], np.full(4, pad_id))
    np.testing.assert_array_equal(
        padded_batch['targets'][1], np.full(4, pad_id)
    )

  def test_eval_batch_padding_not_in_full_batches(self):
    """Full eval batches should contain no padding."""
    hps = _make_hps(seq_len=4)
    fake_ds = _make_fake_token_dataset(13)

    with mock.patch.object(tf.data.Dataset, 'load', return_value=fake_ds):
      _, _, valid_ds = input_pipeline.get_fineweb_edu_dataset(
          hps,
          train_batch_size=2,
          valid_batch_size=2,
          shuffle_seed=0,
          shift=False,
      )

    batches = list(valid_ds.as_numpy_iterator())
    full_batch = batches[0]
    pad_id = int(input_pipeline.PAD_ID.numpy())

    # No element in the full batch should be PAD_ID.
    self.assertTrue(np.all(full_batch['inputs'] != pad_id))
    self.assertTrue(np.all(full_batch['targets'] != pad_id))


if __name__ == '__main__':
  absltest.main()
