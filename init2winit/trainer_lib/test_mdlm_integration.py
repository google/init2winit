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

"""Integration test for MDLM training with a patterned fake dataset.

Verifies that the full training loop (model init -> training -> eval) works
end-to-end and that loss decreases on a simple repeating pattern.

"""

import os
import shutil
import tempfile

from absl import logging
from absl.testing import absltest
from init2winit import utils
from init2winit.dataset_lib import data_utils
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainer
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import pandas
import tensorflow.compat.v1 as tf

Dataset = data_utils.Dataset

# Small vocab and sequence length so the test runs quickly on CPU.
_VOCAB_SIZE = 16
_SEQ_LEN = 32
_BATCH_SIZE = 8
_EVAL_NUM_BATCHES = 2


def _make_patterned_batch(batch_size, vocab_size, seq_len):
  """Creates a batch where each row is a cyclic shift of [0, 1, ..., V-1].

  Row i = [(i % V), (i+1 % V), ..., (i+seq_len-1 % V)].
  This gives the model a simple and learnable pattern.

  Args:
    batch_size: Number of sequences in the batch.
    vocab_size: Size of the vocabulary.
    seq_len: Length of each sequence.

  Returns:
    A dict with 'inputs', 'targets', and 'weights'.
  """
  rows = []
  for i in range(batch_size):
    row = [(i + j) % vocab_size for j in range(seq_len)]
    rows.append(row)
  tokens = jnp.array(rows, dtype=jnp.int32)
  return {
      'inputs': tokens,
      'targets': tokens,  # MDLM: inputs == targets.
      'weights': jnp.ones(tokens.shape),
  }


def _get_patterned_mdlm_dataset(batch_size, eval_num_batches):
  """Returns a fake MDLM dataset with a cyclic-shift pattern."""
  batch = _make_patterned_batch(batch_size, _VOCAB_SIZE, _SEQ_LEN)

  def train_iterator_fn():
    while True:
      yield batch

  def eval_train_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def valid_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  def test_epoch(num_batches=None):
    if num_batches is None:
      num_batches = eval_num_batches
    for _ in range(num_batches):
      yield batch

  meta_data = {
      'apply_one_hot_in_loss': False,
      'shift_inputs': False,
      'causal': False,
  }
  return (
      Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch),
      meta_data,
  )


class MDLMIntegrationTest(absltest.TestCase):
  """Integration test: train MDLM and verify loss decreases."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.trainer = None

  def tearDown(self):
    if self.trainer is not None:
      self.trainer.wait_until_orbax_checkpointer_finished()
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_loss_decreases_on_pattern(self):
    """MDLM should learn a trivial cyclic pattern and decrease loss."""
    np.random.seed(0)
    rng = jnp.array(
        np.random.default_rng(0).integers(0, 2**31, size=2), dtype=jnp.uint32
    )

    model_str = 'mdlm_rope_nanodo'
    model_cls = models.get_model(model_str)
    loss_name = 'passthrough'
    metrics_name = 'mdlm_metrics'

    hps = config_dict.ConfigDict({
        'batch_size': _BATCH_SIZE,
        'emb_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'mlp_dim': 64,
        'vocab_size': _VOCAB_SIZE,
        'input_shape': (_SEQ_LEN,),
        'output_shape': (_SEQ_LEN, _VOCAB_SIZE),
        'computation_dtype': 'float32',
        'model_dtype': 'float32',
        'normalization': 'rmsnorm',
        'mlp_activation': 'glu',
        'qk_norm': True,
        'tie_embeddings': True,
        'noise_schedule': 'log_linear',
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0,
        },
        'lr_hparams': {
            'base_lr': 0.003,
            'schedule': 'constant',
        },
        'l2_decay_factor': 0.0,
        'l2_decay_rank_threshold': 2,
        'grad_clip': None,
        'label_smoothing': 0.0,
        'use_shallue_label_smoothing': False,
        'rng_seed': 0,
        'train_size': _BATCH_SIZE * 100,
        'num_device_prefetches': 0,
    })

    dataset, dataset_meta_data = _get_patterned_mdlm_dataset(
        _BATCH_SIZE, _EVAL_NUM_BATCHES
    )
    model = model_cls(hps, dataset_meta_data, loss_name, metrics_name)
    initializer = initializers.get_initializer('noop')

    num_train_steps = 1000
    eval_frequency = 200

    metrics_logger, init_logger = utils.set_up_loggers(self.test_dir)
    self.trainer = trainer.Trainer(
        train_dir=self.test_dir,
        model=model,
        dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
        initializer=initializer,
        num_train_steps=num_train_steps,
        hps=hps,
        rng=rng,
        eval_batch_size=_BATCH_SIZE,
        eval_use_ema=False,
        eval_num_batches=_EVAL_NUM_BATCHES,
        test_num_batches=0,
        eval_train_num_batches=_EVAL_NUM_BATCHES,
        eval_frequency=eval_frequency,
        checkpoint_steps=[],
        metrics_logger=metrics_logger,
        init_logger=init_logger,
    )
    _ = list(self.trainer.train())

    # ---- Check loss trajectory ----
    with tf.io.gfile.GFile(
        os.path.join(self.test_dir, 'measurements.csv')
    ) as f:
      df = pandas.read_csv(f)
      train_cost = df['train_cost'].values
      self.assertGreater(
          train_cost[0],
          train_cost[-1],
          msg=(
              'Expected loss to decrease. '
              f'Initial: {train_cost[0]:.4f}, Final: {train_cost[-1]:.4f}'
          ),
      )
      self.assertLess(
          train_cost[-1],
          0.5,
          msg=(
              'Expected final loss well below random baseline. '
              f'Final: {train_cost[-1]:.4f}'
          ),
      )

      valid_ce = df['valid/ce_loss'].values
      valid_ppl = df['valid/perplexity'].values
      self.assertTrue(
          all(np.isfinite(valid_ce)),
          msg=f'valid/ce_loss contains non-finite: {valid_ce}',
      )
      self.assertTrue(
          all(np.isfinite(valid_ppl)),
          msg=f'valid/perplexity contains non-finite: {valid_ppl}',
      )
      self.assertLess(
          valid_ce[-1],
          valid_ce[0],
          msg=(
              'Expected valid/ce_loss to decrease. '
              f'Initial: {valid_ce[0]:.4f}, Final: {valid_ce[-1]:.4f}'
          ),
      )
      self.assertGreater(
          valid_ppl[0],
          valid_ppl[-1],
          msg=(
              'Expected valid/perplexity to decrease. '
              f'Initial: {valid_ppl[0]:.4f}, Final: {valid_ppl[-1]:.4f}'
          ),
      )

    # ---- Verify evaluate_batch ----
    params = self.trainer.get_params()
    batch = _make_patterned_batch(_BATCH_SIZE, _VOCAB_SIZE, _SEQ_LEN)
    batch['eval_rng'] = jax.random.PRNGKey(42)
    eval_metrics = model.evaluate_batch(params, batch_stats=None, batch=batch)
    eval_results = eval_metrics.compute()
    self.assertTrue(np.isfinite(eval_results['ce_loss']))
    self.assertTrue(np.isfinite(eval_results['perplexity']))
    logging.info(
        'Direct evaluate_batch: ce_loss=%.4f, perplexity=%.4f',
        eval_results['ce_loss'],
        eval_results['perplexity'],
    )

    # ---- Print model predictions vs ground truth using inference() ----
    rng = jax.random.PRNGKey(42)
    predictions, z_masked, is_masked, t_b = model.inference(params, batch, rng)
    logging.info('===== Model Predictions vs Ground Truth (sampled t) =====')
    num_examples = min(4, _BATCH_SIZE)
    for i in range(num_examples):
      gt = batch['targets'][i].tolist()
      pred = predictions[i].tolist()
      masked = np.array(is_masked[i])
      inp = z_masked[i].tolist()
      n_masked = int(masked.sum())
      n_correct = int((np.array(pred)[masked] == np.array(gt)[masked]).sum())
      logging.info(
          'Example %d (t=%.3f, %d/%d masked tokens correct):\n'
          '  Original:   %s\n'
          '  Masked in:  %s\n'
          '  Predicted:  %s',
          i,
          float(t_b[i]),
          n_correct,
          n_masked,
          gt,
          inp,
          pred,
      )


if __name__ == '__main__':
  absltest.main()
