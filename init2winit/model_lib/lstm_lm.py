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

"""This module defines LSTMModel class for language modeling.

Inspired by
https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""
from typing import Tuple, Union

from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib.lstm import LSTM
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        batch_size=32,
        bidirectional=False,
        dropout_rate=0.2,
        emb_dim=200,
        hidden_size=200,
        layer_rescale_factors={},
        lr_hparams={
            'schedule': 'constant',
            'base_lr': 0.1,
        },
        l2_decay_factor=None,
        model_dtype='float32',
        optimizer='sgd',
        opt_hparams={
            'momentum': 0.9,
            'learning_rate': 0.1,
        },
        num_layers=1,
        rng_seed=-1,
        grad_clip=0.25,
    ))


class WrappedLSTM(nn.Module):
  """Class for LSTM flax linen model with embedding and decoding layer.

  Attributes:
    emb_dim: embedding dimension
    vocab_size: size of vocab of tokens to be embedded
    hidden_size: size of LSTM cell
    embedding_layer: nn.Embed class
    lstm: flax_nlp.recurrent LSTM class
    decoder: nn.Decoder class
  """
  emb_dim: int
  vocab_size: int
  hidden_size: int
  num_layers: int
  dropout_rate: float
  bidirectional: bool

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      train: bool,) -> Array:
    """Returns output of LSTM model on input sequence.

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      train: Whether the model is training
    Returns:
      Decoded outputs for the final LSTM layer.
    """
    # Embed input sequences, resulting shape is (batch_size, seq_len, emb_dim)
    embedded_inputs = nn.Embed(
        num_embeddings=self.vocab_size, features=self.emb_dim)(inputs)
    # Dropout on embeddings
    embedded_inputs = nn.Dropout(
        rate=self.dropout_rate, deterministic=(not train))(
            embedded_inputs)
    # Apply LSTM on inputs embeddings
    # Output of the LSTM layer is a sequence of all outputs of the final layer
    # and a list of final states (h, c) for each cell and direction
    lstm_output, _ = LSTM(
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        bidirectional=self.bidirectional,)(
            embedded_inputs,
            lengths=jnp.sum(jnp.ones(embedded_inputs.shape[:-1],
                                     dtype=jnp.int32), axis=1),
            deterministic=(not train))
    # Apply dropout on outputs
    lstm_output = nn.Dropout(
        rate=self.dropout_rate, deterministic=(not train))(lstm_output)
    # Decode outputs to vector of size vocab_size
    logits = nn.Dense(self.vocab_size)(lstm_output)
    return logits


class LSTMModel(base_model.BaseModel):
  """LSTM model class."""

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates metrics on the given batch.

    This method uses the class method apply_on_batch instead of
    flax_module.apply because the flax_module. The apply_on_batch method
    handles the 'length' input into the flax_nlp LSTM inner module.

    We use the CLU metrics library to evaluate the metrics, and we require
    that each metric_fn in metrics_bundle has the API:
      metric_fn(logits, targets, weights), including the argument names.

    Args:
      params: A dict of trainable model parameters. Passed as
      {'params': params} into flax_module.apply().
      batch_stats: A dict of non-trainable model state. Passed as
        {'batch_stats': batch_stats} into flax_module.apply().
      batch: A dictionary with keys 'inputs', 'targets', 'weights'.

    Returns:
      A dictionary with the same keys as metrics, but mapping to the summed
      metric across the sharded batch_dim.
    """

    logits = self.apply_on_batch(params, batch_stats, batch, train=False)
    targets = batch['targets']

    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = nn.one_hot(batch['targets'], logits.shape[-1])

    eval_batch_size = targets.shape[0]
    weights = batch.get('weights')  # Weights might not be defined.
    if weights is None:
      weights = jnp.ones(eval_batch_size)

    # We don't use CLU's `mask` argument here, we handle it ourselves through
    # `weights`.
    return self.metrics_bundle.gather_from_model_output(
        logits=logits, targets=targets, weights=weights, axis_name='batch')

  def build_flax_module(self):
    return WrappedLSTM(
        emb_dim=self.hps.emb_dim,
        vocab_size=self.hps.vocab_size,
        hidden_size=self.hps.hidden_size,
        num_layers=self.hps.num_layers,
        dropout_rate=self.hps.dropout_rate,
        bidirectional=self.hps.bidirectional)

  def get_fake_batch(self, hps):
    dummy_inputs = jnp.ones((hps.batch_size, hps.sequence_length),
                            dtype='int32')
    return [dummy_inputs]
