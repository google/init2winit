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
from typing import Any, Mapping, Tuple, Union

import flax
from flax import linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib.lstm import LSTM
from jax import lax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]

MASK_TOKEN = 0

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        # training params
        batch_size=256,
        rng_seed=-1,
        # model architecture params
        model_dtype='float32',
        bidirectional=False,
        residual_connections=False,
        cell_kwargs={},
        emb_dim=320,
        hidden_size=1024,
        num_layers=3,
        dropout_rate=0.1,
        recurrent_dropout_rate=0.1,
        # optimizer params
        lr_hparams={
            'schedule': 'constant',
            'base_lr': 1e-3,
        },
        l2_decay_factor=None,
        grad_clip=None,
        optimizer='adam',
        opt_hparams={
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0,
        },
    ))


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(MASK_TOKEN))
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


class LSTMLM(nn.Module):
  """Class for LSTM flax linen model with embedding and decoding layer.

  Attributes:
    emb_dim: Embedding dimension
    vocab_size: size of vocab of tokens to be embedded
    hidden_size: size of LSTM cell
    num_layers: The number of stacked recurrent layers. The output of first
      layer, with optional droput applied, feeds into the next layer.
    dropout_rate: Dropout rate to be applied between LSTM layers and
      on output of final LSTM layer.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      contatenate the outputs of the two directions.
    residual_connections: Add residual connection between layers.
    cell_type: The LSTM cell class to use. Default
      `flax.linen.OptimizedLSTMCell. If you use hidden_size of >2048, consider
      using `flax.linen.LSTMCell` instead.
    cell_kwargs: Optional keyword arguments to instatiate the cell with.
  """
  emb_dim: int
  vocab_size: int
  hidden_size: int
  num_layers: int = 1
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  residual_connections: bool = False
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

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
    # Shift inputs tokens to the right by 1 and pad from left w MASK_TOKEN
    inputs = shift_right(inputs)
    # Embed input sequences, resulting shape is (batch_size, seq_len, emb_dim)
    embedded_inputs = nn.Embed(
        num_embeddings=self.vocab_size + 1, features=self.emb_dim)(inputs)
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
        bidirectional=self.bidirectional,
        dropout_rate=self.dropout_rate,
        recurrent_dropout_rate=self.recurrent_dropout_rate,
        residual_connections=self.residual_connections,
        cell_kwargs=self.cell_kwargs)(
            embedded_inputs,
            lengths=jnp.sum(jnp.ones(embedded_inputs.shape[:-1],
                                     dtype=jnp.int32), axis=1),
            deterministic=(not train))
    # Apply dropout on outputs
    lstm_output = nn.Dropout(
        rate=self.dropout_rate, deterministic=(not train))(lstm_output)
    # Decode outputs to vector of size vocab_size
    logits = nn.Dense(self.vocab_size + 1)(lstm_output)
    return logits


class LSTMModel(base_model.BaseModel):
  """LSTM model class."""

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates metrics on the given batch.

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
    return LSTMLM(
        emb_dim=self.hps.emb_dim,
        vocab_size=self.hps.vocab_size,
        hidden_size=self.hps.hidden_size,
        num_layers=self.hps.num_layers,
        dropout_rate=self.hps.dropout_rate,
        recurrent_dropout_rate=self.hps.recurrent_dropout_rate,
        bidirectional=self.hps.bidirectional,
        residual_connections=self.hps.residual_connections,
        cell_kwargs=self.hps.cell_kwargs,
    )

  def get_fake_inputs(self, hps):
    dummy_inputs = jnp.ones((hps.batch_size, hps.sequence_length),
                            dtype='int32')
    return [dummy_inputs]
