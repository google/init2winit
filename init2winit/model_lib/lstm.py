# coding=utf-8
# Copyright 2024 The init2winit Authors.
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

"""Module with flax LSTM class.

"""
import abc
import functools
from typing import (Any, Mapping, Optional, Sequence, Tuple, Type, Union)

import flax
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
  """Flips a sequence of inputs along the time dimension.

  This function can be used to prepare inputs for the reverse direction of a
  bidirectional LSTM. It solves the issue that, when naively flipping multiple
  padded sequences stored in a matrix, the first elements would be padding
  values for those sequences that were padded. This function keeps the padding
  at the end, while flipping the rest of the elements.

  Example:
  ```python
  inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
  lengths = [1, 2, 3]
  flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
  ```

  Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

  Returns:
    An ndarray with the flipped inputs.
  """
  # Compute the indices to put the inputs in flipped order as per above example.
  max_length = inputs.shape[0]
  idxs = (jnp.arange(max_length - 1, -1, -1) + lengths) % max_length
  return inputs[idxs]


def sample_recurrent_dropout_mask(rng: Any, rate: float, batch_size: int,
                                  hidden_size: int) -> Optional[Array]:
  """Samples a recurrent dropout mask."""
  if rate == 0.:
    return None
  mask = random.bernoulli(rng, p=1 - rate, shape=(batch_size, hidden_size))
  # Scale recurrent dropout mask to control for magnitude at test time.
  return mask / (1.0 - rate)


class RecurrentDropoutCell(abc.ABC):
  """Interface for cells that know how to apply recurrent dropout."""

  def __call__(self,
               cell_state: StateType,
               inputs: Array,
               recurrent_dropout_mask: Optional[Array],
               deterministic: bool = False):
    pass

  def get_recurrent_dropout_mask(self, rate: float, batch_size: int,
                                 hidden_size: int):
    pass


class RecurrentDropoutOptimizedLSTMCell(nn.OptimizedLSTMCell,
                                        RecurrentDropoutCell):
  """An optimized LSTM cell that applies recurrent dropout on h (and not c)."""

  @nn.compact
  def __call__(self,  # pytype: disable=signature-mismatch  # jax-ndarray
               cell_state: Tuple[Array, Array],
               inputs: Array,
               recurrent_dropout_mask: Optional[Array] = None,
               deterministic: bool = False):
    """Applies recurrent dropout on h in the state and performs one step."""
    if not deterministic and recurrent_dropout_mask is not None:
      c, h = cell_state
      cell_state = (c, h * recurrent_dropout_mask)

    return super().__call__(cell_state, inputs)  # pylint: disable=no-value-for-parameter

  def get_recurrent_dropout_mask(self, rate: float, batch_size: int,
                                 hidden_size: int):
    """Returns a recurrent dropout mask for this cell."""
    rng = self.make_rng('dropout')
    return sample_recurrent_dropout_mask(rng, rate, batch_size, hidden_size)


class GenericRNNSequenceEncoder(nn.Module):
  """Encodes a single sequence using any RNN cell, for example `nn.LSTMCell`.

  The sequence can be encoded left-to-right (default) or right-to-left (by
  calling the module with reverse=True). Regardless of encoding direction,
  outputs[i, j, ...] is the representation of inputs[i, j, ...].

  Attributes:
    hidden_size: The hidden size of the RNN cell.
    cell_type: The RNN cell module to use, for example, `nn.LSTMCell`.
    cell_kwargs: Optional keyword arguments for the recurrent cell.
    recurrent_dropout_rate: The dropout to apply across time steps. If this is
      greater than zero, you must use an RNN cell that implements
      `RecurrentDropoutCell` such as RecurrentDropoutOptimizedLSTMCell.
  """
  hidden_size: int
  cell_type: Type[nn.RNNCellBase]
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()
  recurrent_dropout_rate: float = 0.0

  def setup(self):
    self.cell = self.cell_type(features=self.hidden_size, **self.cell_kwargs)

  @functools.partial(  # Repeatedly calls the below method to encode the inputs.
      nn.transforms.scan,
      variable_broadcast=('params', 'params_axes'),
      in_axes=(1, flax.core.axes_scan.broadcast, flax.core.axes_scan.broadcast),
      out_axes=1,
      split_rngs={'params': False})
  def unroll_cell(self, cell_state: StateType, inputs: Array,
                  recurrent_dropout_mask: Optional[Array], deterministic: bool):
    """Unrolls a recurrent cell over an input sequence.

    Args:
      cell_state: The initial cell state, shape: <float32>[batch_size,
        hidden_size] (or an n-tuple thereof).
      inputs: The input sequence. <float32>[batch_size, seq_len, input_dim].
      recurrent_dropout_mask: An optional recurrent dropout mask to apply in
        between time steps. <float32>[batch_size, hidden_size].
      deterministic: Disables recurrent dropout when set to True.

    Returns:
      The cell state after processing the complete sequence (including padding),
      and a tuple with all intermediate cell states and cell outputs.
    """
    # We do not directly scan the cell itself, since it only returns the output.
    # This returns both the state and the output, so we can slice out the
    # correct final states later.
    if isinstance(self.cell, RecurrentDropoutCell):
      new_cell_state, output = self.cell(cell_state, inputs,
                                         recurrent_dropout_mask, deterministic)
    else:
      new_cell_state, output = self.cell(cell_state, inputs)

    return new_cell_state, (new_cell_state, output)

  def __call__(self,
               inputs: Array,
               lengths: Array,
               initial_state: StateType,
               reverse: bool = False,
               deterministic: bool = False):
    """Unrolls the RNN cell over the inputs.

    Arguments:
      inputs: A batch of sequences. Shape: <float32>[batch_size, seq_len,
        input_dim].
      lengths: The lengths of the input sequences.
      initial_state: The initial state for the RNN cell. Shape: [batch_size,
        hidden_size].
      reverse: Process the inputs in reverse order, and reverse the outputs.
        This means that the outputs still correspond to the order of the inputs,
        but their contexts come from the right, and not from the left.
      deterministic: Disables recurrent dropout if set to True.

    Returns:
      The encoded sequence of inputs, shaped <float32>[batch_size, seq_len,
        hidden_size], as well as the final hidden states of the RNN cell. For an
        LSTM cell the final states are a tuple (c, h), each shaped <float32>[
          batch_size, hidden_size].
    """
    if reverse:
      inputs = flip_sequences(inputs, lengths)

    # Sample a recurrent dropout mask if recurrent dropout is requested.
    if self.recurrent_dropout_rate > 0. and not deterministic:
      if not isinstance(self.cell, RecurrentDropoutCell):
        raise ValueError(
            ('The provided cell does not support recurrent dropout, but '
             f'recurrent_dropout_rate is set to {self.recurrent_dropout_rate}. '
             'Please provide a cell that implements `RecurrentDropoutCell`.'))

      recurrent_dropout_mask = self.cell.get_recurrent_dropout_mask(
          rate=self.recurrent_dropout_rate,
          batch_size=inputs.shape[0],
          hidden_size=self.hidden_size)
    else:
      recurrent_dropout_mask = None

    _, (cell_states, outputs) = self.unroll_cell(initial_state, inputs,
                                                 recurrent_dropout_mask,
                                                 deterministic)

    final_state = jax.tree_map(
        lambda x: x[jnp.arange(inputs.shape[0]), lengths - 1], cell_states)

    if reverse:
      outputs = flip_sequences(outputs, lengths)

    return outputs, final_state


class GenericRNN(nn.Module):
  """Generic RNN class.

  This provides generic RNN functionality to encode sequences with any RNN cell.
  The class provides unidirectional and bidirectional layers, and these are
  stacked when asking for multiple layers.

  This class be used to create a specific RNN class such as LSTM or GRU.

  Attributes:
    cell_type: An RNN cell class to use, e.g., `flax.linen.LSTMCell`.
    hidden_sizes: Per layer size of each recurrent cell.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers=len(hidden_sizes) > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    residual_connections: Add residual connection between layers.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  cell_type: Type[nn.RNNCellBase]
  hidden_sizes: Sequence[int]
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  residual_connections: bool = False
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      lengths: Array,
      initial_states: Optional[Sequence[StateType]] = None,
      deterministic: bool = False) -> Tuple[Array, Sequence[StateType]]:
    """Processes the input sequence using the recurrent cell.

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states, where `num_layers=len(hidden_sizes)` (when
        using bidirectional, `num_layers * 2`). These must be ordered in the
        following way: (layer_0_forward, layer_0_backward, layer_1_forward,
        layer_1_backward, ...). If None, all initial states will be initialized
        with zeros.
      deterministic: Disables dropout between layers when set to True.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states for each cell and direction. Directions are alternated (first
      forward, then backward, if bidirectional). For example for a bidirectional
      cell this would be: layer 1 forward, layer 1 backward, layer 2 forward,
      layer 2 backward, etc..
      For some cells like LSTMCell a state consists of an (c, h) tuple, while
      for others cells it only contains a single vector (h,).
    """
    num_layers = len(self.hidden_sizes)
    batch_size = inputs.shape[0]
    final_states = []
    num_directions = 2 if self.bidirectional else 1
    num_cells = num_layers * num_directions

    if self.bidirectional:
      # pylint: disable=g-complex-comprehension
      hidden_size_per_cell = [h for h in self.hidden_sizes for _ in range(2)]
    else:
      hidden_size_per_cell = self.hidden_sizes

    # Construct initial states.
    if initial_states is None:  # Initialize with zeros.
      rng = jax.random.PRNGKey(0)
      initial_states = [
          self.cell_type(hidden_size, parent=None).initialize_carry(
              rng, (batch_size, 1)
          )
          for hidden_size in hidden_size_per_cell
      ]
    assert len(initial_states) == num_cells, (
        f'Please provide {self.num_cells} (`num_layers`, *2 if bidirectional) '
        'initial states.'
    )

    # For each layer, apply the forward and optionally the backward RNN cell.
    cell_idx = 0
    for hidden_size in self.hidden_sizes:
      # Process sequence in forward direction through an RNN Cell for this
      # layer.
      outputs, final_state = GenericRNNSequenceEncoder(
          cell_type=self.cell_type,
          cell_kwargs=self.cell_kwargs,
          hidden_size=hidden_size,
          recurrent_dropout_rate=self.recurrent_dropout_rate,
          name=f'{self.name}SequenceEncoder_{cell_idx}',
      )(
          inputs,
          lengths,
          initial_state=initial_states[cell_idx],
          deterministic=deterministic,
      )
      final_states.append(final_state)
      cell_idx += 1

      # Process sequence in backward direction through an RNN Cell for this
      # layer.
      if self.bidirectional:
        backward_outputs, backward_final_state = GenericRNNSequenceEncoder(
            cell_type=self.cell_type,
            cell_kwargs=self.cell_kwargs,
            hidden_size=hidden_size,
            recurrent_dropout_rate=self.recurrent_dropout_rate,
            name=f'{self.name}SequenceEncoder_{cell_idx}',
        )(
            inputs,
            lengths,
            initial_state=initial_states[cell_idx],
            reverse=True,
            deterministic=deterministic,
        )
        outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)
        final_states.append(backward_final_state)
        cell_idx += 1

      # For the next layer, current outputs become the inputs.
      if self.residual_connections:
        assert inputs.shape == outputs.shape, (
            f'For residual connections, inputs ({inputs.shape}) and '
            f'outputs ({outputs.shape}) must be the same shape.')
        inputs += outputs
      else:
        inputs = outputs

      # Apply dropout between layers.
      inputs = nn.Dropout(
          rate=self.dropout_rate, deterministic=deterministic)(
              inputs)

    return outputs, final_states


class LSTM(nn.Module):
  """LSTM.

  Attributes:
    hidden_size: The size of each recurrent cell.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers=len(hidden_sizes) > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    residual_connections: Add residual connection between layers.
    cell_type: The LSTM cell class to use. Default:
      `flax.linen.OptimizedLSTMCell`. If you use hidden_size of >2048, consider
      using `flax.linen.LSTMCell` instead, since the optimized LSTM cell works
      best for hidden sizes up to 2048.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  hidden_sizes: Sequence[int]
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  residual_connections: bool = False
  cell_type: Any = RecurrentDropoutOptimizedLSTMCell
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      lengths: Array,
      initial_states: Optional[Sequence[StateType]] = None,
      deterministic: bool = False) -> Tuple[Array, Sequence[StateType]]:
    """Processes an input sequence with an LSTM cell.

    Example usage:
    ```
      inputs = np.random.normal(size=(2, 3, 4))
      lengths = np.array([1, 3])
      outputs, final_states = LSTM(hidden_size=10).apply(rngs, inputs, lengths)
    ```

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states, where `num_layers=len(hidden_sizes)` (when
        using bidirectional, `num_layers * 2`). These must be ordered in the
        following way: (layer_0_forward, layer_0_backward, layer_1_forward,
        layer_1_backward, ...). If None, all initial states will be initialized
        with zeros.
      deterministic: Disables dropout between layers when set to True.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states (h, c) for each cell and direction, ordered first by layer number
      and then by direction (first forward, then backward, if bidirectional).
    """
    return GenericRNN(
        cell_type=self.cell_type,
        hidden_sizes=self.hidden_sizes,
        dropout_rate=self.dropout_rate,
        recurrent_dropout_rate=self.recurrent_dropout_rate,
        bidirectional=self.bidirectional,
        residual_connections=self.residual_connections,
        cell_kwargs=self.cell_kwargs,
        name='LSTM',
    )(
        inputs,
        lengths,
        initial_states=initial_states,
        deterministic=deterministic,
    )
