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

r"""Deepspeech.

This model uses a deepspeech2 network to convert speech to text.
paper : https://arxiv.org/abs/1512.02595

# BiLSTM code contributed by bastings@
# github : https://github.com/bastings
# webpage : https://bastings.github.io/
"""

import functools
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, Union

import flax
from flax import linen as nn
from flax import struct
from init2winit.model_lib import base_model
from init2winit.model_lib import librispeech_preprocessor as preprocessor
from init2winit.model_lib import spectrum_augmenter
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

MLCOMMONS_DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        activation_function='relu',
        optimizer='adam',
        opt_hparams={
            'beta1': .9,
            'beta2': .98,
            'epsilon': 1e-9,
            'weight_decay': 0.0
        },
        batch_size=256,
        eval_batch_size=128,
        l2_decay_factor=1e-6,
        l2_decay_rank_threshold=0,
        use_shallue_label_smoothing=False,
        rng_seed=-1,
        model_dtype='float32',
        grad_clip=10.0,
        num_lstm_layers=4,
        num_ffn_layers=3,
        encoder_dim=512,
        freq_mask_count=2,
        freq_mask_max_bins=27,
        time_mask_count=10,
        time_mask_max_frames=40,
        time_mask_max_ratio=0.05,
        time_masks_per_frame=0.0,
        use_dynamic_time_mask_max_frames=True,
        use_specaug=True,
        aux_dropout_rate=0.1,
        dropout_rate=0.1,
        tie_dropouts=False,
        enable_residual_connections=False,
        enable_decoder_layer_norm=False,
        bidirectional=True,
        total_accumulated_batch_size=None,
        enable_subsampling_batchnorm=False,
        enable_synced_batchnorm=False,
        enable_ffn_relu=False,
        layernorm_everywhere=False))


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        activation_function='relu',
        optimizer='adam',
        opt_hparams={
            'beta1': .9,
            'beta2': .98,
            'epsilon': 1e-9,
            'weight_decay': 0.0
        },
        batch_size=256,
        eval_batch_size=128,
        l2_decay_factor=1e-6,
        l2_decay_rank_threshold=0,
        use_shallue_label_smoothing=False,
        rng_seed=-1,
        model_dtype='float32',
        grad_clip=10.0,
        num_lstm_layers=4,
        num_ffn_layers=3,
        encoder_dim=512,
        freq_mask_count=2,
        freq_mask_max_bins=27,
        time_mask_count=10,
        time_mask_max_frames=40,
        time_mask_max_ratio=0.05,
        time_masks_per_frame=0.0,
        use_dynamic_time_mask_max_frames=True,
        use_specaug=True,
        input_dropout_rate=0.1,
        feed_forward_dropout_rate=0.1,
        enable_residual_connections=False,
        enable_decoder_layer_norm=False,
        bidirectional=True,
        total_accumulated_batch_size=None,
        enable_subsampling_batchnorm=False,
        enable_synced_batchnorm=False,
        enable_ffn_relu=False,
        layernorm_everywhere=False))


@struct.dataclass
class DeepspeechConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int = 0
  dtype: Any = jnp.float32
  encoder_dim: int = 0
  num_lstm_layers: int = 3
  num_ffn_layers: int = 2
  conv_subsampling_factor: int = 2
  conv_subsampling_layers: int = 2
  train: bool = False
  use_specaug: bool = False
  freq_mask_count: int = 1
  freq_mask_max_bins: int = 15
  time_mask_count: int = 1
  time_mask_max_frames: int = 50
  time_mask_max_ratio: float = 1.0
  time_masks_per_frame: float = 0.0
  use_dynamic_time_mask_max_frames: bool = False
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  input_dropout_rate: float = 0.1
  feed_forward_dropout_rate: float = 0.1
  enable_residual_connections: bool = False
  enable_decoder_layer_norm: bool = False
  bidirectional: bool = False
  enable_subsampling_batchnorm: bool = False
  enable_synced_batchnorm: bool = False
  enable_ffn_relu: bool = False
  layernorm_everywhere: bool = False


class Subsample(nn.Module):
  """Module to perform strided convolution in order to subsample inputs.

  Attributes:
    encoder_dim: model dimension of conformer.
    input_dropout_rate: dropout rate for inputs.
  """
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, output_paddings, train):
    config = self.config
    outputs = jnp.expand_dims(inputs, axis=-1)

    outputs, output_paddings = Conv2dSubsampling(
        encoder_dim=config.encoder_dim,
        dtype=config.dtype,
        batch_norm_momentum=config.batch_norm_momentum,
        batch_norm_epsilon=config.batch_norm_epsilon,
        input_channels=1,
        output_channels=config.encoder_dim,
        enable_batchnorm=config.enable_subsampling_batchnorm,
        enable_synced_batchnorm=config.enable_synced_batchnorm)(outputs,
                                                                output_paddings,
                                                                train)

    outputs, output_paddings = Conv2dSubsampling(
        encoder_dim=config.encoder_dim,
        dtype=config.dtype,
        batch_norm_momentum=config.batch_norm_momentum,
        batch_norm_epsilon=config.batch_norm_epsilon,
        input_channels=config.encoder_dim,
        output_channels=config.encoder_dim,
        enable_batchnorm=config.enable_subsampling_batchnorm,
        enable_synced_batchnorm=config.enable_synced_batchnorm)(outputs,
                                                                output_paddings,
                                                                train)

    batch_size, subsampled_lengths, subsampled_dims, channels = outputs.shape

    outputs = jnp.reshape(
        outputs, (batch_size, subsampled_lengths, subsampled_dims * channels))

    outputs = nn.Dense(
        config.encoder_dim,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(outputs)

    outputs = nn.Dropout(
        rate=config.input_dropout_rate, deterministic=not train)(outputs)

    return outputs, output_paddings


class Conv2dSubsampling(nn.Module):
  """Helper module used in Subsample layer.

  1) Performs strided convolution over inputs and then applies non-linearity.
  2) Also performs strided convolution over input_paddings to return the correct
  paddings for downstream layers.
  """
  input_channels: int = 0
  output_channels: int = 0
  filter_stride: List[int] = (2, 2)
  padding: str = 'SAME'
  encoder_dim: int = 0
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  enable_batchnorm: bool = False
  enable_synced_batchnorm: bool = False

  def setup(self):
    self.filter_shape = (3, 3, self.input_channels, self.output_channels)
    self.kernel = self.param('kernel', nn.initializers.xavier_uniform(),
                             self.filter_shape)
    self.bias = self.param('bias', lambda rng, s: jnp.zeros(s, jnp.float32),
                           self.output_channels)

  @nn.compact
  def __call__(self, inputs, paddings, train):
    # Computing strided convolution to subsample inputs.
    feature_group_count = inputs.shape[3] // self.filter_shape[2]
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.kernel,
        window_strides=self.filter_stride,
        padding=self.padding,
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)

    outputs += jnp.reshape(self.bias, (1,) * (outputs.ndim - 1) + (-1,))

    if self.enable_batchnorm:
      outputs = BatchNorm(self.encoder_dim, self.dtype,
                          self.batch_norm_momentum, self.batch_norm_epsilon,
                          self.enable_synced_batchnorm)(
                              outputs, input_paddings=None, train=train)

    outputs = nn.relu(outputs)

    # Computing correct paddings post input convolution.
    input_length = paddings.shape[1]
    stride = self.filter_stride[0]

    pad_len = (input_length + stride - 1) // stride * stride - input_length
    out_padding = jax.lax.conv_general_dilated(
        lhs=paddings[:, :, None],
        rhs=jnp.ones([1, 1, 1]),
        window_strides=self.filter_stride[:1],
        padding=[(0, pad_len)],
        dimension_numbers=('NHC', 'HIO', 'NHC'))
    out_padding = jnp.squeeze(out_padding, axis=-1)

    # Mask outputs by correct paddings to ensure padded elements in inputs map
    # to padded value in outputs.
    outputs = outputs * (1.0 -
                         jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1))

    return outputs, out_padding


class FeedForwardModule(nn.Module):
  """Feedforward block of conformer layer."""
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, input_paddings=None, train=False):
    padding_mask = jnp.expand_dims(1 - input_paddings, -1)
    config = self.config

    if config.layernorm_everywhere:
      inputs = LayerNorm(config.encoder_dim)(inputs)
    else:
      inputs = BatchNorm(
          config.encoder_dim,
          config.dtype,
          config.batch_norm_momentum,
          config.batch_norm_epsilon,
          config.enable_synced_batchnorm)(inputs, input_paddings, train)

    inputs = nn.Dense(
        config.encoder_dim,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(inputs)

    if config.enable_ffn_relu:
      inputs = nn.relu(inputs)
    inputs *= padding_mask

    inputs = nn.Dropout(rate=config.feed_forward_dropout_rate)(
        inputs, deterministic=not train)

    return inputs


class LayerNorm(nn.Module):
  """Module implementing layer normalization.

  This implementation is same as in this paper:
  https://arxiv.org/pdf/1607.06450.pdf.

  note: we multiply normalized inputs by (1 + scale) and initialize scale to
  zeros, this differs from default flax implementation of multiplying by scale
  and initializing to ones.
  """
  dim: int = 0
  epsilon: float = 1e-6

  def setup(self):
    self.scale = self.param('scale', nn.initializers.zeros, [self.dim])
    self.bias = self.param('bias', nn.initializers.zeros, [self.dim])

  @nn.compact
  def __call__(self, inputs):
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)

    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
    normed_inputs *= (1 + self.scale)
    normed_inputs += self.bias

    return normed_inputs


class BatchNorm(nn.Module):
  """Implements batch norm respecting input paddings.

  This implementation takes into account input padding by masking inputs before
  computing mean and variance.

  This is inspired by lingvo jax implementation of BatchNorm:
  https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/jax/layers/normalizations.py#L92

  and the corresponding defaults for momentum and epsilon have been copied over
  from lingvo.
  """
  encoder_dim: int = 0
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  enable_synced_batchnorm: bool = False

  def setup(self):
    dim = self.encoder_dim
    dtype = self.dtype

    self.ra_mean = self.variable('batch_stats', 'mean',
                                 lambda s: jnp.zeros(s, dtype), dim)
    self.ra_var = self.variable('batch_stats', 'var',
                                lambda s: jnp.ones(s, dtype), dim)

    self.gamma = self.param('scale', nn.initializers.zeros, dim, dtype)
    self.beta = self.param('bias', nn.initializers.zeros, dim, dtype)

  def _get_default_paddings(self, inputs):
    """Gets the default paddings for an input."""
    in_shape = list(inputs.shape)
    in_shape[-1] = 1

    return jnp.zeros(in_shape, dtype=inputs.dtype)

  @nn.compact
  def __call__(self, inputs, input_paddings=None, train=False):
    rank = inputs.ndim
    reduce_over_dims = list(range(0, rank - 1))

    if input_paddings is None:
      padding = self._get_default_paddings(inputs)
    else:
      padding = jnp.expand_dims(input_paddings, -1)

    momentum = self.batch_norm_momentum
    epsilon = self.batch_norm_epsilon

    if train:
      mask = 1.0 - padding
      sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=True)
      count_v = jnp.sum(
          jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=True)

      if self.enable_synced_batchnorm:
        sum_v = jax.lax.psum(sum_v, axis_name='batch')
        count_v = jax.lax.psum(count_v, axis_name='batch')

      count_v = jnp.maximum(count_v, 1.0)
      mean = sum_v / count_v

      sum_vv = jnp.sum(
          (inputs - mean) * (inputs - mean) * mask,
          axis=reduce_over_dims,
          keepdims=True)

      if self.enable_synced_batchnorm:
        sum_vv = jax.lax.psum(sum_vv, axis_name='batch')

      var = sum_vv / count_v

      self.ra_mean.value = momentum * self.ra_mean.value + (1 - momentum) * mean
      self.ra_var.value = momentum * self.ra_var.value + (1 - momentum) * var
    else:
      mean = self.ra_mean.value
      var = self.ra_var.value

    inv = (1 + self.gamma) / jnp.sqrt(var + epsilon)

    bn_output = (inputs - mean) * inv + self.beta
    bn_output *= 1.0 - padding

    return bn_output



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
  cell_type: Type[nn.recurrent.RNNCellBase]
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()
  recurrent_dropout_rate: float = 0.0

  def setup(self):
    self.cell = self.cell_type(**self.cell_kwargs)

  @functools.partial(  # Repeatedly calls the below method to encode the inputs.
      nn.transforms.scan,
      variable_broadcast='params',
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
    hidden_size: The size of each recurrent cell.
    num_layers: The number of stacked recurrent layers. The output of the first
      layer, with optional dropout applied, feeds into the next layer.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  cell_type: Type[nn.recurrent.RNNCellBase]
  hidden_size: int
  num_layers: int = 1
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      lengths: Array,
      initial_states: Optional[Sequence[StateType]] = None,
      deterministic: bool = False
  ) -> Tuple[Array, Sequence[StateType]]:
    """Processes the input sequence using the recurrent cell.

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`).
        These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
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
    batch_size = inputs.shape[0]
    final_states = []
    num_directions = 2 if self.bidirectional else 1
    num_cells = self.num_layers * num_directions

    # Construct initial states.
    if initial_states is None:  # Initialize with zeros.
      rng = jax.random.PRNGKey(0)
      initial_states = [
          self.cell_type.initialize_carry(rng, (batch_size,), self.hidden_size)
          for _ in range(num_cells)
      ]
    if len(initial_states) != num_cells:
      raise ValueError(
          f'Please provide {self.num_cells} (`num_layers`, *2 if bidirectional)'
          f'initial states.')

    # For each layer, apply the forward and optionally the backward RNN cell.
    cell_idx = 0
    for _ in range(self.num_layers):
      # Unroll an RNN cell (forward direction) for this layer.
      outputs, final_state = GenericRNNSequenceEncoder(
          cell_type=self.cell_type,
          cell_kwargs=self.cell_kwargs,
          hidden_size=self.hidden_size,
          recurrent_dropout_rate=self.recurrent_dropout_rate,
          name=f'{self.name}SequenceEncoder_{cell_idx}')(
              inputs,
              lengths,
              initial_state=initial_states[cell_idx],
              deterministic=deterministic)
      final_states.append(final_state)
      cell_idx += 1

      # Unroll an RNN cell (backward direction) for this layer.
      if self.bidirectional:
        backward_outputs, backward_final_state = GenericRNNSequenceEncoder(
            cell_type=self.cell_type,
            cell_kwargs=self.cell_kwargs,
            hidden_size=self.hidden_size,
            recurrent_dropout_rate=self.recurrent_dropout_rate,
            name=f'{self.name}SequenceEncoder_{cell_idx}')(
                inputs,
                lengths,
                initial_state=initial_states[cell_idx],
                reverse=True,
                deterministic=deterministic)
        outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)
        final_states.append(backward_final_state)
        cell_idx += 1

      inputs = outputs

    return outputs, final_states


class LSTM(nn.Module):
  """LSTM.

  Attributes:
    hidden_size: The size of each recurrent cell.
    num_layers: The number of stacked recurrent layers. The output of the first
      layer, with optional dropout applied, feeds into the next layer.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    cell_type: The LSTM cell class to use. Default:
      `flax.linen.OptimizedLSTMCell`. If you use hidden_size of >2048, consider
      using `flax.linen.LSTMCell` instead, since the optimized LSTM cell works
      best for hidden sizes up to 2048.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  hidden_size: int
  num_layers: int = 1
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  cell_type: Any = nn.OptimizedLSTMCell
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
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`). These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
      deterministic: Disables dropout between layers when set to True.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states (h, c) for each cell and direction, ordered first by layer number
      and then by direction (first forward, then backward, if bidirectional).
    """
    return GenericRNN(
        cell_type=self.cell_type,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout_rate=self.dropout_rate,
        recurrent_dropout_rate=self.recurrent_dropout_rate,
        bidirectional=self.bidirectional,
        cell_kwargs=self.cell_kwargs,
        name='LSTM')(
            inputs,
            lengths,
            initial_states=initial_states,
            deterministic=deterministic)


class BatchRNN(nn.Module):
  """Implements a single conformer encoder layer.

  High level overview:
  """
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config

    if config.layernorm_everywhere:
      inputs = LayerNorm(config.encoder_dim)(inputs)
    else:
      inputs = BatchNorm(
          config.encoder_dim,
          config.dtype,
          config.batch_norm_momentum,
          config.batch_norm_epsilon,
          config.enable_synced_batchnorm)(inputs, input_paddings, train)

    lengths = jnp.sum(1 - input_paddings, axis=-1, dtype=jnp.int32)

    output, _ = LSTM(
        hidden_size=config.encoder_dim // 2, bidirectional=config.bidirectional,
        num_layers=1)(inputs, lengths)

    return output


class DeepSpeechEncoderDecoder(nn.Module):
  """Conformer (encoder + decoder) block.

  Takes audio input signals and outputs probability distribution over vocab size
  for each time step. The output is then fed into a CTC loss which eliminates
  the need for alignment with targets.
  """
  config: DeepspeechConfig

  def setup(self):
    config = self.config
    self.specaug = spectrum_augmenter.SpecAug(
        freq_mask_count=config.freq_mask_count,
        freq_mask_max_bins=config.freq_mask_max_bins,
        time_mask_count=config.time_mask_count,
        time_mask_max_frames=config.time_mask_max_frames,
        time_mask_max_ratio=config.time_mask_max_ratio,
        time_masks_per_frame=config.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=config.use_dynamic_time_mask_max_frames
    )

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config

    outputs = inputs
    output_paddings = input_paddings

    # Compute normalized log mel spectrograms from input audio signal.
    preprocessing_config = preprocessor.LibrispeechPreprocessingConfig()
    outputs, output_paddings = preprocessor.MelFilterbankFrontend(
        preprocessing_config,
        per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
        per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR)(outputs,
                                                            output_paddings)

    # Ablate random parts of input along temporal and frequency dimension
    # following the specaug procedure in https://arxiv.org/abs/1904.08779.
    if config.use_specaug and train:
      outputs, output_paddings = self.specaug(outputs, output_paddings)

    # Subsample input by a factor of 4 by performing strided convolutions.
    outputs, output_paddings = Subsample(
        config=config)(outputs, output_paddings, train)

    # Run the lstm layers.
    for _ in range(config.num_lstm_layers):
      if config.enable_residual_connections:
        outputs = outputs + BatchRNN(config)(outputs, output_paddings, train)
      else:
        outputs = BatchRNN(config)(outputs, output_paddings, train)

    for _ in range(config.num_ffn_layers):
      if config.enable_residual_connections:
        outputs = outputs + FeedForwardModule(config=self.config)(
            outputs, output_paddings, train)
      else:
        outputs = FeedForwardModule(config=self.config)(outputs,
                                                        output_paddings, train)

    # Run the decoder which in this case is a trivial projection layer.
    if config.enable_decoder_layer_norm:
      outputs = LayerNorm(config.encoder_dim)(outputs)

    outputs = nn.Dense(
        config.vocab_size,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(outputs)

    return outputs, output_paddings


class DeepSpeechModel(base_model.BaseModel):
  """Conformer model that takes in log mel spectrograms as inputs.

  outputs probability distribution over vocab size for each time step.
  """

  # Adapted form lingvo's greedy decoding logic here:
  # https://github.com/tensorflow/lingvo/blob/2ee26814c57b7dcead3f0382170f2f3da006f810/lingvo/jax/layers/ctc_objectives.py#L138
  def sequence_mask(self, lengths, maxlen):
    batch_size = lengths.shape[0]
    a = jnp.ones([batch_size, maxlen])
    b = jnp.cumsum(a, axis=-1)
    c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(lengths.dtype)
    return c

  def compute_loss(self, logits, logit_paddings, labels, label_paddings):
    logprobs = nn.log_softmax(logits)
    per_seq_loss = self.loss_fn(logprobs, logit_paddings, labels,
                                label_paddings)
    normalizer = jnp.sum(1 - label_paddings)

    normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)
    return normalized_loss

  def collapse_and_remove_blanks(self, labels, seq_length, blank_id: int = 0):
    b, t = labels.shape
    # Zap out blank.
    blank_mask = 1 - jnp.equal(labels, blank_id)
    labels = (labels * blank_mask).astype(labels.dtype)

    # Mask labels that don't equal previous label.
    label_mask = jnp.concatenate([
        jnp.ones_like(labels[:, :1], dtype=jnp.int32),
        jnp.not_equal(labels[:, 1:], labels[:, :-1])
    ],
                                 axis=1)

    # Filter labels that aren't in the original sequence.
    maxlen = labels.shape[1]
    seq_mask = self.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = label_mask * seq_mask

    # Remove repetitions from the labels.
    ulabels = label_mask * labels

    # Count masks for new sequence lengths.
    label_mask = jnp.not_equal(ulabels, 0).astype(labels.dtype)
    new_seq_len = jnp.sum(label_mask, axis=1)

    # Mask indexes based on sequence length mask.
    new_maxlen = maxlen
    idx_mask = self.sequence_mask(new_seq_len, maxlen=new_maxlen)

    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = jnp.reshape(ulabels, [-1])
    flat_idx_mask = jnp.reshape(idx_mask, [-1])

    indices = jnp.nonzero(flat_idx_mask, size=b * t)[0]
    values = jnp.nonzero(flat_labels, size=b * t)[0]
    updates = jnp.take_along_axis(flat_labels, values, axis=-1)

    # Scatter to flat shape.
    flat = jnp.zeros(flat_idx_mask.shape).astype(labels.dtype)
    flat = flat.at[indices].set(updates)
    # 0'th position in the flat array gets clobbered by later padded updates,
    # so reset it here to its original value
    flat = flat.at[0].set(updates[0])

    # Reshape back to square batch.
    batch_size = labels.shape[0]
    new_shape = [batch_size, new_maxlen]
    return (jnp.reshape(flat, new_shape).astype(labels.dtype),
            new_seq_len.astype(seq_length.dtype))

  def greedy_decode(self, logits, logit_paddings):
    per_frame_max = jnp.argmax(logits, axis=-1)
    seqlen = jnp.sum(1.0 - logit_paddings, axis=-1)
    hyp, _ = self.collapse_and_remove_blanks(per_frame_max, seqlen, blank_id=0)
    hyp_paddings = jnp.equal(hyp, 0).astype(jnp.int32)
    return hyp, hyp_paddings

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates cross_entopy on the given batch."""

    logits, logit_paddings = self.flax_module.apply(
        {
            'params': params,
            'batch_stats': batch_stats
        },
        batch['inputs'],
        batch['input_paddings'],
        train=False,
        mutable=False)

    labels = batch['targets']
    label_paddings = batch['target_paddings']

    normalized_loss = self.compute_loss(logits, logit_paddings, labels,
                                        label_paddings)
    hyps, hyp_paddings = self.greedy_decode(logits, logit_paddings)

    return self.metrics_bundle.gather_from_model_output(
        normalized_loss=normalized_loss,
        hyps=hyps,
        hyp_paddings=hyp_paddings,
        targets=labels,
        target_paddings=label_paddings,
        axis_name='batch')

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    """Return CTC loss."""

    # For more information on flax.linen.Module.apply, see the docs at
    # https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply.
    (outputs, output_paddings), new_batch_stats = self.flax_module.apply(
        {
            'params': params,
            'batch_stats': batch_stats
        },
        batch['inputs'],
        batch['input_paddings'],
        rngs={'dropout': dropout_rng},
        mutable=['batch_stats'],
        train=True)

    labels = batch['targets']
    label_paddings = batch['target_paddings']

    normalized_loss = self.compute_loss(outputs, output_paddings, labels,
                                        label_paddings)
    return normalized_loss, new_batch_stats

  def build_flax_module(self):
    config = DeepspeechConfig(
        vocab_size=self.hps.output_shape[1],
        encoder_dim=self.hps.encoder_dim,
        num_lstm_layers=self.hps.num_lstm_layers,
        num_ffn_layers=self.hps.num_ffn_layers,
        freq_mask_count=self.hps.freq_mask_count,
        freq_mask_max_bins=self.hps.freq_mask_max_bins,
        time_mask_count=self.hps.time_mask_count,
        time_mask_max_frames=self.hps.time_mask_max_frames,
        time_mask_max_ratio=self.hps.time_mask_max_ratio,
        time_masks_per_frame=self.hps.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=self.hps
        .use_dynamic_time_mask_max_frames,
        use_specaug=self.hps.use_specaug,
        input_dropout_rate=self.hps.input_dropout_rate,
        feed_forward_dropout_rate=self.hps.feed_forward_dropout_rate,
        enable_residual_connections=self.hps.enable_residual_connections,
        enable_decoder_layer_norm=self.hps.enable_decoder_layer_norm,
        bidirectional=self.hps.bidirectional,
        enable_subsampling_batchnorm=self.hps.enable_subsampling_batchnorm,
        enable_synced_batchnorm=self.hps.enable_synced_batchnorm,
        enable_ffn_relu=self.hps.enable_ffn_relu,
        layernorm_everywhere=self.hps.layernorm_everywhere)
    module = DeepSpeechEncoderDecoder(config)

    return module

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initializing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *x), dtype=hps.model_dtype)
        for x in hps.input_shape
    ]
    return dummy_inputs


class MLCommonsDeepSpeechModel(DeepSpeechModel):
  """Uses dropout_rate and aux_dropout_rate as hps.

  Dropouts are tied if tie_dropouts is True.
  Otherwise intended to be the same as DeepSpeechModel.
  """

  def build_flax_module(self):

    aux_dropout_rate = (
        self.hps.dropout_rate
        if self.hps.tie_dropouts
        else self.hps.aux_dropout_rate
    )

    config = DeepspeechConfig(
        vocab_size=self.hps.output_shape[1],
        encoder_dim=self.hps.encoder_dim,
        num_lstm_layers=self.hps.num_lstm_layers,
        num_ffn_layers=self.hps.num_ffn_layers,
        freq_mask_count=self.hps.freq_mask_count,
        freq_mask_max_bins=self.hps.freq_mask_max_bins,
        time_mask_count=self.hps.time_mask_count,
        time_mask_max_frames=self.hps.time_mask_max_frames,
        time_mask_max_ratio=self.hps.time_mask_max_ratio,
        time_masks_per_frame=self.hps.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=self.hps
        .use_dynamic_time_mask_max_frames,
        use_specaug=self.hps.use_specaug,
        input_dropout_rate=aux_dropout_rate,
        feed_forward_dropout_rate=self.hps.dropout_rate,
        enable_residual_connections=self.hps.enable_residual_connections,
        enable_decoder_layer_norm=self.hps.enable_decoder_layer_norm,
        bidirectional=self.hps.bidirectional,
        enable_subsampling_batchnorm=self.hps.enable_subsampling_batchnorm,
        enable_synced_batchnorm=self.hps.enable_synced_batchnorm,
        enable_ffn_relu=self.hps.enable_ffn_relu,
        layernorm_everywhere=self.hps.layernorm_everywhere)
    module = DeepSpeechEncoderDecoder(config)

    return module
