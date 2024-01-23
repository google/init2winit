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

"""Transformer-based language model with local attention.

Local and routing attention:
https://arxiv.org/abs/2003.05997

The support functions are refactored TensorFlow code previously
developed by: aurkor@google.com and msaffar@google.com.

"""
import itertools
import math
from typing import Any, Dict, List, Sequence, Tuple, Union
from flax import linen as nn
from flax.core.frozen_dict import unfreeze
from init2winit.model_lib import base_model
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np

INITIALIZERS = {
    'variance_scaling':
        nn.initializers.variance_scaling(
            0.2, mode='fan_avg', distribution='uniform'),
    'glorot_uniform':
        nn.initializers.glorot_uniform(),
    'uniform':
        nn.initializers.uniform()
}

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        kernel_init='variance_scaling',
        preprocessing_init='variance_scaling',
        embedding_init='variance_scaling',
        decode=False,
        decode_step=None,
        query_shape=(256,),
        preprocess_dropout_a=0.0,
        embedding_dims=1032,
        preprocess_dropout_b=0.0,
        hidden_size=1032,
        add_timing_signal=False,
        num_decoder_layers=24,
        padding_bias=None,
        decoder_dropout_a=0.1,
        total_key_depth=1032,
        total_value_depth=1032,
        bias_cache=None,
        local_num_heads=8,
        cache=None,
        memory_query_shape=(512,),
        memory_flange=(256,),
        cache_padding_bias=False,
        max_relative_position=513,
        attention_dropout=0.0,
        memory_antecedent=None,
        masked=True,
        local_relative=True,
        share_qk=True,
        token_bias=None,
        post_attention_epsilon=1e-6,
        post_attention_dropout=0.1,
        feedforward_dropout=0.0,
        feedforward_depths=[4096, 1032],
        model_dtype='float32',
        batch_size=8,
        grad_clip=None,
        lr_hparams={
            'base_lr': 0.01,
            'defer_steps': 10000,
            'schedule': 't2t_rsqrt_normalized_decay',
        },
        optimizer='adafactor',
        opt_hparams={
            'adafactor_decay_rate': 0.8,
            'clipping_threshold': 1.0,
            'factored': True,
            'min_dim_size_to_factor': 128,
            # The 2 hyperparameters cause errors with optax.inject_hyperparams
            # In this case it is not relevant since the default
            # adafactors values are needed
            # 'adafactor_momentum': 0.0,
            # 'multiply_by_parameter_scale': True,
        },
        # Below hyperparameters needed only to make the model
        # compatible with init2winit library
        rng_seed=-1,
        label_smoothing=None,
        weight_decay=None,
        l2_decay_factor=None,))

Tensor = Union[np.array, jnp.ndarray]


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return jax.lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


class FeedForward(nn.Module):
  """Feedforward block in a transformer block.

  Attributes:
    feedforward_depths: number of neurons in the 1st and 2nd Dense layers.
    feedforward_dropout: dropout rate in the Dropout layers.
    kernel_init: kernel initializer in the Dense layers.
  """
  feedforward_depths: Sequence[int] = None
  feedforward_dropout: float = 0.0
  kernel_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(self, input_x: Tensor, train: bool = False) -> Tensor:
    """Applies the FeedForward block.

    Args:
      input_x: an array of shape [batch, max_target_length, hidden_size] with
        the decoder output.
      train: boolean indicating whether training or not.

    Returns:
      output: an array of shape [batch, max_target_length, hidden_size]
      transfomed by the feedforward block
    """
    x = nn.Dense(
        features=self.feedforward_depths[0],
        kernel_init=self.kernel_init,
        use_bias=True,
        name='conv1')(input_x)

    x = nn.relu(x)

    x = nn.Dropout(
        rate=self.feedforward_dropout, deterministic=not train)(x)

    output = nn.Dense(
        features=self.feedforward_depths[1],
        kernel_init=self.kernel_init,
        use_bias=True,
        name='conv2')(x)
    return output


# TODO(krasowiak): Potentially replace lines 128-1226
# with modified Flax SelfAttention
def decode_step_to_index(decode_step: int,
                         array_shape: Tuple[int],
                         query_shape: Tuple[int] = (256,)) -> Tuple[int]:
  """Maps decode step to n-d index according to blocked raster scan order.

  Args:
    decode_step: an integer decode step.
    array_shape: a tuple with an array shape with no batch or depth dimensions.
    query_shape: a tuple with a query shape.

  Returns:
    output: a tuple representing the index of the element at
    `decode_step` w.r.t. blocked raster scan order.
  """
  if len(query_shape) != len(array_shape):
    raise ValueError(f'Query ({query_shape}) and array ({array_shape})'
                     ' shapes not the same length.')

  blocks_per_dimension = [t // q for t, q in zip(array_shape, query_shape)]
  items_in_block = np.prod(query_shape, dtype=jnp.int32)
  step_block = decode_step // items_in_block
  step_within_block = decode_step % items_in_block

  block_index = []
  for q in blocks_per_dimension[::-1]:
    block_index.insert(0, step_block % q)
    step_block //= q

  within_block_index = []
  for q in query_shape[::-1]:
    within_block_index.insert(0, step_within_block % q)
    step_within_block //= q

  output = tuple([
      w + b * q for w, b, q in zip(within_block_index, block_index, query_shape)
  ])
  return output


def get_item_at_decode_step(
    input_array: Tensor,
    decode_step: int = None,
    query_shape: Tuple[int] = (256,)
) -> Tensor:
  """Extracts a single item from an n-d array at `decode_step` position.

  Args:
    input_array: an array of shape [batch, d1, ..., dn, depth] with a single
      item to extract.
    decode_step: an integer decode step.
    query_shape: a tuple with a query shape.

  Returns:
    output: an array of shape [batch, 1, 1, ..., 1, depth] that is a single
    element from `x` at `decode_step` w.r.t. blocked raster scan order.
  """
  x_shape = input_array.shape
  index = decode_step_to_index(
      decode_step=decode_step,
      array_shape=x_shape[1:-1],
      query_shape=query_shape)
  index = [i.tolist() for i in index]
  output = input_array[:x_shape[0], index[0]:index[0] + len(index),
                       0:x_shape[-1]]
  return output


def embedding_to_padding(embedding: Tensor) -> Tensor:
  """Calculates the padding mask based on which embeddings are all zero.

  Args:
    embedding: an array of shape [..., depth] with embeddings.

  Returns:
    output: an array of shape [...] where each element is 1 if its corresponding
      embedding vector is all zero, and is 0 otherwise.
  """
  embbeding_sum = jnp.sum(jnp.absolute(embedding), axis=-1)
  output = jnp.array(jnp.equal(embbeding_sum, 0.0), dtype=jnp.float32)
  return output


def ones_matrix_band_part(
    num_rows: int,
    num_cols: int,
    max_backward: int,
    max_forward: int,
    output_shape: Tuple[int] = None) -> Tensor:
  """Prepares a matrix band part of 1s.

  Args:
    num_rows: number of rows in the output.
    num_cols: number of columns in the output.
    max_backward: maximum distance backward where negative values indicate
      unlimited.
    max_forward: maximum distance forward where negative values indicate
      unlimited.
    output_shape: shape to reshape output by.

  Returns:
    output: an array of size num_rows * num_cols reshaped to output_shape.
  """
  if max_backward < 0:
    max_backward = num_rows - 1
  if max_forward < 0:
    max_forward = num_cols - 1

  lower_mask = jnp.tri(num_cols, num_rows, max_backward).T
  upper_mask = jnp.tri(num_rows, num_cols, max_forward)
  output = jnp.ones(shape=(num_rows, num_cols)) * lower_mask * upper_mask

  if output_shape:
    output = jnp.reshape(output, output_shape)

  output = output.astype(dtype=jnp.float32)
  return output


def attention_bias_local(
    length: int,
    max_backward: int,
    max_forward: int,
    scale_factor: float = -1e9) -> Tensor:
  """Creates a bias array to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  Args:
    length: number of rows and columns in the matrix band part of ones.
    max_backward: maximum distance backward where negative values indicate
      unlimited.
    max_forward: maximum distance forward where negative values indicate
      unlimited.
    scale_factor: a scale factor for the output bias array.

  Returns:
    output: an bias array of shape [1, 1, length, length].
  """
  output = ones_matrix_band_part(
      num_rows=length,
      num_cols=length,
      max_backward=max_backward,
      max_forward=max_forward,
      output_shape=[1, 1, length, length])
  output = scale_factor * (1.0 - output)
  return output


def attention_bias_lower_triangle(length: int,
                                  bias_cache: Dict[str,
                                                   Tensor] = None) -> Tensor:
  """Creates an bias tensor to be added to attention logits.

  Args:
   length: number of rows and columns in the matrix band part of ones.
   bias_cache: attention bias cache.

  Returns:
    output: a bias array of shape [1, 1, length, length].
  """
  cache_key = 'attention_bias_lower_triangle_{}'.format(length)

  if bias_cache and cache_key in bias_cache:
    return bias_cache[cache_key]
  else:
    output = attention_bias_local(length=length, max_backward=-1, max_forward=0)
    bias_cache = {}
    bias_cache[cache_key] = output
  return output


def causal_attention_bias_nd(
    query_shape: Tuple[int] = (256,),
    memory_flange: Tuple[int] = (256,),
    decode_step: int = None,
    bias_cache: Dict[str, Tensor] = None
) -> Tensor:
  """Creates a causal attention bias for local nd attention.

  Args:
    query_shape: a tuple with a query shape.
    memory_flange: a tuple with a memory flange shape.
    decode_step: an integer decode step.
    bias_cache: attention bias cache.

  Returns:
    output: an array of [1, 1, query_items, memory_items] with a mask on and of
    shape [1, 1, 1, memory_items] if decode_step is not None.
  """
  cache_key = 'causal_attention_bias_{}_{}'.format(query_shape, memory_flange)

  if bias_cache and cache_key in bias_cache and decode_step is None:
    return bias_cache[cache_key]

  if all([m % q != 0 for q, m in zip(query_shape, memory_flange)]):
    raise ValueError(f'Query ({query_shape}) and memory ({memory_flange})'
                     ' modulo not equal to 0.')
  blocks_per_memory_flange = [
      m // q for q, m in zip(query_shape, memory_flange)
  ]
  prev_blocks = np.prod(
      [2 * b + 1 for b in blocks_per_memory_flange],
      dtype=jnp.int32) // 2
  all_blocks = np.prod(
      [blocks_per_memory_flange[0] + 1] +
      [2 * b + 1 for b in blocks_per_memory_flange[1:]],
      dtype=jnp.int32)
  future_blocks = all_blocks - prev_blocks - 1
  items_in_block = np.prod(query_shape, dtype=jnp.int32)
  items_in_query = items_in_block if decode_step is None else 1
  prev_blocks_attn = jnp.zeros(
      [1, 1, items_in_query, prev_blocks * items_in_block])

  if decode_step is None:
    center_block_attn = attention_bias_lower_triangle(length=items_in_block,
                                                      bias_cache=bias_cache)
  else:
    step_in_block = decode_step % items_in_block
    cond = jnp.less_equal(
        jnp.arange(stop=items_in_block, dtype=jnp.int32),
        step_in_block).reshape(shape=[1, 1, items_in_query, items_in_block])
    x = jnp.zeros([1, 1, items_in_query, items_in_block])
    y = -1e9 * jnp.ones([1, 1, items_in_query, items_in_block])
    center_block_attn = jnp.where(cond, x, y)
  future_blocks_attn = -1e9 * jnp.ones(
      [1, 1, items_in_query, future_blocks * items_in_block])
  output = jnp.concatenate(
      [prev_blocks_attn, center_block_attn, future_blocks_attn], axis=3)

  if decode_step is None:
    if bias_cache is None:
      bias_cache = {}
    bias_cache[cache_key] = output
  return output


def maybe_tile(input_x: Tensor, input_y: Tensor) -> Tensor:
  """Tiles two arrays so they have the same shape except for batch and depth.

  Args:
    input_x: first array to reshape.
    input_y: second array to reshape.

  Returns:
    output: two arrays with the same shape except for batch and depth.
  """
  x_shape = input_x.shape
  y_shape = input_y.shape
  if len(x_shape) != len(y_shape):
    raise ValueError(f'Query ({x_shape}) and array ({y_shape})'
                     ' shapes not the same length.')
  x_tile = [1]
  y_tile = [1]
  for x_dim, y_dim in zip(x_shape[1:-1], y_shape[1:-1]):
    try:
      if x_dim % y_dim != 0:
        raise ValueError(f'X_dim ({x_dim}) and y_dim ({y_dim})'
                         ' modulos not equal to 0.')
    except ValueError as maybe_tile_error:
      if y_dim % x_dim != 0:
        raise ValueError(f'X_dim ({x_dim}) and y_dim ({y_dim})'
                         ' modulos not equal to 0.') from maybe_tile_error
    if x_dim == y_dim:
      x_tile.append(1)
      y_tile.append(1)
    elif x_dim > y_dim:
      x_tile.append(1)
      y_tile.append(x_dim // y_dim)
    else:
      x_tile.append(y_dim // x_dim)
      y_tile.append(1)

  x_tiled = jnp.tile(input_x, x_tile + [1])
  y_tiled = jnp.tile(input_y, y_tile + [1])
  return x_tiled, y_tiled


def local_attention_bias_nd(
    v_array: Tensor,
    query_shape: Tuple[int] = (256,),
    memory_flange: Tuple[int] = (256,),
    masked: bool = True,
    cache_padding_bias: bool = False,
    decode_step: int = None,
    bias_cache: Dict[str, Tensor] = None
) -> Tensor:
  """Creates an attention bias for local n-d attention.

  Args:
    v_array: array fo shape [batch, num_blocks, items_in_blocks, depth] for v.
    query_shape: a tuple with a query shape.
    memory_flange: a tuple with a memory flange shape.
    masked: indiactor to mask or not mask bias.
    cache_padding_bias: whether to cache padding bias as well to save memory.
    decode_step: an integer decode step.
    bias_cache: attention bias cache.

  Returns:
    output: the local attention bias array of shape [batch * heads, num_blocks,
    items_in_query, items_in_memory] or of shape [1, num_blocks, items_in_query,
    items_in_memory] if cache_padding_bias is True.
  """
  cache_names = ['_'.join(map(str, i)) for i in [query_shape, memory_flange]]
  cache_key = 'local_attention_bias_{}_{}_{}_{}'.format(cache_names[0],
                                                        cache_names[1], masked,
                                                        cache_padding_bias)
  if bias_cache and cache_key in bias_cache and decode_step is None:
    return bias_cache[cache_key]

  if cache_padding_bias:
    array = embedding_to_padding(embedding=v_array[:1, :, :, :]) * -1e9,
    padding_attn_bias = jnp.expand_dims(array, axis=-2)
  else:
    array = embedding_to_padding(embedding=v_array) * -1e9
    padding_attn_bias = jnp.expand_dims(array, axis=-2)

  if masked:
    causal_attn_bias = causal_attention_bias_nd(
        query_shape=query_shape,
        memory_flange=memory_flange,
        decode_step=decode_step,
        bias_cache=bias_cache)
    causal_attn_bias, padding_attn_bias = maybe_tile(input_x=causal_attn_bias,
                                                     input_y=padding_attn_bias)
    output = jnp.minimum(causal_attn_bias, padding_attn_bias)
  else:
    output = padding_attn_bias

  if cache_padding_bias and decode_step is None:
    if bias_cache is None:
      bias_cache = {}
    bias_cache[cache_key] = output
  return output


def pad_to_multiple_nd(input_x: Tensor, block_shape: Tuple[int]) -> Tensor:
  """Ensures the input is a multiple of a provided shape.

  Args:
    input_x: an input array of shape [batch, d1, ..., dn, depth].
    block_shape: a block shape.

  Returns:
    output: padded array where each dimension is a multiple of corresponding
    block length.
  """
  shape = input_x.shape
  paddings = [-l % b for l, b in zip(shape[1:-1], block_shape)]
  output = jnp.pad(
      input_x, [(0, 0)] + [(0, p) for p in paddings] + [(0, 0)])
  return output


def select_block_for_decode_step(
    input_x: Tensor,
    decode_step: int = None,
    query_shape: Tuple[int] = (256,)
) -> Tensor:
  """Selects one block from the input array that contains position `decode_step`.

  Args:
    input_x: an array of shape [batch, blocks_per_d1, ..., blocks_per_dn, b1 *
      ...* bn, depth] with a block to extract.
    decode_step: an integer decode step.
    query_shape: a tuple with a query shape.

  Returns:
     output: an array of shape [batch, [1] * n, b1 * ... * bn, depth] with the
     extracted block.
  """
  blocked_x_shape = input_x.shape
  x_shape = [b * q for b, q in zip(blocked_x_shape[1:-2], query_shape)]
  index = decode_step_to_index(
      decode_step=decode_step, array_shape=query_shape, query_shape=x_shape)
  blocked_index = [i // q for i, q in zip(index, query_shape)]
  output = input_x[:blocked_x_shape[0],
                   blocked_index[0]:blocked_index[0] + len(blocked_index),
                   0:blocked_x_shape[-2], 0:blocked_x_shape[-1]]
  return output


def break_into_blocks_nd(input_x: Tensor, block_shape: Tuple[int]) -> Tensor:
  """Breaks the input array into blocks of `block_shape`.

  Args:
    input_x: an array of shape [batch, d1, d2, ..., dn, depth] to be broken down
      into blocks.
    block_shape: a block shape.

  Returns:
    output: an array of shape [batch, d1//block1, ..., dn//blockn, block1 *... *
    blockn, depth] broken down into blocks.
  """
  x_shape = list(input_x.shape)
  if all([l % b != 0 for l, b in zip(x_shape[1:], block_shape)]):
    raise ValueError(f'X_shape[1:] ({x_shape[1:]}) and block ({block_shape})'
                     ' modulo not equal to 0.')
  blocks_per_dimension = [l // b for l, b in zip(x_shape[1:], block_shape)]
  reshape_to = list(
      itertools.chain.from_iterable(zip(blocks_per_dimension, block_shape)))
  input_x = jnp.reshape(input_x, [-1] + reshape_to + x_shape[-1:])
  block_dimensions_index = [2 * (i + 1) for i in range(len(block_shape))]
  axes = [0] + [i - 1 for i in block_dimensions_index
               ] + block_dimensions_index + [2 * len(block_shape) + 1]
  input_x = jnp.transpose(input_x, axes)
  axes = [-1] + blocks_per_dimension + [
      np.prod(block_shape, dtype=jnp.int32)
  ] + x_shape[-1:]
  output = jnp.reshape(input_x, axes)
  return output


def break_into_memory_blocks_nd(
    input_x: Tensor,
    query_shape: Tuple[int] = (256,),
    memory_flange: Tuple[int] = (256,),
    masked: bool = True) -> Tensor:
  """Breaks an input array into memory blocks around query blocks.

  Args:
    input_x: an array of shape [batch, d1, d2, ..., dn, depth].
    query_shape: a tuple with a query shape.
    memory_flange: a tuple with a memory flange shape.
    masked: indiactor to mask or not mask bias.

  Returns:
    output: an array split of shape [batch, blocks_per_d1, ..., blocks_per_dn,
    b1 * ...* bn, depth] where b[i] is the memory block size in dimension i
    which is equal to q[i] + 2m[i] or q[i] + m[i] if masked attention and i = 1
  """
  if all([m % q != 0 for q, m in zip(query_shape, memory_flange)]):
    raise ValueError(f'Query ({query_shape}) and memory ({memory_flange})'
                     ' modulo not equal to 0.')
  original_x_shape = input_x.shape
  blocks_in_memory_flange = [m // b for b, m in zip(query_shape, memory_flange)]
  num_query_blocks = [
      l // q for l, q in zip(original_x_shape[1:-1], query_shape)
  ]

  if masked:
    input_x = jnp.pad(
        input_x,
        [[0, 0], [memory_flange[0], 0]] +
        [[p, p] for p in memory_flange[1:]] + [[0, 0]])
  else:
    input_x = jnp.pad(
        input_x,
        [[0, 0]] + [[p, p] for p in memory_flange] + [[0, 0]])

  query_blocks = break_into_blocks_nd(input_x=input_x, block_shape=query_shape)
  start_indices_per_dimension = []

  for dimension, blocks in enumerate(blocks_in_memory_flange):
    if masked and dimension == 0:
      size = blocks + 1
    else:
      size = 2 * blocks + 1
    start_indices_per_dimension.append(range(size))

  slices = []
  for start_indices in itertools.product(*start_indices_per_dimension):
    s = query_blocks[0:,
                     start_indices[0]:start_indices[0] + num_query_blocks[0],
                     0:, 0:]
    slices.append(s)
  output = jnp.concatenate(slices, axis=-2)
  return output


def flatten_blocks_nd(input_x: Tensor) -> Tensor:
  """Flattens blocks of the input array.

  Args:
    input_x: an array of shape [batch, b1, ..., bn, items_in_block, depth] to
      flatten.

  Returns:
    output: a flattened array of shape [batch, b1 * ...* bm, items_in_block,
    depth].
  """
  x_shape = list(input_x.shape)
  num_blocks = np.prod(x_shape[1:-2], dtype=jnp.int32)
  output = jnp.reshape(input_x, [-1, num_blocks] + x_shape[-2:])
  return output


def unflatten_blocks_nd(input_x: Tensor,
                        blocks_per_dimension: List[int]) -> Tensor:
  """Converts a flattened array to a blocked array.

  Args:
    input_x: an array of shape [batch, d1 * ... dn, items_in_block, depth].
    blocks_per_dimension: number of blocks in each dimension.

  Returns:
    output: an array of shape [batch, d1, d2, ..., dn, items_in_block, depth].
  """
  x_shape = list(input_x.shape)
  assert x_shape[1] == np.prod(
      blocks_per_dimension, dtype=jnp.int32)
  output = jnp.reshape(
      input_x, [-1] + blocks_per_dimension + x_shape[-2:])
  return output


def break_bias_into_blocks(
    input_bias: Tensor,
    local_num_heads: int = 8,
    memory_query_shape: Tuple[int] = (256,),
    memory_flange: Tuple[int] = (256,),
    masked: bool = True,
    decode_step: int = None) -> Tensor:
  """Breaks bias into a blocked array.

  Args:
    input_bias: a bias array of shape of shape [batch * heads, num_blocks,
    items_in_query, items_in_memory].
    local_num_heads: a number of local attention heads.
    memory_query_shape: a tuple with a memory query shape.
    memory_flange: a tuple with a memory flange shape.
    masked: indiactor to mask or not mask bias.
    decode_step: an integer decode step.

  Returns:
    output: a bias array broken into blocks.
  """
  x = jnp.expand_dims(input_bias, axis=1)
  x = jnp.tile(x, [1, local_num_heads, 1])
  x = jnp.expand_dims(x, axis=-1)
  x = jnp.reshape(x, [-1] + list(x.shape[2:]))
  x = pad_to_multiple_nd(input_x=x, block_shape=memory_query_shape)
  x = break_into_memory_blocks_nd(
      input_x=x,
      query_shape=memory_query_shape,
      memory_flange=memory_flange,
      masked=masked)

  if decode_step is not None:
    x = select_block_for_decode_step(
        input_x=x, decode_step=decode_step, query_shape=memory_query_shape)

  x = flatten_blocks_nd(input_x=x)
  output = jnp.squeeze(x, axis=-1)
  return output


def cast_like(
    input_x: Tensor,
    input_y: Tensor) -> Tensor:
  """Cast the same dtype on the first array as on the second if necessary.

  Args:
    input_x: first array to compare.
    input_y: second array to compare.

  Returns:
    output: first array with the same dtype as the second.
  """

  if input_x.dtype == input_y.dtype:
    return input_x

  output = input_x.astype(input_y.dtype)
  return output


def generate_relative_positions_matrix(
    length_q: int,
    length_k: int,
    max_relative_position: int = 513,
    query_shape: Tuple[int] = (256,),
    decode_step: int = None) -> Tensor:
  """Generates matrix of relative positions.

  Args:
    length_q: length of the query array.
    length_k: length of the key array.
    max_relative_position: how much distance to consider for relative positions.
    query_shape: a tuple with a query shape.
    decode_step: an integer decode step.

  Returns:
    output: a matrix of relative positions of shape [length_q, length_k].
  """
  if decode_step is None:

    if length_q == length_k:
      range_vec_q = range_vec_k = jnp.arange(length_q)
    else:
      range_vec_k = jnp.arange(length_k)
      range_vec_q = range_vec_k[-length_q:]
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]

  else:
    block_len = np.prod(query_shape)
    positive_positions = block_len - decode_step % block_len
    distance_mat = jnp.expand_dims(
        jnp.arange(-length_k, 0, 1),
        axis=0) + positive_positions
  distance_mat_clipped = jnp.clip(
      distance_mat, -max_relative_position, max_relative_position)
  output = distance_mat_clipped + max_relative_position
  return output


class RelativePositionEmbeddings(nn.Module):
  """Generates an array of relative positions of specified shape.

  Attributes:
    max_relative_position: how much distance to consider for relative positions.
    decode_step: the decode step in fast decoding mode.
    query_shape: a tuple with a query shape.
    embedding_init: embeddings initializer.
  """
  embed_layer_name: str
  max_relative_position: int = 513
  depth: int = 129
  query_shape: Tuple[int] = (256,)
  decode_step: int = None
  embedding_init: Any = nn.initializers.glorot_uniform()

  def setup(self):
    self.embedding = self.param(
        self.embed_layer_name, self.embedding_init,
        (self.max_relative_position * 2 + 1, self.depth))

  def __call__(self, length_q: int, length_k: int) -> Tensor:
    """Applies the RelativePositionEmbeddings module.

    Args:
      length_q: length of the query array.
      length_k: length of the key array.

    Returns:
      output: an array of shape [1 if decode else length_q, length_k, depth].
    """
    relative_positions_matrix = generate_relative_positions_matrix(
        length_q=length_q,
        length_k=length_k,
        max_relative_position=self.max_relative_position,
        query_shape=self.query_shape,
        decode_step=self.decode_step)
    output = jnp.take(
        self.embedding, indices=relative_positions_matrix, axis=0)
    return output


def relative_attention_inner(input_x: Tensor,
                             input_y: Tensor,
                             input_z: Tensor,
                             transpose: bool) -> Tensor:
  """Calculates position-aware inner dot-product attention.

  Args:
    input_x: input array of shape [batch_size, heads, length or 1, length or
      depth].
    input_y: second input array of shape [batch_size, heads, length or 1,
      depth].
    input_z: third input array of shape [length or 1, length, depth].
    transpose: whther to transpose inner matrices of input_y and input_z.

  Returns:
    output: an array of shape [batch_size, heads, length, length or depth].
  """
  if transpose:
    xy_matmul = jnp.einsum(
        'bhxd,bhyd->bhxy',
        input_x,
        input_y,
        precision=jax.lax.Precision.HIGHEST)
    x_tz_matmul_r_t = jnp.einsum(
        'bhxd,xyd->bhxy', input_x, input_z, precision=jax.lax.Precision.HIGHEST)
  else:
    xy_matmul = jnp.einsum(
        'bhxd,bhdy->bhxy',
        input_x,
        input_y,
        precision=jax.lax.Precision.HIGHEST)
    x_tz_matmul_r_t = jnp.einsum(
        'bhxd,xdy->bhxy', input_x, input_z, precision=jax.lax.Precision.HIGHEST)
  output = xy_matmul + x_tz_matmul_r_t
  return output


class RelativeDotProductAttention(nn.Module):
  """Calculates relative position-aware dot-product self-attention.

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Attributes:
    max_relative_position: how much distance to consider for relative positions.
    attention_dropout: an attention dropout rate.
    decode_step: the decode step in fast decoding mode.
    query_shape: a tuple with a query shape.
    embedding_init: embeddings initializer.
  """
  max_relative_position: int = 513
  query_shape: Tuple[int] = (256,)
  attention_dropout: float = 0.0
  decode_step: int = None
  embedding_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(
      self, q_array: Tensor,
      k_array: Tensor,
      v_array: Tensor,
      bias_array: Tensor,
      train: bool = False) -> Tensor:
    """Applies the RelativeDotProductAttention module.

    Args:
      q_array: a query array of shape [batch, heads, length, depth].
      k_array: a key array of shape [batch, heads, length, depth].
      v_array: a value array of shape [batch, heads, length, depth].
      bias_array: a bias array of shape of shape [batch * heads, num_blocks,
        items_in_query, items_in_memory].
      train: whether a deterministic mode or not.

    Returns:
      output: an array with dot-product attention, an array with attention
      weights
    """
    if not self.max_relative_position:
      raise ValueError('Max relative position (%s) should be > 0 when using '
                       'relative self attention.' %
                       (self.max_relative_position))
    depth = k_array.shape[3]
    length_k = k_array.shape[2]
    length_q = q_array.shape[2]

    relations_keys = RelativePositionEmbeddings(
        embed_layer_name='relative_positions_keys',
        max_relative_position=self.max_relative_position,
        depth=depth,
        query_shape=self.query_shape,
        decode_step=self.decode_step,
        embedding_init=self.embedding_init)(
            length_q=length_q, length_k=length_k)
    relations_values = RelativePositionEmbeddings(
        embed_layer_name='relative_positions_values',
        max_relative_position=self.max_relative_position,
        depth=depth,
        query_shape=self.query_shape,
        decode_step=self.decode_step,
        embedding_init=self.embedding_init)(
            length_q=length_q, length_k=length_k)
    logits = relative_attention_inner(
        input_x=q_array,
        input_y=k_array,
        input_z=relations_keys,
        transpose=True)
    if bias_array is not None:
      logits += bias_array
    weights = nn.softmax(logits)
    if self.attention_dropout:
      weights = nn.Dropout(
          rate=self.attention_dropout, deterministic=not train)(weights)
    output = relative_attention_inner(
        input_x=weights,
        input_y=v_array,
        input_z=relations_values,
        transpose=False)
    return output, weights


class DotProductAttention(nn.Module):
  """Calculates a dot product attention.

  Attributes:
    local_relative: whether to use a local relative attention.
    max_relative_position: how much distance to consider for relative positions.
    attention_dropout: an attention dropout rate.
    decode_step: the decode step in fast decoding mode.
    query_shape: a tuple with a query shape.
    embedding_init: embeddings initializer.
  """
  local_relative: bool = True
  max_relative_position: int = 513
  attention_dropout: float = 0.0
  decode_step: int = None
  query_shape: Tuple[int] = (256,)
  embedding_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(
      self, q_array: Tensor,
      k_array: Tensor,
      v_array: Tensor,
      bias_array: Tensor, train: bool = False) -> Tensor:
    """Applies the DotProductAttention module.

    Args:
      q_array: a query array of shape [..., length_q, depth_k].
      k_array: a key array of shape [..., length_kv, depth_k].
      v_array: a value array of shape [..., length_kv, depth_v].
      bias_array: a bias array of shape of shape [batch * heads, num_blocks,
        items_in_query, items_in_memory].
      train: whether a deterministic mode or not.

    Returns:
      output: an array of shape [..., length_q, depth_v], an array of shape
      [..., length_q, length_kv] with attention weights.
    """
    if self.local_relative:
      return RelativeDotProductAttention(
          max_relative_position=self.max_relative_position,
          attention_dropout=self.attention_dropout,
          decode_step=self.decode_step,
          query_shape=self.query_shape,
          embedding_init=self.embedding_init,
      )(q_array=q_array,
        k_array=k_array,
        v_array=v_array,
        bias_array=bias_array,
        train=train)
    logits = jnp.matmul(q_array, jnp.transpose(k_array, axes=(0, 1, 3, 2)))
    if bias_array is not None:
      bias_array = cast_like(input_x=bias_array, input_y=logits)
      logits += bias_array
    weights = nn.softmax(logits)
    if self.attention_dropout:
      weights = nn.Dropout(
          rate=self.attention_dropout, deterministic=not train)(weights)
    output = jnp.matmul(weights, v_array)
    return output, weights


def combine_heads_nd(input_x: Tensor) -> Tensor:
  """Inverses the split_heads_nd function.

  Args:
    input_x: an array of shape [batch, num_heads, d1, ..., dn, depth //
      num_heads].

  Returns:
    output: an array of shape [batch, d1, ...., dn, depth].
  """
  num_dimensions = len(input_x.shape) - 3
  axes = [0] + list(range(2, num_dimensions + 2)) + [1, num_dimensions + 2]
  input_x = jnp.transpose(input_x, axes=axes)
  x_shape = list(input_x.shape)
  a, b = x_shape[-2:]
  output = jnp.reshape(input_x, x_shape[:-2] + [a * b])
  return output


def put_back_blocks_nd(input_x: Tensor, block_shape: Tuple[int]) -> Tensor:
  """Restructures an input array from blocks to normal ordering.

  Args:
    input_x: an input array of shape [batch, b1, ..., bn, items_in_block,
      depth].
    block_shape: a block shape.

  Returns:
    output: an output array of shape [batch, d1, ..., dn, depth].
  """
  x_shape = list(input_x.shape)

  if isinstance(x_shape[-2], int):
    if x_shape[-2] != np.prod(block_shape):
      raise ValueError(f'X_shape[-2] ({x_shape[-2]}) and block ({block_shape})'
                       ' are not equal.')

  x = jnp.reshape(
      input_x, x_shape[:-2] + list(block_shape) + x_shape[-1:])
  block_dimension_index = list(range(1, len(block_shape) + 1))
  block_shape_index = [b + len(block_shape) for b in block_dimension_index]
  interleaved_dimensions = list(
      itertools.chain.from_iterable(
          zip(block_dimension_index, block_shape_index)))
  x = jnp.transpose(x,
                    [0] + interleaved_dimensions + [2 * len(block_shape) + 1])
  axes = [-1] + [
      x_shape[2 * i + 1] * x_shape[2 * i + 2] for i in range(len(block_shape))
  ] + x_shape[-1:]
  output = jnp.reshape(x, axes)
  return output


class LocalAttention(nn.Module):
  """Calculates a local attention.

  Attributes:
    query_shape: a tuple with a query shape.
    memory_query_shape: a tuple with a memory query shape.
    memory_flange: a tuple with a memory flange shape.
    local_num_heads: a number of local attention heads.
    local_relative: whether to use a local relative attention.
    memory_antecedent: whether there is memory antecedent.
    masked: indiactor to mask or not mask bias.
    decode_step: the decode step in fast decoding mode.
    max_relative_position: how much distance to consider for relative positions.
    cache_padding_bias: whether to cache padding bias as well to save memory.
    attention_dropout: an attention dropout rate.
    bias_cache: attention bias cache.
    share_qk: whether to share the query and key.
    token_bias: a token bias array.
    padding_bias: a padding bias array.
    kernel_init: kernel initializer in the Dense layers.
    embedding_init: embeddings initializer.
  """
  query_shape: Tuple[int] = (256,)
  memory_query_shape: Tuple[int] = (512,)
  memory_flange: Tuple[int] = (256,)
  local_num_heads: int = 8
  local_relative: bool = True
  memory_antecedent: bool = None
  masked: bool = True
  decode_step: int = None
  max_relative_position: int = 513
  cache_padding_bias: bool = False
  attention_dropout: float = 0.0
  bias_cache: Dict[str, Tensor] = None
  share_qk: bool = True
  token_bias: Tensor = None
  padding_bias: Tensor = None
  kernel_init: Any = nn.initializers.glorot_uniform()
  embedding_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(self,
               q_array: Tensor,
               k_array: Tensor,
               v_array: Tensor,
               train: bool = False) -> Tensor:
    """Applies the LocalAttention module.

    Args:
      q_array: a query array of shape [batch, heads, d1, d2, ..., dn, depth_k].
      k_array: a key array of shape [batch, heads, d1, d2, ..., dn, depth_k].
      v_array: a value array of shape [batch, heads, d1, d2, ..., dn, depth_v].
      train: whether a deterministic mode or not.

    Returns:
      output: an array of shape [batch, head, d1, d2, ..., dn, depth_v], an
      array of shape [batch, head, 1, 1, ..., 1, depth_v] with attention
      weights.
    """
    if all([m % b != 0 for m, b in zip(self.memory_flange, self.query_shape)]):
      raise ValueError(
          f'Query ({self.query_shape}) and memory ({self.memory_flange})'
          ' modulo not equal to 0.')

    if self.decode_step is not None:
      q_array = jnp.reshape(
          q_array, [-1] + list(q_array.shape[2:]))
      latest_q = get_item_at_decode_step(
          input_array=q_array,
          decode_step=self.decode_step,
          query_shape=self.query_shape)
      q_array = jnp.reshape(
          q_array,
          [-1, self.local_num_heads] +
          list(q_array.shape[1:]))
      latest_q = jnp.reshape(
          latest_q,
          [-1, self.local_num_heads] +
          list(latest_q.shape[1:]))
      q_shape = list(latest_q.shape)
    else:
      q_shape = list(q_array.shape)
    k_shape = list(k_array.shape)
    v_shape = list(v_array.shape)

    outputs = []
    if self.decode_step is not None:
      q_array = latest_q

    q_array = jnp.reshape(q_array, [-1] + q_shape[2:])
    k_array = jnp.reshape(k_array, [-1] + k_shape[2:])
    v_array = jnp.reshape(v_array, [-1] + v_shape[2:])

    mem_query_shape = self.query_shape

    if self.memory_antecedent is not None:
      mem_query_shape = self.memory_query_shape

    if self.decode_step is None:
      q_array = pad_to_multiple_nd(
          input_x=q_array, block_shape=self.query_shape)
    k_array = pad_to_multiple_nd(input_x=k_array, block_shape=mem_query_shape)
    v_array = pad_to_multiple_nd(input_x=v_array, block_shape=mem_query_shape)

    if self.decode_step is None:
      q_array = break_into_blocks_nd(
          input_x=q_array, block_shape=self.query_shape)
    else:
      q_array = jnp.reshape(
          q_array, [-1] + [1] * (len(q_shape) - 3) + [q_shape[-1]])

    k_array = break_into_memory_blocks_nd(
        input_x=k_array,
        query_shape=mem_query_shape,
        memory_flange=self.memory_flange,
        masked=self.masked)
    v_array = break_into_memory_blocks_nd(
        input_x=v_array,
        query_shape=mem_query_shape,
        memory_flange=self.memory_flange,
        masked=self.masked)
    blocks_per_dim = list(q_array.shape[1:-2])

    if self.decode_step is not None:
      k_array = select_block_for_decode_step(
          input_x=k_array,
          decode_step=self.decode_step,
          query_shape=mem_query_shape)
      v_array = select_block_for_decode_step(
          input_x=v_array,
          decode_step=self.decode_step,
          query_shape=mem_query_shape)

    q_array = flatten_blocks_nd(input_x=q_array)
    k_array = flatten_blocks_nd(input_x=k_array)
    v_array = flatten_blocks_nd(input_x=v_array)

    if self.bias_cache is None:
      bias_cache = {}
    else:
      bias_cache = unfreeze(self.bias_cache)

    attn_bias = local_attention_bias_nd(
        query_shape=mem_query_shape,
        memory_flange=self.memory_flange,
        v_array=v_array,
        masked=self.masked,
        cache_padding_bias=self.cache_padding_bias,
        decode_step=self.decode_step,
        bias_cache=bias_cache)

    if self.padding_bias is not None:
      padding_bias = break_bias_into_blocks(input_bias=self.padding_bias)
      padding_bias = jnp.expand_dims(padding_bias * -1e9, axis=-2)
      attn_bias = jnp.minimum(attn_bias, padding_bias)

    if self.token_bias is not None:
      token_bias = break_bias_into_blocks(input_bias=self.token_bias)
      token_bias = jnp.expand_dims(token_bias, axis=-2)
      token_bias_weight = jnp.array(1.0)
      attn_bias += token_bias_weight * token_bias

    output, _ = DotProductAttention(
        local_relative=self.local_relative,
        max_relative_position=self.max_relative_position,
        attention_dropout=self.attention_dropout,
        decode_step=self.decode_step,
        query_shape=self.query_shape,
        embedding_init=self.embedding_init)(
            q_array=q_array,
            k_array=k_array,
            v_array=v_array,
            bias_array=attn_bias,
            train=train)
    output = unflatten_blocks_nd(
        input_x=output, blocks_per_dimension=blocks_per_dim)
    output = jnp.reshape(
        output,
        [q_shape[0], self.local_num_heads] +
        list(output.shape[1:]))
    outputs.append(output)

    output = jnp.concatenate(outputs, axis=1)
    output_shape = list(output.shape)
    output = jnp.reshape(
        output, [output_shape[0], self.local_num_heads, -1, output_shape[-1]])
    output = nn.Dense(
        output_shape[-1],
        kernel_init=self.kernel_init,
        use_bias=False,
        name='dense')(
            output)
    output = jnp.reshape(output, output_shape)

    if self.decode_step is None:
      output = jnp.reshape(
          output, [-1] + list(output.shape[2:]))
      output = put_back_blocks_nd(
          input_x=output, block_shape=self.query_shape)
      output = jnp.reshape(
          output, q_shape[:2] + list(output.shape[1:]))
      output = output[0:, 0:, 0:q_shape[2:-1][0], 0:]
    return output


# TODO(krasowiak): Potentially replace lines 1231-1501
# with Flax MultiHeadDotProductAttention
def split_heads_nd(input_x: Tensor, num_heads: int = 8) -> Tensor:
  """Splits the depth dimension into multiple heads.

  Args:
    input_x: an input array of shape [batch, d1, ..., dn, depth].
    num_heads: a number of attention heads.

  Returns:
    output: an array of shape [batch, num_heads, d1, ..., dn, depth //
    num_heads].
  """
  num_dimensions = len(input_x.shape) - 2
  x_shape = list(input_x.shape)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(num_heads, int):
    if m % num_heads != 0:
      raise ValueError(f'X_shape[-1] ({m}) and memory ({num_heads})'
                       ' modulo not equal to 0.')
  x = jnp.reshape(input_x, x_shape[:-1] + [num_heads, m // num_heads])
  axes = [0, num_dimensions + 1] + list(range(
      1, num_dimensions + 1)) + [num_dimensions + 2]
  output = jnp.transpose(x, axes=axes)
  return output


def put_item_in_decode_step(
    input_x: Tensor,
    decode_step: int = None,
    query_shape: Tuple[int] = (256,),
    replacement: Any = 1.0) -> Tensor:
  """Puts a single item into an an array at the `decode_step` position.

  Args:
    input_x: an array of shape [batch, heads, d1, d2, ..., dn, depth].
    decode_step: the decode step in fast decoding mode.
    query_shape: a tuple with a query shape.
    replacement: new values that will be used as a replacement, if array then of
      shape [batch, heads, 1, 1, ..., 1, depth]

  Returns:
    output: an array of shape [batch, heads, d1, d2, ..., dn, depth] with value
    at `decode_step`
    w.r.t. blocked raster scan order is updated to be `replacement`.
  """
  x_shape = list(input_x.shape)
  index = decode_step_to_index(
      decode_step=decode_step,
      array_shape=query_shape,
      query_shape=x_shape[2:-1])
  flattened_x = jnp.reshape(
      input_x,
      [
          -1, x_shape[1],
          np.prod(x_shape[2:-1]), x_shape[-1]
      ])
  flattened_x = jnp.transpose(flattened_x, axes=[2, 0, 1, 3])
  flattened_index = 0

  factor = 1
  for d, idx in zip(x_shape[-2:1:-1], index[::-1]):
    flattened_index += idx * factor
    factor *= d

  updated_x = input_x.at[flattened_index].set(replacement)
  updated_x = jnp.transpose(updated_x, axes=(1, 2, 0, 3))
  output = jnp.reshape(updated_x, [-1, x_shape[1]] + x_shape[2:])
  return output


class MultiHeadAttention(nn.Module):
  """Calculates a multi-head dot product attention.

  Attributes:
    hidden_size: a hidden size depth.
    memory_query_shape: a tuple with a memory query shape.
    cache: cash dictionary for query, key and value.
    bias_cache: attention bias cache.
    memory_antecedent: whether there is memory antecedent.
    total_key_depth: total key depth.
    total_value_depth: total value depth.
    query_shape: a tuple with a query shape.
    memory_flange: a tuple with a memory flange shape.
    local_num_heads: a number of local attention heads.
    local_relative: whether to use a local relative attention.
    masked: indiactor to mask or not mask bias.
    decode_step: the decode step in fast decoding mode.
    cache_padding_bias: whether to cache padding bias as well to save memory.
    max_relative_position: how much distance to consider for relative positions.
    attention_dropout: an attention dropout rate.
    share_qk: whether to share the query and key.
    token_bias: a token bias array.
    padding_bias: a padding bias array.
    kernel_init: kernel initializer in the Dense layers.
    embedding_init: embeddings initializer.
  """
  hidden_size: int = 1032
  memory_query_shape: Tuple[int] = (512,)
  cache: Dict[str, Tensor] = None
  bias_cache: Dict[str, Tensor] = None
  memory_antecedent: Tensor = None
  total_key_depth: int = 1032
  total_value_depth: int = 1032
  query_shape: Tuple[int] = (256,)
  memory_flange: Tuple[int] = (256,)
  local_num_heads: int = 8
  local_relative: bool = True
  masked: bool = True
  decode_step: int = None
  cache_padding_bias: bool = False
  max_relative_position: int = 513
  attention_dropout: float = 0.0
  share_qk: bool = True
  token_bias: Tensor = None
  padding_bias: Tensor = None
  kernel_init: Any = nn.initializers.glorot_uniform()
  embedding_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(self, query_antecedent: Tensor, train: bool = False) -> Tensor:
    """Applies the MultiHeadAttention module.

    Args:
      query_antecedent: an array fo shape [batch, d1, ..., dn, depth_q].
      train: whether a deterministic mode or not.

    Returns:
      output: an array of shape [batch, d1, ..., dn, output_depth] or
      [batch, 1, ..., 1, output_depth] if decode_step is set.
    """
    if self.total_key_depth % self.local_num_heads != 0:
      raise ValueError('Key depth (%d) must be divisible by the number of '
                       'attention heads (%d).' %
                       (self.total_key_depth, self.local_num_heads))
    if self.total_value_depth % self.local_num_heads != 0:
      raise ValueError('Value depth (%d) must be divisible by the number of '
                       'attention heads (%d).' %
                       (self.total_value_depth, self.local_num_heads))

    if self.share_qk:
      if self.memory_antecedent:
        raise ValueError(f'Memory ({self.mmemory_antecedent}) must be None '
                         'if share_qk is True.')

    if self.cache is None:
      cache = {}
    else:
      cache = unfreeze(self.cache)

    if self.decode_step is not None:
      if 'q' not in cache:
        raise ValueError('Query not in cache.')
      if 'k' not in cache:
        raise ValueError('Key not in cache.')
      if 'v' not in cache:
        raise ValueError('Value not in cache.')

    if self.decode_step is not None:
      latest_antecedent = get_item_at_decode_step(
          input_array=query_antecedent,
          decode_step=self.decode_step,
          query_shape=self.query_shape)
      latest_q = nn.Dense(
          self.total_key_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='latest_q')(
              latest_antecedent)
      latest_k = nn.Dense(
          self.total_key_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='latest_k')(
              latest_antecedent)
      latest_v = nn.Dense(
          self.total_value_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='latest_v')(
              latest_antecedent)

      latest_q = split_heads_nd(
          input_x=latest_q, num_heads=self.local_num_heads)
      key_depth_per_head = self.total_key_depth // self.local_num_heads
      latest_k = split_heads_nd(
          input_x=latest_k, num_heads=self.local_num_heads)
      latest_v = split_heads_nd(
          input_x=latest_v, num_heads=self.local_num_heads)
      q_array = cache['q']
      k_array = cache['k']
      v_array = cache['v']
      q_array = put_item_in_decode_step(q_array, latest_q, self.decode_step,
                                        self.query_shape)

      if self.memory_antecedent is None:
        k_array = put_item_in_decode_step(
            input_x=k_array,
            replacement=latest_k,
            decode_step=self.decode_step,
            query_shape=self.query_shape)
        v_array = put_item_in_decode_step(
            input_x=v_array,
            replacement=latest_v,
            decode_step=self.decode_step,
            query_shape=self.query_shape)

      cache['q'] = q_array
      cache['k'] = k_array
      cache['v'] = v_array

    else:
      q_array = nn.Dense(
          self.total_key_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='q')(
              query_antecedent)
      k_array = nn.Dense(
          self.total_key_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='k')(
              query_antecedent)
      v_array = nn.Dense(
          self.total_value_depth,
          kernel_init=self.kernel_init,
          use_bias=False,
          name='v')(
              query_antecedent)

      q_array = split_heads_nd(input_x=q_array, num_heads=self.local_num_heads)
      key_depth_per_head = self.total_key_depth // self.local_num_heads
      q_array *= key_depth_per_head ** -0.5
      k_array = split_heads_nd(input_x=k_array, num_heads=self.local_num_heads)
      v_array = split_heads_nd(input_x=v_array, num_heads=self.local_num_heads)

      if cache is not None:
        cache['q'] = q_array
        cache['k'] = k_array
        cache['v'] = v_array

    if self.bias_cache is None:
      bias_cache = {}
    else:
      bias_cache = unfreeze(self.bias_cache)

    output = LocalAttention(
        query_shape=self.query_shape,
        memory_query_shape=self.memory_query_shape,
        memory_flange=self.memory_flange,
        local_num_heads=self.local_num_heads,
        local_relative=self.local_relative,
        memory_antecedent=self.memory_antecedent,
        masked=self.masked,
        decode_step=self.decode_step,
        max_relative_position=self.max_relative_position,
        cache_padding_bias=self.cache_padding_bias,
        attention_dropout=self.attention_dropout,
        bias_cache=bias_cache,
        share_qk=self.share_qk,
        token_bias=self.token_bias,
        padding_bias=self.padding_bias,
        kernel_init=self.kernel_init,
        embedding_init=self.embedding_init)(
            q_array=q_array, k_array=k_array, v_array=v_array, train=train)

    output = combine_heads_nd(input_x=output)

    output = nn.Dense(
        self.hidden_size,
        kernel_init=self.kernel_init,
        use_bias=False,
        name='output_transform')(
            output)
    return output


# TODO(krasowiak): Potentially replace lines 1506-1563
# with modified AddPositionEmbs from init2winit.model_lib.transformer_lm
def get_timing_signal_1d(length: int,
                         channels: int,
                         min_timescale: float = 1.0,
                         max_timescale: float = 1.0e4,
                         start_index: int = 0) -> Tensor:
  """Positional encoding helper function.

  Args:
    length: length of timing signal sequence.
    channels: size of timing embeddings to create.
    min_timescale: a minimium timescale.
    max_timescale: a maximum timescale.
    start_index: index of first position.

  Returns:
    outputs: an array of shape [1, length, channels].
  """
  position = jnp.arange(length + start_index, dtype=jnp.float32)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      jnp.maximum(num_timescales - 1, 1))
  inv_timescales = min_timescale * jnp.exp(
      jnp.arange(num_timescales, dtype=jnp.float32) *
      -log_timescale_increment)
  scaled_time = jnp.expand_dims(
      position, axis=1) * jnp.expand_dims(
          inv_timescales, axis=0)
  signal = jnp.concatenate(
      [jnp.sin(scaled_time),
              jnp.cos(scaled_time)], axis=1)
  signal = jnp.pad(
      signal, [[0, 0], [0, jnp.mod(channels, 2)]])
  output = jnp.reshape(signal, [1, length, channels])
  return output


class ProcessInput(nn.Module):
  """Preprocess a tokenized input array before ingesting it to decoder blocks.

  Attributes:
    query_shape: a tuple with a query shape.
    preprocess_dropout_a: a dropout rate in the first preprocessing Dropout
      layer.
    vocab_size: vocabulary size.
    preprocess_dropout_b: a dropout rate in the second preprocessing Dropout
      layer.
    hidden_size: a hidden size depth.
    add_timing_signal: whether to add positional encoding.
    max_target_length: the maximum allowed length of the tokenized sequence.
    kernel_init: kernel initializer in the Dense layers.
    preprocessing_init: initializer for the first preprocessing step.
    embedding_init: embeddings initializer.
  """
  query_shape: Tuple[int] = (256,)
  preprocess_dropout_a: float = 0.0
  vocab_size: int = 98302
  embedding_dims: int = 1032
  preprocess_dropout_b: float = 0.0
  hidden_size: int = 1032
  add_timing_signal: bool = False
  max_target_length: int = 8192
  kernel_init: Any = nn.initializers.glorot_uniform()
  preprocessing_init: Any = nn.initializers.uniform()
  embedding_init: Any = nn.initializers.glorot_uniform()
  embed_layer_name: str = 'embeddings'

  def setup(self):
    self.embedding = self.param(self.embed_layer_name,
                                self.embedding_init,
                                (self.vocab_size, self.embedding_dims))

  @nn.compact
  def __call__(self, input_x: Tensor, train: bool = False) -> Tensor:
    """Applies the ProcessInput module.

    Args:
      input_x: a tokenized input of shape [batch, max_target_length] to
        preprocess.
      train: whether a deterministic mode or not.

    Returns:
      output: a preprocessed input array of shape [batch, max_target_length,
      hidden_size].
    """
    shape = list(input_x.shape)
    if len(shape) != 2:
      raise ValueError(f'Input shape length ({shape}) is not equal 2.')

    if len(self.query_shape) != 1:
      raise ValueError(f'Query length ({self.query_shape}) is not equal 1.')

    if self.preprocess_dropout_a:
      dropout_array = self.param('dropout_array', self.preprocessing_init,
                                 shape)
      input_x = jnp.where(dropout_array < self.preprocess_dropout_a,
                          jnp.zeros_like(input_x), input_x)

    output = jnp.expand_dims(input_x, axis=-1)
    output = shift_right(output)
    output = jnp.squeeze(output, axis=-1)
    output = jnp.take(self.embedding, output, axis=0)

    if self.preprocess_dropout_b:
      output = nn.Dropout(
          rate=self.preprocess_dropout_b, deterministic=not train)(
              output)

    output = nn.Dense(
        self.hidden_size,
        kernel_init=self.kernel_init,
        use_bias=True,
        name='emb_dense')(
            output)

    if self.add_timing_signal:
      output += get_timing_signal_1d(
          length=self.max_target_length, channels=self.hidden_size)
    return output


def process_partial_targets_decoding(
    targets: Tensor, query_shape: Tuple[int] = (256,)) -> Tensor:
  """Preprocesses tokenized input sequences in the decoding process.

  Args:
    targets: array of shape [batch, max_target_length].
    query_shape: a tuple with a query shape.

  Returns:
    outputs: a processed input array [batch, max_target_length].
  """
  targets_shape = list(targets.shape)
  seq_length = targets_shape[1]
  blocks_per_dim = [seq_length // q for q in query_shape]
  targets = jnp.reshape(
      targets,
      [
          targets_shape[0], -1,
          np.prod(query_shape), 1
      ])
  targets = unflatten_blocks_nd(
      input_x=targets, blocks_per_dimension=blocks_per_dim)
  targets = put_back_blocks_nd(input_x=targets, block_shape=query_shape)
  outputs = jnp.reshape(targets, [-1, seq_length])
  return outputs


class LayerPostProcess(nn.Module):
  """Postprocesses a decoder output.

  Attributes:
    post_attention_epsilon: an epsilon value for LayerNorm.
    post_attention_dropout: a dropout rate in the postprocessing process.
  """
  post_attention_epsilon: float = 1e-6
  post_attention_dropout: float = 0.1

  @nn.compact
  def __call__(self,
               input_x: Tensor,
               input_y: Tensor,
               train: bool = False) -> Tensor:
    """Applies the LayerPostProcess module.

    Args:
      input_x: an array with the decoder output of shape [batch,
        max_target_length, hidden_size].
      input_y: an array with the decoder and FeedForward layer output of shape
        [batch, max_target_length, hidden_size].
      train: whether a deterministic mode or not.

    Returns:
      output: a postprocessed array of shape [batch, max_target_length,
      hidden_size].
    """
    y = nn.Dropout(
        rate=self.post_attention_dropout, deterministic=not train)(input_y)

    output = y + input_x

    output = nn.LayerNorm(
        epsilon=self.post_attention_epsilon, name='layer_norm')(output)

    return output


class DecoderBlock(nn.Module):
  """Passess a preprocessed tokenized sequence through a decoder block.

  Attributes:
    layer: a number of the decoder layer in the transformer.
    decoder_dropout_a: a dropout rate of the first Dropout layer in the decoder
      block.
    post_attention_epsilon: an epsilon value for LayerNorm.
    feedforward_depths: number of neurons in the 1st and 2nd Dense layers.
    feedforward_dropout: dropout rate in the Dropout layers.
    cache: cash dictionary for query, key and value.
    bias_cache: attention bias cache.
    memory_antecedent: whether there is memory antecedent.
    total_key_depth: total key depth.
    total_value_depth: total value depth.
    query_shape: a tuple with a query shape.
    memory_query_shape: a tuple with a memory query shape.
    memory_flange: a tuple with a memory flange shape.
    local_num_heads: a number of local attention heads.
    local_relative: whether to use a local relative attention.
    masked: indiactor to mask or not mask bias.
    decode_step: the decode step in fast decoding mode.
    cache_padding_bias: whether to cache padding bias as well to save memory.
    max_relative_position: how much distance to consider for relative positions.
    attention_dropout: an attention dropout rate.
    post_attention_dropout: a dropout rate in the postprocessing process.
    share_qk: whether to share the query and key.
    token_bias: a token bias array.
    padding_bias: a padding bias array.
    hidden_size: a hidden size depth.
    kernel_init: kernel initializer in the Dense layers.
    embedding_init: embeddings initializer.
  """
  layer: int = None
  decoder_dropout_a: float = 0.1
  post_attention_epsilon: float = 1e-6
  feedforward_depths: Sequence[int] = None
  feedforward_dropout: float = 0.0
  cache: Dict[str, Tensor] = None
  bias_cache: Dict[str, Tensor] = None
  memory_antecedent: Tensor = None
  total_key_depth: int = 1032
  total_value_depth: int = 1032
  query_shape: Tuple[int] = (256,)
  memory_query_shape: Tuple[int] = (512,)
  memory_flange: int = (256,)
  local_num_heads: int = 8
  local_relative: bool = True
  masked: bool = True
  decode_step: int = None
  cache_padding_bias: bool = False
  max_relative_position: int = 513
  attention_dropout: float = 0.0
  post_attention_dropout: float = 0.1
  share_qk: bool = True
  token_bias: Tensor = None
  padding_bias: Tensor = None
  hidden_size: int = 1032
  kernel_init: Any = nn.initializers.glorot_uniform()
  embedding_init: Any = nn.initializers.glorot_uniform()

  @nn.compact
  def __call__(self, input_x: Tensor, train: bool = False) -> Tensor:
    """Applies the DecoderBlock module.

    Args:
      input_x: a preprocessed tokenized input in an array fo shape [batch,
        max_target_length, hidden_size].
      train: boolean indicating whether training or not.

    Returns:
      output: an array of shape [batch, max_target_length, hidden_size].
    """
    x = nn.Dropout(
        rate=self.decoder_dropout_a, deterministic=not train)(input_x)

    if self.cache is None:
      cache = {}
    else:
      cache = unfreeze(self.cache)

    if self.decode_step is None and self.cache is not None:
      cache[self.layer] = {}
      layer_cache = cache[self.layer]

    if self.decode_step is not None:
      layer_cache = cache[self.layer]
    else:
      layer_cache = cache

    key_depth = self.total_key_depth or self.hidden_size
    value_depth = self.total_value_depth or self.hidden_size

    y = MultiHeadAttention(
        cache=layer_cache,
        bias_cache=self.bias_cache,
        memory_antecedent=self.memory_antecedent,
        total_key_depth=key_depth,
        total_value_depth=value_depth,
        hidden_size=self.hidden_size,
        query_shape=self.query_shape,
        memory_query_shape=self.memory_query_shape,
        memory_flange=self.memory_flange,
        local_num_heads=self.local_num_heads,
        local_relative=self.local_relative,
        masked=self.masked,
        decode_step=self.decode_step,
        cache_padding_bias=self.cache_padding_bias,
        max_relative_position=self.max_relative_position,
        attention_dropout=self.attention_dropout,
        share_qk=self.share_qk,
        token_bias=self.token_bias,
        padding_bias=self.padding_bias,
        kernel_init=self.kernel_init,
        embedding_init=self.embedding_init)(query_antecedent=x, train=train)

    x = LayerPostProcess(
        post_attention_epsilon=self.post_attention_epsilon,
        post_attention_dropout=self.post_attention_dropout)(
            input_x=x, input_y=y, train=train)

    y = FeedForward(
        feedforward_depths=self.feedforward_depths,
        feedforward_dropout=self.feedforward_dropout,
        kernel_init=self.kernel_init)(
            input_x=x, train=train)

    output = LayerPostProcess(
        post_attention_epsilon=self.post_attention_epsilon,
        post_attention_dropout=self.post_attention_dropout)(
            input_x=x, input_y=y, train=train)

    if self.decode_step is not None:
      output = get_item_at_decode_step(
          input_array=output,
          decode_step=self.decode_step,
          query_shape=self.query_shape)
    return output


class LocalAttentionTransformerArchitecture(nn.Module):
  """Runs and end-to-end forward pass through the transformer with the local attention.

  Attributes:
    kernel_init: kernel initializer in the Dense layers.
    preprocessing_init: initializer for the first preprocessing step.
    embedding_init: embeddings initializer.
    decode: whether to run in decode mode.
    decode_step: the decode step in fast decoding mode.
    query_shape: a tuple with a query shape.
    max_target_length: the maximum allowed length of the tokenized sequence.
    preprocess_dropout_a: a dropout rate in the first preprocessing Dropout
          layer.
    embedding_dims: an embedding dimension.
    vocab_size: vocabulary size.
    preprocess_dropout_b: a dropout rate in the second preprocessing Dropout
          layer.
    hidden_size: a hidden size depth.
    add_timing_signal: whether to add positional encoding.
    num_decoder_layers: number of decoder blocks/layers to use in the
      transformer.
    padding_bias: a padding bias array.
    decoder_dropout_a: a dropout rate of the first Dropout layer in the decoder
      block.
    total_key_depth: total key depth.
    total_value_depth: total value depth.
    bias_cache: attention bias cache.
    local_num_heads: a number of local attention heads.
    cache: cash dictionary for query, key and value.
    memory_query_shape: a tuple with a memory query shape.
    memory_flange: a tuple with a memory flange shape.
    cache_padding_bias: whether to cache padding bias as well to save memory.
    max_relative_position: how much distance to consider for relative positions.
    attention_dropout: an attention dropout rate.
    memory_antecedent: whether there is memory antecedent.
    masked: indiactor to mask or not mask bias.
    local_relative: whether to use a local relative attention.
    share_qk: whether to share the query and key.
    token_bias: a token bias array.
    post_attention_epsilon: an epsilon value for LayerNorm.
    post_attention_dropout: a dropout rate in the postprocessing process.
    feedforward_dropout: dropout rate in the Dropout layers.
    feedforward_depths: number of neurons in the 1st and 2nd Dense layers.
    dtype: data types of the final logits.
  """
  kernel_init: Any = nn.initializers.glorot_uniform()
  preprocessing_init: Any = nn.initializers.uniform()
  embedding_init: Any = nn.initializers.glorot_uniform()
  decode: bool = False
  decode_step: int = None
  query_shape: Tuple[int] = (256,)
  max_target_length: int = 8192
  preprocess_dropout_a: float = 0.0
  embedding_dims: int = 1032
  vocab_size: int = 98302
  preprocess_dropout_b: float = 0.0
  hidden_size: int = 1032
  add_timing_signal: bool = False
  num_decoder_layers: int = 24
  padding_bias: Tensor = None
  decoder_dropout_a: float = 0.1
  total_key_depth: int = 1032
  total_value_depth: int = 1032
  bias_cache: Dict[str, Tensor] = None
  local_num_heads: int = 8
  cache: Dict[str, Tensor] = None
  memory_query_shape: Tuple[int] = (512,)
  memory_flange: Tuple[int] = (256,)
  cache_padding_bias: bool = False
  max_relative_position: int = 513
  attention_dropout: float = 0.0
  memory_antecedent: Tensor = None
  masked: bool = True
  local_relative: bool = True
  share_qk: bool = True
  token_bias: Tensor = None
  post_attention_epsilon: float = 1e-6
  post_attention_dropout: float = 0.1
  feedforward_dropout: float = 0.0
  feedforward_depths: Sequence[int] = None
  dtype: str = 'float32'

  @nn.compact
  def __call__(self,
               input_x: Tensor,
               train: bool = False) -> Tensor:
    """Applies the TransformerLocalAttentionArchitecture module.

    Args:
      input_x: a preprocessed tokenized input array of shape [batch,
        max_target_length].
      train: boolean indicating whether training or not.

    Returns:
      output: an array of shape [batch, max_target_length, vocab_size] with
      logits.
    """
    if not self.decode:
      decode_step = None
    else:
      decode_step = self.decode_step

    x = input_x.astype('int32')

    if self.decode:
      x = process_partial_targets_decoding(
          targets=x, query_shape=self.query_shape)

    x = ProcessInput(
        preprocess_dropout_a=self.preprocess_dropout_a,
        vocab_size=self.vocab_size,
        embedding_dims=self.embedding_dims,
        preprocess_dropout_b=self.preprocess_dropout_b,
        hidden_size=self.hidden_size,
        add_timing_signal=self.add_timing_signal,
        max_target_length=self.max_target_length,
        kernel_init=self.kernel_init,
        preprocessing_init=self.preprocessing_init,
        embedding_init=self.embedding_init)(
            input_x=x, train=train)

    for layer in range(self.num_decoder_layers):
      y = DecoderBlock(
          layer=layer,
          decoder_dropout_a=self.decoder_dropout_a,
          decode_step=decode_step,
          cache=self.cache,
          total_key_depth=self.total_key_depth,
          total_value_depth=self.total_value_depth,
          hidden_size=self.hidden_size,
          bias_cache=self.bias_cache,
          memory_antecedent=self.memory_antecedent,
          query_shape=self.query_shape,
          memory_query_shape=self.memory_query_shape,
          memory_flange=self.memory_flange,
          local_num_heads=self.local_num_heads,
          local_relative=self.local_relative,
          masked=self.masked,
          cache_padding_bias=self.cache_padding_bias,
          max_relative_position=self.max_relative_position,
          attention_dropout=self.attention_dropout,
          share_qk=self.share_qk,
          token_bias=self.token_bias,
          padding_bias=self.padding_bias,
          post_attention_epsilon=self.post_attention_epsilon,
          post_attention_dropout=self.post_attention_dropout,
          feedforward_depths=self.feedforward_depths,
          feedforward_dropout=self.feedforward_dropout,
          kernel_init=self.kernel_init,
          embedding_init=self.embedding_init)(
              input_x=x, train=train)

    output = nn.Dense(
        self.vocab_size,
        kernel_init=self.kernel_init,
        use_bias=True,
        name='final_dense_2')(y)

    output = output.astype(self.dtype)
    return output


class LocalAttentionTransformer(base_model.BaseModel):
  """Transformer with local attention, comaptible with a PG19 dataset."""

  def evaluate_batch(self, params, batch_stats, batch):
    """Returns evaulation metrics on the given batch."""
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = self.flax_module.apply(
        variables, batch['inputs'], mutable=False, train=False)
    targets = batch['targets']
    # Class 0 is reserved for padding
    weights = jnp.not_equal(targets, 0).astype(jnp.float32)

    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = jax.nn.one_hot(targets, logits.shape[-1])

    return self.metrics_bundle.gather_from_model_output(
        logits=logits,
        targets=targets,
        weights=weights,
        axis_name='batch')

  def training_cost(self, params, batch, batch_stats=None, dropout_rng=None):
    """Returns loss."""
    all_variables = {'params': params}
    apply_kwargs = {'train': True}

    if batch_stats is not None:
      all_variables['batch_stats'] = batch_stats
      apply_kwargs['mutable'] = ['batch_stats']

    if dropout_rng is not None:
      apply_kwargs['rngs'] = {'dropout': dropout_rng}

    logits, new_batch_stats = self.flax_module.apply(
        all_variables, batch['inputs'], **apply_kwargs)
    targets = batch['targets']
    # Class 0 is reserved for padding
    weights = jnp.not_equal(targets, 0).astype(jnp.float32)

    if self.dataset_meta_data['apply_one_hot_in_loss']:
      targets = jax.nn.one_hot(targets, logits.shape[-1])

    total_loss = self.loss_fn(logits, targets, weights)

    return total_loss, new_batch_stats

  def build_flax_module(self):
    """Builds the module of the transformer with local attention."""
    return LocalAttentionTransformerArchitecture(
        kernel_init=INITIALIZERS[self.hps.kernel_init],
        preprocessing_init=INITIALIZERS[self.hps.preprocessing_init],
        embedding_init=INITIALIZERS[self.hps.embedding_init],
        decode=self.hps.decode,
        decode_step=self.hps.decode_step,
        query_shape=self.hps.query_shape,
        max_target_length=self.hps.max_target_length,
        preprocess_dropout_a=self.hps.preprocess_dropout_a,
        embedding_dims=self.hps.embedding_dims,
        vocab_size=self.hps.vocab_size,
        preprocess_dropout_b=self.hps.preprocess_dropout_b,
        hidden_size=self.hps.hidden_size,
        add_timing_signal=self.hps.add_timing_signal,
        num_decoder_layers=self.hps.num_decoder_layers,
        padding_bias=self.hps.padding_bias,
        decoder_dropout_a=self.hps.decoder_dropout_a,
        total_key_depth=self.hps.total_key_depth,
        total_value_depth=self.hps.total_value_depth,
        bias_cache=self.hps.bias_cache,
        local_num_heads=self.hps.local_num_heads,
        cache=self.hps.cache,
        memory_query_shape=self.hps.memory_query_shape,
        memory_flange=self.hps.memory_flange,
        cache_padding_bias=self.hps.cache_padding_bias,
        max_relative_position=self.hps.max_relative_position,
        attention_dropout=self.hps.attention_dropout,
        memory_antecedent=self.hps.memory_antecedent,
        masked=self.hps.masked,
        local_relative=self.hps.local_relative,
        share_qk=self.hps.share_qk,
        token_bias=self.hps.token_bias,
        post_attention_epsilon=self.hps.post_attention_epsilon,
        post_attention_dropout=self.hps.post_attention_dropout,
        feedforward_dropout=self.hps.feedforward_dropout,
        feedforward_depths=self.hps.feedforward_depths,
        dtype=self.hps.model_dtype)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initialzing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs
