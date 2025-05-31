# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Flax layers with binarization support.

BinarizeOps is adapted from third_party/py/aqt/jax_legacy/jax/primitives.py.
Others are adapted from
third_party/py/flax/linen/linear.py and third_party/py/flax/linen/attention.py.
"""


import dataclasses
import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from flax import struct as flax_struct
from flax.linen.attention import combine_masks
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import zeros
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module
from init2winit.model_lib import model_utils
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
dataclass = flax_struct.dataclass if not TYPE_CHECKING else dataclasses.dataclass
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()


def _default_binarize_hparams() -> config_dict.ConfigDict:
  return config_dict.ConfigDict({
      'w_hparams': None,  # no weight binarization by default
      'a_hparams': None,  # no activation binarization by default
  })


@dataclass
class DynamicContext:
  """Quantization flags visible to all model layers. Used for delayed quant.

  A single instance of this dataclass is intended to be threaded throughout the
  entire hierarchy of model layers, making it easy when the model is called
  (whether for training or inference) for the caller to set contextual
  variables that all parts of the model that deal with quantization need.
  """

  # The construct `field_name = flax.struct.field(pytree_node=False)` causes a
  # field to be treated as a *static* argument to a JITed function. ie, the
  # function will recompile when the value of that field changes, but normal
  # Python control flow can be used inside the model (eg, `if
  # context.update_bounds:`) with those fields. Other fields will be treated as
  # *dynamic*, so the model will not recompile when those fields change, but
  # they cannot be used in normal Python control flow (typically, you would use
  # `lax.cond` instead).

  # quantization flags for feed forward layers
  quant_ff_weights: bool = flax_struct.field(default=False, pytree_node=False)
  quant_ff_acts: bool = flax_struct.field(default=False, pytree_node=False)

  # quantization flags for attention layers
  quant_att_weights: bool = flax_struct.field(default=False, pytree_node=False)
  quant_att_out_acts: bool = flax_struct.field(default=False, pytree_node=False)
  quant_att_kqv_acts: bool = flax_struct.field(default=False, pytree_node=False)


def add_straight_through_estimator(jax_function) -> None:
  """Defines the gradient of a function to be the straight-through-estimator.

  Specifically, the Jacobian-vector product associated with the function is
  defined to be the identity.

  This causes Jax to effectively ignore this function in the backwards pass.

  Args:
    jax_function: A Jax function that has been decorated with @jax.custom_vjp.
      It is expected to take in and return one positional argument.
  """

  # See
  # https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
  def ste(primals, tangents):
    return jax_function(primals[0]), tangents[0]

  jax_function.defjvp(ste)


@jax.custom_jvp
def floor_with_gradient(x: jax.Array) -> jax.Array:
  """Floor with Straight-Through-Estimator gradient."""
  return jnp.floor(x)

add_straight_through_estimator(floor_with_gradient)


class BinarizeOps:
  """Binarization operator.
  """

  @dataclasses.dataclass
  class HParams:
    """Hyperparameters for binarization."""
    # A fixed quantization bound within which inputs have gradients
    bound: Optional[Union[float, jnp.ndarray]]
    # A small value subtracted from the clipping bound to prevent from overflow
    epsilon: float
    # Axis along which to automatically get bound values
    scale_axis: Optional[Union[Iterable[int], str]]

  def __init__(self,
               bound: jnp.ndarray,
               epsilon: float,  # default value 2**(-7)
               dtype: Any):
    self.bound = bound
    self.epsilon = epsilon
    self.dtype = dtype

  @classmethod
  def create_ops(cls,
                 x: jnp.ndarray,
                 hparams: HParams,
                 dtype: Any = jnp.bfloat16):
    """Create BinarizationOps for symmetric inputs clipped to [-bounds, bounds].

    Args:
      x: input array
      hparams: hyperparameters for the binarization function
      dtype: return datatype

    Returns:
      QuantOps for quantizing/dequantizing signed activations.
    """
    if hparams.bound is None:
      # When bound is None, use the maximum abs value along the specified
      # scale_axis in the array as the bound.
      # Essentially no inputs will be clipped during forward pass, so gradients
      # will all pass through during the backprop.
      bound = jnp.max(jnp.abs(x), axis=hparams.scale_axis, keepdims=True)
    else:
      bound = jnp.abs(hparams.bound)  # ensure that bound > 0
    bound = jnp.asarray(bound, dtype)
    bound += jnp.finfo(jnp.float32).eps  # avoid dividing by zero
    bound = jax.lax.stop_gradient(bound)  # no gradietns on the bound
    return cls(
        bound=bound,
        dtype=dtype,
        epsilon=hparams.epsilon)

  def binarize(self, x: jnp.ndarray) -> jnp.ndarray:
    """The binarization operation.

    (-inf, 0) is mapped to -bound/2, and [0, +inf) is mapped to +bound/2.
    Unlike jnp.sign, this operation avoids mapping input zeros to output zeros.

    Args:
      x: input array

    Returns:
      Fake binarized data
    """
    scale = jnp.divide(1.0, self.bound)
    x = jnp.multiply(x, scale)
    clip_bound = 1.0 - self.epsilon
    x = jnp.clip(x, min=-clip_bound, max=clip_bound).astype(self.dtype)
    x = floor_with_gradient(x) + 0.5  # x is either -0.5 or +0.5
    x = jnp.divide(x, scale)
    return x.astype(self.dtype)


class BiDense(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    weight_bin_hparams: hyperparameters for binarizing the kernels.
    inputs_bin_hparams: hyperparameters for binarizing the inputs.
  """
  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  weight_bin_hparams: Optional[BinarizeOps.HParams] = None
  inputs_bin_hparams: Optional[BinarizeOps.HParams] = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param('kernel',
                        self.kernel_init,
                        (jnp.shape(inputs)[-1], self.features),
                        self.param_dtype)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,),
                        self.param_dtype)
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    # binarization ops
    # ----------------------
    if self.weight_bin_hparams is not None:
      whps = self.weight_bin_hparams
      assert whps.scale_axis in [
          'auto', None
      ], f'scale_axis can only be auto or None. Got {whps.scale_axis}.'
      if whps.scale_axis == 'auto':
        # automatically select one bound per out_feature
        whps = whps.to_dict()
        whps['scale_axis'] = (0,)
        whps = config_dict.ConfigDict(whps)
      weight_ops = BinarizeOps.create_ops(kernel, whps, dtype=self.dtype)
      w_bound_shape = weight_ops.bound.shape
      assert w_bound_shape == (1, self.features) or w_bound_shape == (
          1, 1), f'kernel bound shape {w_bound_shape} is not realistic'
      kernel = weight_ops.binarize(kernel)
    if self.inputs_bin_hparams is not None:
      ahps = self.inputs_bin_hparams
      assert ahps.scale_axis in [
          'auto', None
      ], f'scale_axis can only be auto or None. Got {ahps.scale_axis}.'
      if ahps.scale_axis == 'auto':
        ahps = ahps.to_dict()
        ahps['scale_axis'] = (2,)
        ahps = config_dict.ConfigDict(ahps)
      inputs_ops = BinarizeOps.create_ops(inputs, ahps, dtype=self.dtype)
      a_bound_shape = inputs_ops.bound.shape
      assert a_bound_shape == (
          inputs.shape[0], inputs.shape[1], 1) or a_bound_shape == (
              1, 1, 1), f'inputs bound shape {a_bound_shape} is not realistic'
      inputs = inputs_ops.binarize(inputs)
    # ----------------------
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class BiDenseGeneral(Module):
  """A linear transformation with flexible axes.

  Attributes:
    features: int or tuple with number of output features.
    axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    batch_dims: tuple with batch axes.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    weight_bin_hparams: hyperparameters for binarizing the kernels.
    inputs_bin_hparams: hyperparameters for binarizing the inputs.
  """
  features: Union[int, Sequence[int]]
  axis: Union[int, Sequence[int]] = -1
  batch_dims: Sequence[int] = ()
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  precision: PrecisionLike = None
  weight_bin_hparams: Optional[BinarizeOps.HParams] = None
  inputs_bin_hparams: Optional[BinarizeOps.HParams] = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)
    batch_dims = _canonicalize_tuple(self.batch_dims)
    if batch_dims:
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError('batch_dims %s must be consecutive leading '
                         'dimensions starting from 0.' % str(batch_dims))

    ndim = inputs.ndim
    n_batch_dims = len(batch_dims)
    axis = _normalize_axes(axis, ndim)
    batch_dims = _normalize_axes(batch_dims, ndim)
    n_axis, n_features = len(axis), len(features)

    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
      flat_shape = (np.prod(shape[n_batch_dims:n_axis + n_batch_dims]),
                    np.prod(shape[-n_features:]),)
      kernel = jnp.concatenate([self.kernel_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
      return jnp.reshape(kernel, shape)

    batch_shape = tuple(inputs.shape[ax] for ax in batch_dims)
    # batch and non-contracting dims of input with 1s for batch dims.
    expanded_batch_shape = tuple(
        inputs.shape[ax] if ax in batch_dims else 1
        for ax in range(inputs.ndim) if ax not in axis)
    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel = self.param('kernel', kernel_init_wrap, batch_shape + kernel_shape,
                        self.param_dtype)

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))

    if self.use_bias:
      def bias_init_wrap(rng, shape, dtype=jnp.float32):
        size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
        flat_shape = (np.prod(shape[-n_features:]),)
        bias = jnp.concatenate([self.bias_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
        return jnp.reshape(bias, shape)

      bias = self.param('bias', bias_init_wrap, batch_shape + features,
                        self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    # binarization ops
    # ----------------------
    if self.weight_bin_hparams is not None:
      whps = self.weight_bin_hparams
      assert whps.scale_axis in [
          'auto', None
      ], f'scale_axis can only be auto or None. Got {whps.scale_axis}.'
      if whps.scale_axis == 'auto':
        # automatically select one bound per out_feature
        whps = whps.to_dict()
        whps['scale_axis'] = tuple(range(n_axis))
        whps = config_dict.ConfigDict(whps)
      weight_ops = BinarizeOps.create_ops(kernel, whps, dtype=self.dtype)
      w_bound_shape = weight_ops.bound.shape
      assert w_bound_shape == (1,) * n_axis + features or w_bound_shape == (
          1,
      ) * kernel.ndim, f'kernel bound shape {w_bound_shape} is not realistic'
      kernel = weight_ops.binarize(kernel)
    if self.inputs_bin_hparams is not None:
      ahps = self.inputs_bin_hparams
      assert ahps.scale_axis in [
          'auto', None
      ], f'scale_axis can only be auto or None. Got {ahps.scale_axis}.'
      if ahps.scale_axis == 'auto':
        ahps = ahps.to_dict()
        ahps['scale_axis'] = axis
        ahps = config_dict.ConfigDict(ahps)
      inputs_ops = BinarizeOps.create_ops(inputs, ahps, dtype=self.dtype)
      a_bound_shape = inputs_ops.bound.shape
      assert a_bound_shape == tuple([
          inputs.shape[i] for i in range(ndim - n_axis)
      ]) + (1,) * n_axis or a_bound_shape == (
          1,) * ndim, f'inputs bound shape {a_bound_shape} is not realistic'
      inputs = inputs_ops.binarize(inputs)
    # ----------------------

    out = lax.dot_general(inputs,
                          kernel,
                          ((axis, contract_ind), (batch_dims, batch_ind)),
                          precision=self.precision)
    # dot_general output has shape [batch_dims/group_dims] + [feature_dims]
    if self.use_bias:
      # expand bias shape to broadcast bias over batch dims.
      bias = jnp.reshape(bias, expanded_batch_shape + features)
      out += bias
    return out


def dot_product_attention_weights(query: Array,
                                  key: Array,
                                  bias: Optional[Array] = None,
                                  mask: Optional[Array] = None,
                                  broadcast_dropout: bool = True,
                                  dropout_rng: Optional[PRNGKey] = None,
                                  dropout_rate: float = 0.,
                                  deterministic: bool = False,
                                  dtype: Optional[Dtype] = None,
                                  precision: PrecisionLike = None):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype(query, key, dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], (
      'q, k batch dims must match.')
  assert query.shape[-2] == key.shape[-2], (
      'q, k num_heads must match.')
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                            precision=precision)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = (keep.astype(dtype) /
                  jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          mask: Optional[Array] = None,
                          broadcast_dropout: bool = True,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: Optional[Dtype] = None,
                          precision: PrecisionLike = None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype(query, key, value, dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
      query, key, bias, mask, broadcast_dropout, dropout_rng, dropout_rate,
      deterministic, dtype, precision)

  # return weighted sum over values for each query position
  return jnp.einsum('...hqk,...khd->...qhd', attn_weights, value,
                    precision=precision)


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
      binarize_hparams: hyperparameters for binarization.
      dynamic_context: contains flags that decide if performing quantization.
  """
  num_heads: int
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False
  binarize_hparams: config_dict.ConfigDict = dataclasses.field(
      default_factory=_default_binarize_hparams
  )
  dynamic_context: DynamicContext = dataclasses.field(
      default_factory=DynamicContext
  )

  @compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    # quantization hps are set to None before flags are turned on
    quant_att_weights = self.dynamic_context.quant_att_weights
    quant_att_out_acts = self.dynamic_context.quant_att_out_acts
    quant_att_kqv_acts = self.dynamic_context.quant_att_kqv_acts
    whps = self.binarize_hparams.w_hparams if quant_att_weights else None
    out_ahps = self.binarize_hparams.a_hparams if quant_att_out_acts else None
    kqv_ahps = self.binarize_hparams.a_hparams if quant_att_kqv_acts else None

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        BiDenseGeneral,  # use the binarized version of DenseGeneral
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        weight_bin_hparams=whps,
        inputs_bin_hparams=kqv_ahps)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(inputs_q),
                         dense(name='key')(inputs_kv),
                         dense(name='value')(inputs_kv))

    # add layernorms after K,Q,V transformation to support binarization
    query_normalize = model_utils.get_normalizer(
        'layer_norm', not deterministic, dtype=self.dtype)
    key_normalize = model_utils.get_normalizer(
        'layer_norm', not deterministic, dtype=self.dtype)
    value_normalize = model_utils.get_normalizer(
        'layer_norm', not deterministic, dtype=self.dtype)
    query = query_normalize()(query)
    key = key_normalize()(key)
    value = value_normalize()(value)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    dropout_rng = None
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      m_deterministic = merge_param('deterministic', self.deterministic,
                                    deterministic)
      if not m_deterministic:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = BiDenseGeneral(  # use the binarized version of DenseGeneral
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        weight_bin_hparams=whps,
        inputs_bin_hparams=out_ahps,
        name='out')(x)

    # add a layernorm followed by a local shortcut to support binarization
    out_normalize = model_utils.get_normalizer(
        'layer_norm', not deterministic, dtype=self.dtype)
    out = out_normalize()(out)
    out = out + jnp.reshape(x, out.shape)
    return out


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention."""

  @compact
  def __call__(self, inputs_q: Array, mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product self-attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    return super().__call__(inputs_q, inputs_q, mask,
                            deterministic=deterministic)
