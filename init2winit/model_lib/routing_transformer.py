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

"""Transformer-based language model with local and routing attention.

Base transfomer architecture:
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

Base architecture verified with the experiment from lm1b_flax_repro.py

Local and routing attention:
https://arxiv.org/pdf/2003.05997.pdf
"""
from typing import Any
from typing import Optional
from typing import Union
from flax import linen as nn
from init2winit import utils
from init2winit.model_lib import base_model
from jax import lax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        # Model hyperparameters
        emb_dim=128,
        num_heads=8,
        num_layers=6,
        qkv_dim=128,
        mlp_dim=512,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        attention_function=None,
        model_dtype='float32',
        pre_normalization=True,
        post_normalization=True,
        # Data pipeline hyperparameters
        batch_size=512,
        # Training hyperparameters
        grad_clip=None,
        use_shallue_label_smoothing=False,
        label_smoothing=None,
        l2_decay_factor=None,
        l2_decay_rank_threshold=0,
        rng_seed=-1,
        lr_hparams={
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup'
        },
        optimizer='adam',
        opt_hparams={
            'beta1': .9,
            'beta2': .98,
            'epsilon': 1e-9,
            'weight_decay': 0.1
        },
        layer_rescale_factors={},
    ))


class FeedForward(nn.Module):
  """Feedforward block in a transformer block.

  Attributes:
    mlp_dim: number of neurons in the 1st Dense layer
    dropout_rate: dropout rate in the Dropout layers
    dim_model: number of neurons in the 2nd/output Dense layer
    kernel_init: kernel initializer in the Dense layers
    bias_init: bias initialiazer in the Dense layers
  """
  mlp_dim: int = 512
  dropout_rate: float = 0.1
  dim_model: Optional[int] = None
  kernel_init: Any = nn.initializers.xavier_uniform()
  bias_init: Any = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self,
               input_x: Union[np.array, jnp.ndarray],
               train: bool = False) -> Union[np.array, jnp.ndarray]:
    """Applies the FeedForward block.

    Args:
      input_x: input data
      train: boolean indicating whether training or not

    Returns:
      output: an array transfomed by the feedforward block
    """
    dim_model = input_x.shape[-1] if self.dim_model is None else self.dim_model
    x = nn.Dense(
        features=self.mlp_dim,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs=input_x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(inputs=x)
    x = nn.Dense(
        features=dim_model,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs=x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(inputs=x)
    return x


class TransformerBlock(nn.Module):
  """A single transformer block.

  Attributes:
    qkv_dim: dimension of the query/key/value
    mlp_dim: number of neurons in the 1st Dense layer in the feedforward block
    num_heads: number of attention heads
    dropout_rate: dropout rate in the Dropout layers in the feedforward block
    attention_dropout_rate: dropout rate in the Dropout layers in the attention
      block
    attention_function: attention function to use in the attention block
    dtype: data type to be used in the model training
    pre_normalization: boolean if the data should be normalized before feeding
      it to the attention block and to the feedforward block
    post_normalization: boolean if the data should be normalized after
      processing it by the attention block and to the feedforward block
  """
  qkv_dim: int = 512
  mlp_dim: int = 512
  num_heads: int = 8
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_function: Any = None
  dtype: Any = jnp.float32
  pre_normalization: bool = True
  post_normalization: bool = True

  @nn.compact
  def __call__(self,
               input_x: Union[np.array, jnp.ndarray],
               mask: Union[np.array, jnp.ndarray],
               train: bool = False) -> Union[np.array, jnp.ndarray]:
    """Applies the TransformerBlock.

    Args:
      input_x: input data
      mask: self_attention mask
      train: boolean indicating whether training or not

    Returns:
      output: an array transfomed by the transformer block
    """
    original_input_x = input_x
    if self.pre_normalization:
      x = nn.LayerNorm(dtype=self.dtype)(input_x)
    else:
      x = input_x

    # NOTE(krasowiak): build local and routing attention functions
    if self.attention_function is None:
      attention_function = nn.dot_product_attention
    else:
      attention_function = self.attention_function

    x = nn.SelfAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=nn.initializers.xavier_uniform(),
        use_bias=False,
        broadcast_dropout=False,
        dtype=self.dtype,
        attention_fn=attention_function,
        deterministic=not train)(x, mask=mask)

    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = x + original_input_x

    if self.post_normalization or self.pre_normalization:
      x = nn.LayerNorm(dtype=self.dtype)(x)

    feedforward_output = FeedForward(
        mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate)(
            x, train=train)
    transformer_layer_output = feedforward_output + x

    if self.post_normalization:
      transformer_layer_output = nn.LayerNorm(dtype=self.dtype)(
          transformer_layer_output)

    return transformer_layer_output


class PositionalEmbeddings(nn.Module):
  """Adds positional embeddings to the input.

  Attributes:
    max_len: the maximum possible length of the input
  """
  max_len: int = 2048

  def positional_encoding_support_function(self,
                                           max_len: int = 2048,
                                           dim_feature: int = 512):
    """Sinusoidal position embedding support function.

    Args:
      max_len: maximum possible length for the input
      dim_feature: number of feature dimensions

    Returns:
      output: a table with fixed sinusoidal embedding table
    """
    pe = jnp.zeros((max_len, dim_feature))
    position = jnp.expand_dims(jnp.arange(0, max_len), 1)
    div_term = jnp.exp(
        jnp.arange(0, dim_feature, 2) * -jnp.log(10000) / dim_feature)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    pe = jnp.expand_dims(pe, 0)
    return pe

  @nn.compact
  def __call__(self, input_x) -> Union[np.array, jnp.ndarray]:
    """Applies the PositionalEmbeddings.

    Args:
      input_x: input data

    Returns:
      output: an input array with added poisitonal embeddings
    """
    positional_embedding = self.positional_encoding_support_function(
        self.max_len, input_x.shape[-1])
    pe = positional_embedding[:, :input_x.shape[1], :]
    return input_x + pe


class RoutingTransformerArchitecture(nn.Module):
  """An end-to-end transformer architecture.

  Attributes:
    vocab_size: vocabulary size
    emb_dim: embeddings dimension
    num_heads: number of attention heads
    num_layers: number of TransformerBlocks
    qkv_dim: dimension of the query/key/value
    mlp_dim: number of neurons in the 1st Dense layer in the feedforward block
    max_len: maximum possible length for the input
    dropout_rate: dropout rate in the Dropout layers in the feedforward block
    attention_dropout_rate: dropout rate in the Dropout layers in the attention
      block
    attention_function: attention function to use in the attention block
    dtype: data type to be used in the model training
    pre_normalization: boolean if the data should be normalized before feeding
      it to the attention block and to the feedforward block
    post_normalization: boolean if the data should be normalized after
      processing it by the attention block and to the feedforward block
  """
  vocab_size: int
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_function: Any = None
  dtype: str = None
  pre_normalization: bool = True
  post_normalization: bool = True

  def shift_right_support_function(
      self, input_x: Union[np.array,
                           jnp.ndarray]) -> Union[np.array, jnp.ndarray]:
    """Shift the input to the right by padding and slicing on axis.

    Args:
      input_x: input data

    Returns:
      output: an input data sifted to the right
    """
    pad_widths = [(0, 0)] * len(input_x.shape)
    pad_widths[1] = (1, 0)
    padded = jnp.pad(
        input_x,
        pad_widths,
        mode='constant',
        constant_values=input_x.dtype.type(0))
    return lax.dynamic_slice_in_dim(padded, 0, padded.shape[1] - 1, 1)

  @nn.compact
  def __call__(self,
               input_x: Union[np.array, jnp.ndarray],
               train: bool = False) -> Union[np.array, jnp.ndarray]:
    """Applies the RoutingTransformerArchitecture.

    Args:
      input_x: input data
      train: boolean indicating whether training or not

    Returns:
      output: an array with logit probabilities per word
    """
    dtype = utils.dtype_from_str(self.dtype)

    mask = nn.combine_masks(
        nn.make_attention_mask(input_x > 0, input_x > 0, dtype=dtype),
        nn.make_causal_mask(input_x, dtype=dtype))

    y = input_x.astype('int32')
    y = self.shift_right_support_function(y)
    output_emebeddings = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))
    y = output_emebeddings(y)
    y = PositionalEmbeddings(max_len=self.max_len)(y)
    y = nn.Dropout(
        rate=self.dropout_rate,
        deterministic=not train)(y).astype(dtype)

    y = y.astype(dtype)
    for _ in range(self.num_layers):
      y = TransformerBlock(
          num_heads=self.num_heads,
          qkv_dim=self.qkv_dim,
          attention_dropout_rate=self.attention_dropout_rate,
          attention_function=self.attention_function,
          dropout_rate=self.dropout_rate,
          mlp_dim=self.mlp_dim,
          dtype=dtype,
          pre_normalization=self.pre_normalization,
          post_normalization=self.post_normalization)(
              y, mask=mask, train=train)

    y = nn.LayerNorm(dtype=dtype)(y)

    logits = nn.Dense(
        self.vocab_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        dtype=dtype)(
            y)
    return logits.astype(dtype)


class RoutingTransformer(base_model.BaseModel):

  def build_flax_module(self):
    max_len = max(self.hps.max_target_length, self.hps.max_eval_target_length)
    return RoutingTransformerArchitecture(
        vocab_size=self.hps['output_shape'][-1],
        emb_dim=self.hps.emb_dim,
        num_heads=self.hps.num_heads,
        num_layers=self.hps.num_layers,
        qkv_dim=self.hps.qkv_dim,
        mlp_dim=self.hps.mlp_dim,
        max_len=max_len,
        dropout_rate=self.hps.dropout_rate,
        attention_dropout_rate=self.hps.attention_dropout_rate,
        attention_function=self.hps.attention_function,
        dtype=self.hps.model_dtype,
        pre_normalization=self.hps.pre_normalization,
        post_normalization=self.hps.post_normalization,
    )
