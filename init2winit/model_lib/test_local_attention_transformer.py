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

"""Tests for local atention transformer.
"""

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.model_lib import local_attention_transformer
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class LocalAttentionTransformerTests(parameterized.TestCase):
  """Tests support functions in local_attention_transformer.py.

  Reference shapes and data types come from Colab notebooks,
  where TF and JAX/FLAX support functions where compared.

  """

  def test_decode_step_to_index(self):
    """Tests support function decode_step_to_index."""
    decode_step = 10
    array_shape = (8192,)
    output = local_attention_transformer.decode_step_to_index(
        decode_step, array_shape)

    self.assertEqual(output, (10,))

  def test_get_item_at_decode_step(self):
    """Tests support function get_item_at_decode_step."""
    x = np.array(np.random.rand(1, 256, 16), dtype=np.float32)
    decode_step = 10
    output = local_attention_transformer.get_item_at_decode_step(x, decode_step)

    self.assertEqual(output.shape, (1, 1, 16))
    self.assertEqual(output.dtype, np.float32)

  def test_embedding_to_padding(self):
    """Tests support function embedding_to_padding."""
    x = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    output = local_attention_transformer.embedding_to_padding(x)

    self.assertEqual(output.shape, (8, 1, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_ones_matrix_band_part(self):
    """Tests support function ones_matrix_band_part."""
    num_rows = 2
    num_cols = 2
    max_backward = 10
    max_forward = 10
    output_shape = [1, 1, 2, 2]
    output = local_attention_transformer.ones_matrix_band_part(
        num_rows, num_cols, max_backward, max_forward, output_shape)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_bias_local(self):
    """Tests support function attention_bias_local."""
    length = 2
    max_backward = 10
    max_forward = 10
    output = local_attention_transformer.attention_bias_local(
        length, max_backward, max_forward)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_bias_lower_triangle(self):
    """Tests support function attention_bias_lower_triangle."""
    length = 2
    output = local_attention_transformer.attention_bias_lower_triangle(length)

    self.assertEqual(output.shape, (1, 1, 2, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_causal_attention_bias_nd(self):
    """Tests support function causal_attention_bias_nd."""
    output = local_attention_transformer.causal_attention_bias_nd()

    self.assertEqual(output.shape, (1, 1, 256, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_maybe_tile(self):
    """Tests support function maybe_tile."""
    x = np.array(np.random.rand(1, 2, 2), dtype=np.float32)
    y = np.array(np.random.rand(1, 4, 3), dtype=np.float32)
    output = local_attention_transformer.maybe_tile(x, y)

    self.assertEqual(output[0].shape, (1, 4, 2))
    self.assertEqual(output[1].shape, (1, 4, 3))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_local_attention_bias_nd(self):
    """Tests support function local_attention_bias_nd."""
    x = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    output = local_attention_transformer.local_attention_bias_nd(x)

    self.assertEqual(output.shape, (8, 1, 256, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_pad_to_multiple_nd(self):
    """Tests support function pad_to_multiple_nd."""
    x = np.array(np.random.rand(1, 4, 4), dtype=np.float32)
    block_shape = (2, 10)
    output = local_attention_transformer.pad_to_multiple_nd(x, block_shape)

    self.assertEqual(output.shape, (1, 4, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_select_block_for_decode_step(self):
    """Tests support function select_block_for_decode_step."""
    x = np.array(np.random.rand(1, 10, 256, 16), dtype=np.float32)
    decode_step = 10
    output = local_attention_transformer.select_block_for_decode_step(
        x, decode_step)

    self.assertEqual(output.shape, (1, 1, 256, 16))
    self.assertEqual(output.dtype, np.float32)

  def test_break_into_blocks_nd(self):
    """Tests support function break_into_blocks_nd."""
    x = np.array(np.random.rand(9, 9, 9, 9), dtype=np.float32)
    block_shape = (3,)
    output = local_attention_transformer.break_into_blocks_nd(x, block_shape)

    self.assertEqual(output.shape, (81, 3, 3, 9))
    self.assertEqual(output.dtype, np.float32)

  def test_break_into_memory_blocks_nd(self):
    """Tests support function break_into_memory_blocks_nd."""
    x = np.array(np.random.rand(1, 256, 1), dtype=np.float32)
    output = local_attention_transformer.break_into_memory_blocks_nd(x)

    self.assertEqual(output.shape, (1, 1, 512, 1))
    self.assertEqual(output.dtype, np.float32)

  def test_flatten_blocks_nd(self):
    """Tests support function flatten_blocks_nd."""
    x = np.array(np.random.rand(1, 1, 3, 4), dtype=np.float32)
    output = local_attention_transformer.flatten_blocks_nd(x)

    self.assertEqual(output.shape, (1, 1, 3, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_unflatten_blocks_nd(self):
    """Tests support function unflatten_blocks_nd."""
    x = np.array(np.random.rand(1, 2, 3, 4), dtype=np.float32)
    blocks_per_dimension = [2]
    output = local_attention_transformer.unflatten_blocks_nd(
        x, blocks_per_dimension)

    self.assertEqual(output.shape, (1, 2, 3, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_break_bias_into_blocks(self):
    """Tests support function break_bias_into_blocks."""
    x = np.array(np.random.rand(2, 10), dtype=np.float32)
    output = local_attention_transformer.break_bias_into_blocks(x)

    self.assertEqual(output.shape, (16, 1, 512))
    self.assertEqual(output.dtype, np.float32)

  def test_cast_like(self):
    """Tests support function cast_like."""
    x = np.array(np.random.rand(1, 2, 10), dtype=np.float32)
    y = np.array(np.random.rand(1, 2, 10), dtype=np.int32)
    output = local_attention_transformer.cast_like(x, y)

    self.assertEqual(output.shape, (1, 2, 10))
    self.assertEqual(output.dtype, np.int32)

  def test_generate_relative_positions_matrix(self):
    """Tests support function generate_relative_positions_matrix."""
    length_q = 10
    length_k = 10
    output = local_attention_transformer.generate_relative_positions_matrix(
        length_q, length_k)

    self.assertEqual(output.shape, (10, 10))
    self.assertEqual(output.dtype, np.int32)

  def test_generate_relative_positions_embeddings(self):
    """Tests support function generate_relative_positions_embeddings."""
    length_q = 10
    length_k = 10
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.RelativePositionEmbeddings(
        embed_layer_name='unit_test')
    params = model.init(rng_key, length_q, length_k)
    output = model.apply(params, length_q, length_k)

    self.assertEqual(output.shape, (10, 10, 129))
    self.assertEqual(output.dtype, np.float32)

  def test_relative_attention_inner(self):
    """Tests support function relative_attention_inner."""
    x = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)
    y = np.array(np.random.rand(1, 1, 4, 1), dtype=np.float32)
    z = np.array(np.random.rand(1, 4, 3), dtype=np.float32)
    transpose = False
    output = local_attention_transformer.relative_attention_inner(
        x, y, z, transpose)

    self.assertEqual(output.shape, (1, 1, 1, 3))
    self.assertEqual(output.dtype, np.float32)

  def test_dot_product_attention_relative(self):
    """Tests support function dot_product_attention_relative."""
    q = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)
    k = np.array(np.random.rand(1, 1, 4, 4), dtype=np.float32)
    v = np.array(np.random.rand(1, 1, 4, 4), dtype=np.float32)
    b = np.array(np.random.rand(1, 1, 1, 4), dtype=np.float32)

    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.RelativeDotProductAttention()
    params = model.init(rng_key, q, k, v, b)
    output = model.apply(params, q, k, v, b)

    self.assertEqual(output[0].shape, (1, 1, 1, 4))
    self.assertEqual(output[1].shape, (1, 1, 1, 4))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_dot_product_attention(self):
    """Tests support function dot_product_attention."""
    q = np.array(np.random.rand(8, 1, 256, 1024), dtype=np.float32)
    k = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    v = np.array(np.random.rand(8, 1, 512, 1024), dtype=np.float32)
    b = np.array(np.random.rand(1, 1, 256, 512), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.DotProductAttention()
    params = model.init(rng_key, q, k, v, b)
    output = model.apply(params, q, k, v, b)

    self.assertEqual(output[0].shape, (8, 1, 256, 1024))
    self.assertEqual(output[1].shape, (8, 1, 256, 512))
    self.assertEqual(output[0].dtype, np.float32)
    self.assertEqual(output[1].dtype, np.float32)

  def test_combine_heads_nd(self):
    """Tests support function combine_heads_nd."""
    x = np.array(np.random.rand(1, 2, 3), dtype=np.float32)
    output = local_attention_transformer.combine_heads_nd(x)

    self.assertEqual(output.shape, (1, 6))
    self.assertEqual(output.dtype, np.float32)

  def test_put_back_blocks_nd(self):
    """Tests support function put_back_blocks_nd."""
    x = np.array(np.random.rand(1, 2, 3, 5), dtype=np.float32)
    block_shape = (3,)
    output = local_attention_transformer.put_back_blocks_nd(x, block_shape)

    self.assertEqual(output.shape, (1, 6, 5))
    self.assertEqual(output.dtype, np.float32)

  def test_attention_nd(self):
    """Tests support function attention_nd."""
    q = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    k = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    v = np.array(np.random.rand(1, 8, 256, 1024), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.LocalAttention()
    params = model.init(rng_key, q, k, v)
    output = model.apply(params, q, k, v)

    self.assertEqual(output.shape, (1, 8, 256, 1024))
    self.assertEqual(output.dtype, np.float32)

  def test_split_heads_nd(self):
    """Tests support function split_heads_nd."""
    x = np.array(np.random.rand(8, 8), dtype=np.float32)
    output = local_attention_transformer.split_heads_nd(x)

    self.assertEqual(output.shape, (8, 8, 1))
    self.assertEqual(output.dtype, np.float32)

  def test_put_item_in_decode_step(self):
    """Tests support function put_item_in_decode_step."""
    x = jnp.array(np.random.rand(3, 8, 1, 4), dtype=np.float32)
    output = local_attention_transformer.put_item_in_decode_step(
        input_x=x, decode_step=1)

    self.assertEqual(output.shape, (3, 8, 1, 4))
    self.assertEqual(output.dtype, np.float32)

  def test_multihead_attention_nd(self):
    """Tests support function multihead_attention_nd."""
    x = np.array(np.random.rand(1, 2, 3), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.MultiHeadAttention()
    params = model.init(rng_key, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 2, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_get_timing_signal_1d(self):
    """Tests support function get_timing_signal_1d."""
    length = 10
    channels = 2
    output = local_attention_transformer.get_timing_signal_1d(length, channels)

    self.assertEqual(output.shape, (1, 10, 2))
    self.assertEqual(output.dtype, np.float32)

  def test_process_input(self):
    """Tests support function process_input."""
    x = jnp.array(np.random.rand(1, 8192), dtype=jnp.int32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.ProcessInput()
    params = model.init(rng_key, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 8192, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_process_partial_targets_decoding(self):
    """Tests support function process_partial_targets_decoding."""
    x = np.array(np.random.rand(1, 256), dtype=np.float32)
    output = local_attention_transformer.process_partial_targets_decoding(
        targets=x)

    self.assertEqual(output.shape, (1, 256))
    self.assertEqual(output.dtype, np.float32)

  def test_feedforward(self):
    """Tests support function feedforward."""
    x = np.array(np.random.rand(1, 256), dtype=np.float32)
    feedforward_depths = [4096, 1032]

    key1 = jax.random.PRNGKey(0)
    model = local_attention_transformer.FeedForward(
        feedforward_depths=feedforward_depths)
    params = model.init(key1, x)
    output = model.apply(params, x)

    self.assertEqual(output.shape, (1, 1032))
    self.assertEqual(output.dtype, np.float32)

  def test_layer_postprocess(self):
    """Tests support function layer_postprocess."""
    x = np.array(np.random.rand(2, 3), dtype=np.float32)
    y = np.array(np.random.rand(2, 3), dtype=np.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.LayerPostProcess()
    params = model.init(rng_key, x, y)
    output = model.apply(params, x, y)

    self.assertEqual(output.shape, (2, 3))
    self.assertEqual(output.dtype, np.float32)

  def test_decoder_block(self):
    """Tests support function decoder_block."""
    feedforward_depths = [4096, 1032]
    x = jnp.array(np.random.rand(1, 2, 1032), dtype=jnp.float32)
    rng_key = jax.random.PRNGKey(0)
    model = local_attention_transformer.DecoderBlock(
        feedforward_depths=feedforward_depths)
    params = model.init(rng_key, x)
    imp_output = model.apply(params, x)

    self.assertEqual(imp_output.shape, (1, 2, 1032))
    self.assertEqual(imp_output.dtype, np.float32)


if __name__ == '__main__':
  absltest.main()

