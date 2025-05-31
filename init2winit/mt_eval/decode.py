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

"""Fast decoding routines for inference from a trained model.

Temperature sampling and beam search routines.
"""

import functools
import typing
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import flax
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# Constants
# We assume the default End-of-Sentence token is 2 (SentencePiece).
EOS_ID = 2
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = np.array(-1.0e7)
# Temperatures lower than this are considered 0.0, which is handled specially
# with a conditional. This is to avoid numeric issues from exponentiating on
# 1.0/temperature when temperature is close to 0.0.
MIN_TEMPERATURE = np.array(1e-4)

# Beam Search
#
# We try to match the logic of the original t2t implementation and the mlperf
# reference tensorflow implementation at:
# https://github.com/mlperf/training/blob/master/translation/tensorflow/transformer/model/beam_search.py
#
# Using JAX we are directly programming with the XLA computation model,
# so here we initialize and update static-sized arrays inside an XLA while loop
# rather than concatenating onto a growing chain of sequences.


def brevity_penalty(alpha, length):
  """Brevity penalty function for beam search penalizing short sequences.

  Args:
    alpha: float: brevity-penalty scaling parameter.
    length: int: length of considered sequence.

  Returns:
    Brevity penalty score as jax scalar.
  """
  return jnp.power(((5.0 + length) / 6.0), alpha)


# Beam handling utility functions:


def is_scalar(x, offset: int) -> bool:
  """Checks if x is scalar."""
  # The `classical` scalar
  if x.ndim == 0:
    return True
  # The scalar in the case of scan over layers:
  if x.ndim < offset + 1:
    return True
  return False


def add_beam_dim(x, beam_size: int, offset: int = 0):
  """Creates new beam dimension in non-scalar array and tiles into it."""
  if is_scalar(x, offset):
    return x
  x = jnp.expand_dims(x, axis=offset + 1)
  tile_dims = [1] * x.ndim
  tile_dims[offset + 1] = beam_size
  return jnp.tile(x, tile_dims)


def flatten_beam_dim(x, offset: int = 0):
  """Flattens the first two dimensions of a non-scalar array."""
  if is_scalar(x, offset):
    return x
  xshape = list(x.shape)
  b_sz = xshape.pop(offset)
  xshape[offset] *= b_sz
  return x.reshape(xshape)


def unflatten_beam_dim(x, batch_size, beam_size, offset: int = 0):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  if is_scalar(x, offset):
    return x
  assert batch_size * beam_size == x.shape[offset]
  xshape = list(x.shape)
  newshape = xshape[:offset] + [batch_size, beam_size] + xshape[offset + 1:]
  return x.reshape(newshape)


def flat_batch_beam_expand(x, beam_size, offset: int = 0):
  """Expands the each batch item by beam_size in batch_dimension."""
  return flatten_beam_dim(add_beam_dim(x, beam_size, offset), offset)


def gather_beams(nested, beam_indices, batch_size, new_beam_size,
                 offset: int = 0):
  """Gathers the beam slices indexed by beam_indices into new beam array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    beam_indices: array of beam_indices
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ beam dimension.
    offset: int : cache axis from scan over layers.

  Returns:
    New pytree with new beam arrays.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  batch_indices = jnp.reshape(
      jnp.arange(batch_size * new_beam_size) // new_beam_size,
      (batch_size, new_beam_size))
  assert offset < 4, 'scan_over_layers_offset >= 4 is not supported'
  def gather_fn(x):
    if is_scalar(x, offset):
      return x
    # Unfortunately an elegant indexing with arbitrary number of :, :,...
    # prepended is not available.
    elif offset == 0:
      return x[batch_indices, beam_indices]
    elif offset == 1:
      return x[:, batch_indices, beam_indices]
    elif offset == 2:
      return x[:, :, batch_indices, beam_indices]
    else:
      return x[:, :, :, batch_indices, beam_indices]
  return jax.tree.map(gather_fn, nested)


def gather_topk_beams(nested, score_or_log_prob, batch_size, new_beam_size):
  """Gathers the top-k beam slices given by score_or_log_prob array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    score_or_log_prob: [batch_size, old_beam_size] array of values to sort by
      for top-k selection of beam slices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ top-k selected beam dimension

  Returns:
    New pytree with new beam arrays containing top k new_beam_size slices.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  _, topk_indices = lax.top_k(score_or_log_prob, k=new_beam_size)
  topk_indices = jnp.flip(topk_indices, axis=1)
  return gather_beams(nested, topk_indices, batch_size, new_beam_size)


# Beam search state:
@flax.struct.dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: jax.Array  # float32: [batch_size, beam_size]
  finished_scores: jax.Array  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: jax.Array  # int32: [batch_size, beam_size, max_decode_len]
  finished_seqs: jax.Array  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jax.Array  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  cache: typing.Any  # Any pytree of arrays, e.g. flax attention Cache object


# Sampling State
@flax.struct.dataclass
class SamplingState:
  """Holds sampling state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence probabilities and finished sequence scores.
  all_log_probs: jax.Array  # float32: [batch_size, sample_size]
  # All sequences we are sampling.
  all_seqs: jax.Array  # int32: [batch_size, sample_size, max_decode_len]
  # Sequences that are finished.
  finished_seqs: jax.Array  # int32: [batch_size, sample_size,
  #                                         max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jax.Array  # bool: [batch_size, sample_size]
  cache: Any  # Any pytree of arrays, e.g. flax attention Cache object


def beam_init(batch_size,
              beam_size,
              max_decode_len,
              cache,
              offset: int = 0):
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(0)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (beam_size - 1)),
      [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * NEG_INF
  live_seqs0 = jnp.zeros(
      (batch_size, beam_size, max_decode_len), jnp.int32)
  finished_seqs0 = jnp.zeros(
      (batch_size, beam_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  beam_cache0 = jax.tree.map(lambda x: add_beam_dim(x, beam_size, offset),
                             cache)
  return BeamState(cur_index=cur_index0,
                   live_logprobs=live_logprobs0,
                   finished_scores=finished_scores0,
                   live_seqs=live_seqs0,
                   finished_seqs=finished_seqs0,
                   finished_flags=finished_flags0,
                   cache=beam_cache0)


def sampling_init(batch_size: int,
                  sample_size: int,
                  max_decode_len: int,
                  cache):
  """Initializes the sampling state data structure."""
  cur_index0 = jnp.array(0)
  all_log_probs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (sample_size - 1)), [batch_size, 1])
  all_seqs0 = jnp.zeros((batch_size, sample_size, max_decode_len), jnp.int32)
  finished_seqs0 = jnp.zeros((batch_size, sample_size, max_decode_len),
                             jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, sample_size), jnp.bool_)
  # add sample dimension to attention cache pytree elements
  sample_cache0 = jax.tree.map(lambda x: add_beam_dim(x, sample_size), cache)

  return SamplingState(
      cur_index=cur_index0,
      all_log_probs=all_log_probs0,
      all_seqs=all_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      cache=sample_cache0)


# Beam search routine:
def beam_search(inputs,
                cache,
                tokens_to_logits,
                beam_size=4,
                alpha=0.6,
                eos_id=EOS_ID,
                max_decode_len=None,
                offset: int = 0):
  """Beam search for transformer machine translation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    beam_size: int: number of beams to use in beam search.
    alpha: float: scaling factor for brevity penalty.
    eos_id: int: end-of-sentence token for target vocabulary.
    max_decode_len: int: maximum length of decoded translations.
    offset: int: used by scan over layers

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  # We liberally annotate shape information for clarity below.

  batch_size = inputs.shape[0]
  if max_decode_len is None:
    max_decode_len = inputs.shape[1]
  end_marker = jnp.array(eos_id)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size,
                                     beam_size,
                                     max_decode_len,
                                     cache, offset)

  def beam_search_loop_cond_fn(state):
    """Beam search loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = (state.cur_index < max_decode_len - 1)

    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_decode_len)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores = jnp.min(
        state.finished_scores, axis=1, keepdims=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = jnp.where(
        state.finished_flags, worst_finished_scores, NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = jnp.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)

  def beam_search_loop_body_fn(state):
    """Beam search loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the beam dimension into batch
    # dimension for feeding into the model.
    # --> [batch * beam, 1]
    flat_ids = flatten_beam_dim(
        lax.dynamic_slice(state.live_seqs, (0, 0, state.cur_index),
                          (batch_size, beam_size, 1)))
    # Flatten beam dimension into batch to be compatible with model.
    # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
    flat_cache = jax.tree.map(functools.partial(flatten_beam_dim,
                                                offset=offset), state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * beam, vocab]
    flat_logits, new_flat_cache = tokens_to_logits(flat_ids, flat_cache)
    # Tokens to logits
    # unflatten beam dimension
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    # Unflatten beam dimension in attention cache arrays
    # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}

    def unflatten_beam_dim_in_cache(x):
      return unflatten_beam_dim(x, batch_size, beam_size, offset=offset)

    new_cache = jax.tree.map(unflatten_beam_dim_in_cache, new_flat_cache)

    # Gather log probabilities from logits
    candidate_log_probs = jax.nn.log_softmax(logits)
    # Add new logprobs to existing prefix logprobs.
    # --> [batch, beam, vocab]
    log_probs = (
        candidate_log_probs + jnp.expand_dims(state.live_logprobs, axis=2))

    # We'll need the vocab size, gather it from the log probability dimension.
    vocab_size = log_probs.shape[2]

    # Each item in batch has beam_size * vocab_size candidate sequences.
    # For each item, get the top 2*k candidates with the highest log-
    # probabilities. We gather the top 2*K beams here so that even if the best
    # K sequences reach EOS simultaneously, we have another K sequences
    # remaining to continue the live beam search.
    beams_to_keep = 2 * beam_size
    # Flatten beam and vocab dimensions.
    flat_log_probs = log_probs.reshape((batch_size, beam_size * vocab_size))
    # Gather the top 2*K scores from _all_ beams.
    # --> [batch, 2*beams], [batch, 2*beams]
    topk_log_probs, topk_indices = lax.top_k(flat_log_probs, k=beams_to_keep)
    # Recover the beam index by floor division.
    topk_beam_indices = topk_indices // vocab_size
    # Gather 2*k top beams and beam-associated caches.
    # --> [batch, 2*beams, length], {[batch, 2*beams, ...], ...}
    topk_seq = gather_beams(state.live_seqs, topk_beam_indices, batch_size,
                            beams_to_keep)

    # Append the most probable 2*K token IDs to the top 2*K sequences
    # Recover token id by modulo division and expand Id array for broadcasting.
    # --> [batch, 2*beams, 1]
    topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(topk_seq, topk_ids,
                                        (0, 0, state.cur_index + 1))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index + 1] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # Gather top-k beams.
    _, new_topk_indices = lax.top_k(new_log_probs, k=beam_size)
    # --> [batch, beams, length], [batch, beams], {[batch, beams, ...], ...}
    top_alive_seq, top_alive_log_probs = gather_beams([topk_seq, new_log_probs],
                                                      new_topk_indices,
                                                      batch_size, beam_size)

    top_alive_indices = gather_beams(topk_beam_indices, new_topk_indices,
                                     batch_size, beam_size)
    # Apply offset to the cache
    top_alive_cache = gather_beams(new_cache, top_alive_indices, batch_size,
                                   beam_size, offset=offset)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index + 1)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq],
        axis=1)
    finished_scores = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], axis=1)
    finished_flags = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], axis=1)
    # --> [batch, beams, length], [batch, beams], [batch, beams]
    top_finished_seq, top_finished_scores, top_finished_flags = (
        gather_topk_beams([finished_seqs, finished_scores, finished_flags],
                          finished_scores, batch_size, beam_size))

    return BeamState(
        cur_index=state.cur_index + 1,
        live_logprobs=top_alive_log_probs,
        finished_scores=top_finished_scores,
        live_seqs=top_alive_seq,
        finished_seqs=top_finished_seq,
        finished_flags=top_finished_flags,
        cache=top_alive_cache)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn,
                               beam_search_loop_body_fn, beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(none_finished[:, None, None],
                            final_state.finished_seqs, final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(none_finished[:,
                                            None], final_state.finished_scores,
                              final_state.live_logprobs)

  return finished_seqs, finished_scores


def sampling(inputs: jax.Array,
             cache: Any,
             tokens_to_logits: Callable[..., Tuple[jax.Array, Any]],
             rng: Union[jnp.ndarray, np.ndarray, int],
             sample_size: int,
             eos_id: int,
             max_decode_len: Optional[int],
             temperature: Optional[int],
             rescale_log_probs: Optional[int]):
  """Sampling for transformer machine translation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    rng: random generator keys for sampling.
    sample_size: int: number of samples.
    eos_id: int: id of end-of-sentence token for target vocabulary.
    max_decode_len: int: maximum length of decoded translations.
    temperature: float: sampling temperature. temp ~ 0.0 approaches greedy
    sampling. temp = 1.0 means no-op. temp >> 1 approaches uniform sampling.
    rescale_log_probs: bool: whether to apply temperature, topp, and topk
      rescaling to the log probs which are returned. If True, the log_probs will
      include these transformations (for example, with topk=1, all log_probs
      will be identically 0.0). If False, the log_probs will not be affected,
      and topk/topp/temperature will not affect sequence probabilities.

  Returns:
     Tuple of:
       [batch_size, sample_size, max_decode_len] sampled sequences.
  """
  # We liberally annotate shape information for clarity below.

  batch_size = inputs.shape[0]
  if max_decode_len is None:
    max_decode_len = inputs.shape[1]
  end_marker = jnp.array(eos_id)

  # initialize sampling search state
  sampling_init_state = sampling_init(batch_size, sample_size, max_decode_len,
                                      cache)

  def sampling_loop_cond_fn(state: SamplingState):
    """Sampling loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = (state.cur_index < max_decode_len - 1)
    return not_at_end

  def sampling_loop_body_fn(state: SamplingState):
    """Sampling loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the sample dimension into batch
    # dimension for feeding into the model.
    # --> [batch * sample, 1]
    # Using beam flattening function.
    flat_ids = flatten_beam_dim(
        lax.dynamic_slice(state.all_seqs, (0, 0, state.cur_index),
                          (batch_size, sample_size, 1)))
    # Flatten sample dimension into batch to be compatible with model.
    # {[batch, sample, ...], ...} --> {[batch * sample, ...], ...}
    flat_cache = jax.tree.map(flatten_beam_dim, state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * sample, vocab]
    flat_logits, new_flat_cache = tokens_to_logits(flat_ids, flat_cache)

    # Unflatten sample dimension in attention cache arrays
    # {[batch * sample, ...], ...} --> {[batch, sample, ...], ...}
    new_cache = jax.tree.map(
        lambda x: unflatten_beam_dim(x, batch_size, sample_size),
        new_flat_cache)

    def sample_logits_with_nonzero_temperature(flat_logits_to_sample):
      # TODO(ankugarg): Implement top-p and top-k aka nucleus sampling here.
      scaled_logits = flat_logits_to_sample / jnp.maximum(
          temperature, MIN_TEMPERATURE)
      # rngs = jax.random.split(rng, batch_size * sample_size + 1)
      sampled_ids = jax.random.categorical(rng, scaled_logits).astype(jnp.int32)
      if rescale_log_probs:
        log_probs = jax.nn.log_softmax(scaled_logits)
      else:
        log_probs = jax.nn.log_softmax(flat_logits_to_sample)
      # [batch * sample size, vocab] -> [batch * sample_size]
      sampled_log_probs = jnp.squeeze(
          jnp.take_along_axis(
              log_probs, jnp.expand_dims(sampled_ids, axis=1), axis=-1),
          axis=-1)
      return (sampled_ids, sampled_log_probs)

    def sample_logits_with_zero_temperature(flat_logits_to_sample):
      # For zero temperature, we always want the greedy output, regardless
      # of the fact we have top-k or top-p (nucleus sampling) values.
      sampled_ids = jnp.argmax(flat_logits_to_sample, -1).astype(jnp.int32)
      if rescale_log_probs:
        sampled_log_probs = jnp.zeros_like(sampled_ids, dtype=jnp.float32)
      else:
        log_probs = jax.nn.log_softmax(flat_logits_to_sample)
        # [batch * sample size, vocab] -> [batch * sample_size]
        sampled_log_probs = jnp.squeeze(
            jnp.take_along_axis(
                log_probs, jnp.expand_dims(sampled_ids, axis=1), axis=-1),
            axis=-1)
      return (sampled_ids, sampled_log_probs)

    (sampled_ids,
     sampled_log_probs) = lax.cond(temperature > MIN_TEMPERATURE,
                                   sample_logits_with_nonzero_temperature,
                                   sample_logits_with_zero_temperature,
                                   flat_logits)

    # Reshape.
    sampled_ids = sampled_ids.reshape(batch_size, sample_size)
    sampled_ids = jnp.expand_dims(sampled_ids, axis=2)
    sampled_log_probs = sampled_log_probs.reshape(batch_size, sample_size)
    # Add new log probabilities to the previous ones.
    new_log_probs = sampled_log_probs + state.all_log_probs
    # Need to get the new sequences now
    # [batch, sample, length]
    updated_seq = lax.dynamic_update_slice(state.all_seqs, sampled_ids,
                                           (0, 0, state.cur_index + 1))

    # Did any of these sequences reach an end marker? --> [batch, sample]
    newly_finished_flag = (updated_seq[:, :, state.cur_index + 1] == end_marker)

    # Make sure not to add already finished sequence again to the list of
    # finished sequences.
    newly_finished_flag = jnp.where(
        state.finished_flags, jnp.zeros_like(newly_finished_flag, jnp.bool_),
        newly_finished_flag)

    # new set of finished sequences
    finished_seqs = state.finished_seqs
    finished_seqs = jnp.where(newly_finished_flag[:, :, None], updated_seq,
                              finished_seqs)

    # create final finished flags that combine sequences finished on this
    # iteration with sequences finished in the past.
    finished_flags = jnp.where(newly_finished_flag, newly_finished_flag,
                               state.finished_flags)

    return SamplingState(
        cur_index=state.cur_index + 1,
        all_log_probs=new_log_probs,
        all_seqs=updated_seq,
        finished_seqs=finished_seqs,
        finished_flags=finished_flags,
        cache=new_cache)

  # Run while loop and get final sampling result.
  final_state = lax.while_loop(sampling_loop_cond_fn, sampling_loop_body_fn,
                               sampling_init_state)

  finished_seqs = jnp.where(final_state.finished_flags[:, :, None],
                            final_state.finished_seqs, final_state.all_seqs)
  # Ignore the first token in each sequence.
  return finished_seqs[:, :, 1:]


def decode_step(batch,
                params,
                cache,
                max_decode_len,
                flax_module,
                eos_id=EOS_ID,
                beam_size=4,
                offset: int = 0):
  """Predict translation with fast decoding beam search on a batch."""
  inputs = batch['inputs']

  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = flax_module.apply(
      {'params': params},
      inputs,
      train=False,
      method=flax_module.encode)
  # Inputs don't need an offset in case of scan over layers.
  encoded_inputs = flat_batch_beam_expand(encoded_inputs, beam_size, offset=0)
  raw_inputs = flat_batch_beam_expand(inputs, beam_size, offset=0)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = flax_module.apply(
        {'params': params, 'cache': flat_cache},
        encoded=encoded_inputs,
        # Tile the inputs so that they have the same leading dim as flat_ids so
        # that the masking that is calculated from them is correctly shaped.
        inputs=raw_inputs,
        targets=flat_ids,
        train=False,
        mutable=['cache'],
        method=flax_module.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_id=eos_id,
      max_decode_len=max_decode_len,
      offset=offset)

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  return beam_seqs[:, -1, 1:]


def sampling_step(batch: np.ndarray,
                  params: Mapping[str, Any],
                  cache: Mapping[str, Any],
                  max_decode_len: int,
                  rng: Union[jnp.ndarray, np.ndarray, int],
                  flax_module,
                  eos_id=EOS_ID,
                  sample_size: int = 20,
                  temperature: int = 1.0,
                  rescale_log_probs: int = 1,
                  ):
  """Performs one step of sampling."""
  inputs = batch['inputs']
  # Prepare transformer fast-decoder call for sampling: for sampling we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * sample_size, where each batch item's data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> sample_size=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = flax_module.apply({'params': params},
                                     inputs,
                                     train=False,
                                     method=flax_module.encode)
  encoded_inputs = flat_batch_beam_expand(encoded_inputs, sample_size)
  raw_inputs = flat_batch_beam_expand(inputs, sample_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch_size * sample_size, 1, vocab]
    flat_logits, new_vars = flax_module.apply(
        {'params': params, 'cache': flat_cache},
        encoded=encoded_inputs,
        # Tile the inputs so that they have the same leading dim as flat_ids so
        # that the masking that is calculated from them is correctly shaped.
        inputs=raw_inputs,
        targets=flat_ids,
        train=False,
        mutable=['cache'],
        method=flax_module.decode)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch_size * sample_size, 1, vocab] --> [batch_size * sample_size, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # sampling of possible sequences given input encoding.
  sampling_seqs = sampling(
      inputs,
      cache,
      tokens_ids_to_logits,
      rng,
      sample_size=sample_size,
      eos_id=eos_id,
      max_decode_len=max_decode_len,
      temperature=temperature,
      rescale_log_probs=rescale_log_probs)

  # Sampling returns [batch_size, sampling_size, max_predict_length -1].
  return sampling_seqs

