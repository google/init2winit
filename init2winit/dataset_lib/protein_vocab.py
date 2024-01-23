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

"""Classes specifying different protein domains."""
import collections
from collections import abc
import copy

import numpy as np
from six.moves import range

# For the sake of backwards compatibility, we don't use the ordering of amino
_AA_TOKENS = 'ACDEFGHIKLMNPQRSTVWY'
_ANOMALOUS_AA_TOKENS = 'BOUXZ'
# Alignment tokens.
GAP = '-'  # Default gap character.
INSERT_GAP = '.'  # Gap character in insert states of an MSA.
ALIGN_TOKENS = (INSERT_GAP, GAP)

BOS_TOKEN = '<'  # Beginning of sequence token.
EOS_TOKEN = '>'  # End of sequence token.
PAD_TOKEN = '_'  # Padding token.
MASK_TOKEN = '*'  # Mask token.
SEP_TOKEN = '|'  # A special token for separating tokens for serialization.
_DEFAULT_VOCAB_NAME = 'vocab'


class Vocabulary(object):
  """Basic vocabulary used to represent output tokens for domains."""

  def __init__(self,
               tokens,
               name=None,
               include_bos=False,
               include_eos=False,
               include_pad=False,
               include_mask=False,
               bos_token=BOS_TOKEN,
               eos_token=EOS_TOKEN,
               pad_token=PAD_TOKEN,
               mask_token=MASK_TOKEN,
               disallow_sep_token=True):
    """A token vocabulary.

    Args:
      tokens: An list of tokens to put in the vocab. If an int, will be
        interpreted as the number of tokens and '0', ..., 'tokens-1' will be
        used as tokens.
      name: An optional name for the vocab.
      include_bos: Whether to append `bos_token` to `tokens` that marks the
        beginning of a sequence.
      include_eos: Whether to append `eos_token` to `tokens` that marks the
        end of a sequence.
      include_pad: Whether to append `pad_token` to `tokens` to marks past end
        of sequence.
      include_mask: Whether to append `mask_token` to `tokens` to mark masked
        positions.
      bos_token: A special token than marks the beginning of sequence.
        Ignored if `include_bos == False`.
      eos_token: A special token than marks the end of sequence.
        Ignored if `include_eos == False`.
      pad_token: A special token than marks past the end of sequence.
        Ignored if `include_pad == False`.
      mask_token: A special token than marks MASKED positions for e.g. BERT.
        Ignored if `include_mask == False`.
      disallow_sep_token: If True, disallow `|` appearing in the vocabulary,
      which is used as separator token when serializing to csv.
    """
    if not isinstance(tokens, abc.Iterable):
      tokens = range(tokens)
    tokens = [str(token) for token in tokens]
    if include_bos:
      tokens.append(bos_token)
    if include_eos:
      tokens.append(eos_token)
    if include_pad:
      tokens.append(pad_token)
    if include_mask:
      tokens.append(mask_token)
    if len(set(tokens)) != len(tokens):
      raise ValueError('tokens not unique!')
    if disallow_sep_token:
      special_tokens = sorted(set(tokens) & set([SEP_TOKEN]))
      if special_tokens:
        raise ValueError(
            f'tokens contains reserved special tokens: {special_tokens}!')

    self._name = name
    self._set_tokens(tokens)
    self._bos_token = bos_token if include_bos else None
    self._eos_token = eos_token if include_eos else None
    self._mask_token = mask_token if include_mask else None
    self._pad_token = pad_token if include_pad else None

  def __eq__(self, other):
    self_state = self.__getstate__()
    other_state = other.__getstate__()
    return self_state == other_state

  def _set_tokens(self, tokens):
    """Set self._tokens and related indices from list of token strings."""
    self._tokens = tokens
    self._token_ids = list(range(len(tokens)))
    self._id_to_token = collections.OrderedDict(
        zip(self._token_ids, self._tokens))
    self._token_to_id = collections.OrderedDict(
        zip(self._tokens, self._token_ids))

  def __setstate__(self, state):
    """Create vocab from dict version."""
    tokens = state['tokens']
    self._set_tokens(tokens)
    self._name = state['name']
    self._bos_token = state['bos_token']
    self._eos_token = state['eos_token']
    self._mask_token = state['mask_token']
    self._pad_token = state['pad_token']

  def __getstate__(self):
    """Serialize vocabulary to dict."""
    return copy.deepcopy(
        dict(
            tokens=self._tokens,
            name=self._name,
            token_ids=self._tokens,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            pad_token=self._pad_token,
            mask_token=self._mask_token,
        ))

  def as_dict(self):
    """Serialize vocabulary to dict."""
    return self.__getstate__()

  @classmethod
  def from_dict(cls, state):
    """Create vocabulary from dict."""
    vocab = Vocabulary(1)
    vocab.__setstate__(state)
    return vocab

  def copy(self, **kwargs):
    """Returns new Vocabulary with fields in **kwargs replaced."""
    field_dict = self.as_dict()
    field_dict.update(**kwargs)
    return Vocabulary.from_dict(field_dict)

  def __len__(self):
    return len(self._tokens)

  @property
  def name(self):
    return self._name

  @property
  def tokens(self):
    """Return the tokens of the vocabulary."""
    return list(self._tokens)

  @property
  def token_ids(self):
    """Return the tokens ids of the vocabulary."""
    return list(self._token_ids)

  @property
  def bos(self):
    """Returns the index of the BOS token or None if unspecified."""
    return (None if self._bos_token is None else
            self._token_to_id[self._bos_token])

  @property
  def eos(self):
    """Returns the index of the EOS token or None if unspecified."""
    return (None if self._eos_token is None else
            self._token_to_id[self._eos_token])

  @property
  def mask(self):
    """Returns the index of the MASK token or None if unspecified."""
    return (None if self._mask_token is None else
            self._token_to_id[self._mask_token])

  @property
  def pad(self):
    """Returns the index of the PAD token or None if unspecified."""
    return (None
            if self._pad_token is None else self._token_to_id[self._pad_token])

  def is_valid(self, value):
    """Tests if a value is a valid token id and returns a bool."""
    return value in self._token_ids

  def are_valid(self, values):
    """Tests if values are valid token ids and returns an array of bools."""
    return np.array([self.is_valid(value) for value in values])

  def encode_token(self, token):
    """Maps a single character to an int."""
    return self._token_to_id[token]

  def encode(self, tokens):
    """Maps an iterable of string tokens to a list of integer token ids."""
    if isinstance(tokens, bytes):
      tokens = tokens.decode('utf-8')
    return [self.encode_token(token) for token in tokens]

  def decode_token(self, int_token):
    """Maps a single int to a character."""
    return self._id_to_token[int_token]

  def decode(self, values, stop_at_eos=False, as_str=False):
    """Maps an iterable of integer token ids to string tokens.

    Args:
      values: An iterable of token ids.
      stop_at_eos: Whether to ignore all values after the first EOS token id.
      as_str: Whether to return a list of tokens or a concatenated string.

    Returns:
      A string of tokens or a list of tokens if `as_str == False`.
    """
    if stop_at_eos and self.eos is None:
      raise ValueError('EOS unspecified!')
    tokens = []
    for value in values:
      value = int(value)  # Requires if value is a scalar tensor.
      if stop_at_eos and value == self.eos:
        break
      tokens.append(self.decode_token(value))
    return ''.join(tokens) if as_str else tokens


def make_protein_vocab(include_anomalous_amino_acids=True,
                       include_bos=True,
                       include_eos=True,
                       include_pad=True,
                       include_mask=True,
                       include_align_tokens=False):
  """Returns a vocabulary for proteins."""
  tokens = list(_AA_TOKENS)
  if include_anomalous_amino_acids:
    tokens.extend(list(_ANOMALOUS_AA_TOKENS))
  if include_align_tokens:
    tokens += list(ALIGN_TOKENS)

  return Vocabulary(
      tokens=tokens,
      include_bos=include_bos,
      include_eos=include_eos,
      include_pad=include_pad,
      include_mask=include_mask)

