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

"""Common utils for nanodo data loaders.

Implementation based on:
https://github.com/google-deepmind/nanodo/blob/main/nanodo/data.py
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
from typing import Iterable, Iterator, Union

import grain.python as grain
import jax
import jax.numpy as jnp

import numpy as np

import sentencepiece as spm


PAD_ID = 0


class Preprocess(enum.Enum):
  NOAM_PACKED = 1
  PADDED = 2


### pure python helpers for use with grain ###
# Need this because we can't pickle SentencePieceProcessor object
class SPTokenizer:
  """Wrapper class for SentencePiece tokenizer."""

  def __init__(self, vocab_path):
    self._tokenizer = None
    self._vocab_path = vocab_path

  def get_tokenizer(self) -> spm.SentencePieceProcessor:
    if not self._tokenizer:
      self._tokenizer = get_py_tokenizer(self._vocab_path)
    return self._tokenizer


class SentencePieceByteTokenizer(spm.SentencePieceProcessor):
  """A simple Byte level tokenizer."""

  def eos_id(self) -> int:
    return 1

  def bos_id(self) -> int:
    return 2

  def pad_id(self) -> int:
    return PAD_ID

  def GetPieceSize(self) -> int:
    return 256

  # pylint: disable=invalid-name
  def EncodeAsIds(self, text: Union[bytes, str]) -> list[int]:
    if isinstance(text, str):
      return list(bytes(text, 'utf-8'))
    if isinstance(text, bytes):
      return [int(x) for x in text]
    raise ValueError(f'Invalid text: {text} type={type(text)}')

  def DecodeIds(self, ids: Iterable[int]) -> str:
    return bytes(ids).decode('utf-8')
  # pylint: enable=invalid-name


def py_tokenize(
    features: Mapping[str, str],
    spt: SPTokenizer,
    pad_len: int | None = None,
    pad_id: int = PAD_ID,
) -> Sequence[int]:
  """Tokenizes text into ids, optionally pads or truncates to pad_len."""
  text = features['text']
  tokenizer = spt.get_tokenizer()
  bos_id = tokenizer.bos_id()
  eos_id = tokenizer.eos_id()
  ids = tokenizer.EncodeAsIds(text)

  ids.insert(0, bos_id)
  ids.append(eos_id)
  if pad_len is not None:
    if len(ids) < pad_len:
      ids.extend([pad_id] * (pad_len - len(ids)))
    elif len(ids) > pad_len:
      ids = ids[:pad_len]
  return ids


@dataclasses.dataclass
class NoamPack:
  """Pygrain operation for tokenizing and Noam packing text."""

  context_size: int

  def __call__(
      self, idseq_iterator: Iterator[grain.Record]
  ) -> Iterator[grain.Record]:
    packed_ids = []
    for input_record in idseq_iterator:
      start = 0
      while start < len(input_record.data):
        rem_data = input_record.data[start:]
        if len(packed_ids) + len(rem_data) < self.context_size:
          packed_ids.extend(rem_data)  # use rest of example, move-on
          break
        else:
          take = self.context_size - len(packed_ids)
          packed_ids.extend(rem_data[:take])
          last_record_key = input_record.metadata.remove_record_key()
          yield grain.Record(
              last_record_key, np.array(packed_ids, dtype=np.int32)
          )
          start += take
          packed_ids = []
          # Drop remainder for simplicity.
          # We lose the rest of the example on restore.


# pylint: disable=invalid-name


def get_py_tokenizer(path: str) -> spm.SentencePieceProcessor:
  if not path:
    # byte tokenizer shortcut
    return SentencePieceByteTokenizer()
  sp = spm.SentencePieceProcessor()
  sp.Load(path)
  assert sp.pad_id() == PAD_ID
  assert sp.eos_id() != -1
  assert sp.bos_id() != -1
  return sp


def get_in_out(
    in_BxL: jax.Array,
    pad_id: int = PAD_ID,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns input, output, and weights for a batch of examples."""
  # Assumes input of the form <BOS> <IDs> <EOS> for eval.
  x_BxL = in_BxL
  y_BxL = jnp.pad(
      in_BxL[:, 1:],
      ((0, 0), (0, 1)),
      mode='constant',
      constant_values=pad_id,
  )
  weights_BxL = jnp.where(y_BxL != pad_id, 1, 0).astype(jnp.float32)

  return x_BxL, y_BxL, weights_BxL
