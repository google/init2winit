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

"""Nanodo data pipeline."""

# Implementation based on:
# https://github.com/google-deepmind/nanodo/blob/main/nanodo/data.py
# This pygrain dataloader can be used with any TFDS datasource (e.g. c4, lm1b,
# etc.) but is primarily used to load C4 dataset in it's current form.
# This data loader is deterministically reproducible with shuffling guarantees
# but we don't checkpoint dataset iterators in i2w yet so this is not
# guaranteed to be reproducible across restarts.

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import itertools
from typing import Iterable, Iterator, Union

import grain.python as grain
from init2winit.dataset_lib import data_utils
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow_datasets as tfds

import sentencepiece as spm


PAD_ID = 0
### pure python helpers for use with grain ###


VOCAB_SIZE = 32101

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        max_target_length=512,
        max_eval_target_length=512,
        vocab_size=VOCAB_SIZE,
        train_size=364613570,
        eval_split='validation',
        vocab_path=None,
        pygrain_worker_count=16,
        pygrain_worker_buffer_size=10,
        pygrain_eval_worker_count=0,
    )
)

METADATA = {
    'apply_one_hot_in_loss': True,
}


class Preprocess(enum.Enum):
  NOAM_PACKED = 1
  PADDED = 2


def py_batched_tfds(
    *,
    tfds_name: str,
    split: str,
    context_size: int,
    worker_count: int,
    vocab_path: str,
    batch_size: int,
    seed: int | None = 1234,
    num_epochs: int | None = None,
    num_records: int | None = None,
    preprocessing: Preprocess = Preprocess.NOAM_PACKED,
    worker_buffer_size: int = 2,
    shuffle: bool = True,
) -> grain.DataLoader:
  """Returns iterator for regularly batched text examples."""
  datasource = tfds.data_source(tfds_name, split=split)
  index_sampler = grain.IndexSampler(
      num_records=num_records if num_records is not None else len(datasource),
      num_epochs=num_epochs,
      shard_options=grain.ShardByJaxProcess(),
      shuffle=shuffle,
      seed=seed,
  )
  spt = _SPTokenizer(vocab_path)
  per_device_batch_size = batch_size // jax.device_count()
  per_host_batch_size = per_device_batch_size * jax.local_device_count()

  pad_len = None if preprocessing == Preprocess.NOAM_PACKED else context_size
  pygrain_ops = [
      grain.MapOperation(
          map_function=functools.partial(
              _py_tokenize,
              spt=spt,
              pad_len=pad_len,
          )
      )
  ]
  if preprocessing == Preprocess.NOAM_PACKED:
    pygrain_ops.append(_NoamPack(context_size=context_size))
  elif preprocessing == Preprocess.PADDED:
    pygrain_ops.append(grain.MapOperation(map_function=np.array))
  else:
    raise ValueError(f'Unknown preprocessing: {preprocessing}')
  pygrain_ops.append(
      grain.Batch(batch_size=per_host_batch_size, drop_remainder=True)
  )
  batched_dataloader = grain.DataLoader(
      data_source=datasource,
      operations=pygrain_ops,
      sampler=index_sampler,
      worker_count=worker_count,
      worker_buffer_size=worker_buffer_size
  )
  return batched_dataloader


def get_py_tokenizer(path: str) -> spm.SentencePieceProcessor:
  if not path:
    # byte tokenizer shortcut
    return _SentencePieceByteTokenizer()
  sp = spm.SentencePieceProcessor()
  sp.Load(path)
  assert sp.pad_id() == PAD_ID
  assert sp.eos_id() != -1
  assert sp.bos_id() != -1
  return sp


# Need this because we can't pickle SentencePieceProcessor object
class _SPTokenizer:
  """Wrapper class for SentencePiece tokenizer."""

  def __init__(self, vocab_path):
    self._tokenizer = None
    self._vocab_path = vocab_path

  def get_tokenizer(self) -> spm.SentencePieceProcessor:
    if not self._tokenizer:
      self._tokenizer = get_py_tokenizer(self._vocab_path)
    return self._tokenizer


class _SentencePieceByteTokenizer(spm.SentencePieceProcessor):
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


def _py_tokenize(
    features: Mapping[str, str],
    spt: _SPTokenizer,
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
class _NoamPack:
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


def get_c4(
    shuffle_rng, batch_size, eval_batch_size=None, hps=None
):
  """Data generators for Nanodo."""

  shuffle_seed = data_utils.convert_jax_to_tf_random_seed(shuffle_rng)

  train_ds = py_batched_tfds(
      tfds_name='c4:3.1.0',
      split='train',
      context_size=hps.max_target_length,
      worker_count=hps.pygrain_worker_count,
      vocab_path=hps.vocab_path,
      batch_size=batch_size,
      preprocessing=Preprocess.NOAM_PACKED,
      worker_buffer_size=hps.pygrain_worker_buffer_size,
      seed=int(shuffle_seed),
      shuffle=True,
  )

  if not eval_batch_size:
    eval_batch_size = batch_size

  eval_ds = py_batched_tfds(
      tfds_name='c4:3.1.0',
      split=hps.eval_split,
      context_size=hps.max_eval_target_length,
      worker_count=hps.pygrain_eval_worker_count,
      vocab_path=hps.vocab_path,
      batch_size=eval_batch_size,
      num_epochs=1,
      num_records=None,
      preprocessing=Preprocess.PADDED,
      shuffle=False,
  )

  def train_iterator_fn():
    for example in iter(train_ds):
      inputs, targets, weights = get_in_out(example)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights
      }
      yield batch

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(eval_ds)
    for example in itertools.islice(eval_train_iter, num_batches):
      inputs, targets, weights = get_in_out(example)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights
      }

      yield batch

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for example in itertools.islice(valid_iter, num_batches):
      inputs, targets, weights = get_in_out(example)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights
      }
      yield batch

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable
  return data_utils.Dataset(
      train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch
  )
