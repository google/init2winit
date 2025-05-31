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

"""Data loader for fineweb_edu dataset."""

import functools
import itertools
from typing import List

from etils import epath
import grain.python as grain
from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import nanodo_data_loader_shared as data_loader
import jax
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow_datasets as tfds


VOCAB_SIZE = 100864
EVAL_SPLIT = 'cc_main_2024_10'


DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        max_target_length=2048,
        max_eval_target_length=2048,
        vocab_size=VOCAB_SIZE,
        train_size=1300000000000,
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

Preprocess = data_loader.Preprocess


def build_data_source(*, dataset_name: str, config: str) -> grain.MapDataset:
  return tfds.data_source(f'huggingface:{dataset_name}/{config}', split='train')


def py_batched_tfds(
    tfds_name: str,
    splits: List[str],
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
  # 2. Build the mixture of all configs.

  ds_list = []
  for split in splits:
    ds = build_data_source(dataset_name=tfds_name, config=split)
    ds = grain.MapDataset.source(ds)
    ds_list.append(ds)

  mix_weights = [len(ds) for ds in ds_list]
  datasource = grain.MapDataset.mix(ds_list, weights=mix_weights)

  index_sampler = grain.IndexSampler(
      num_records=num_records if num_records is not None else len(datasource),
      num_epochs=num_epochs,
      shard_options=grain.ShardByJaxProcess(),
      shuffle=shuffle,
      seed=seed,
  )
  spt = data_loader.SPTokenizer(vocab_path)
  per_device_batch_size = batch_size // jax.device_count()
  per_host_batch_size = per_device_batch_size * jax.local_device_count()

  pad_len = None if preprocessing == Preprocess.NOAM_PACKED else context_size
  pygrain_ops = [
      grain.MapOperation(
          map_function=functools.partial(
              data_loader.py_tokenize,
              spt=spt,
              pad_len=pad_len,
          )
      )
  ]
  if preprocessing == Preprocess.NOAM_PACKED:
    pygrain_ops.append(data_loader.NoamPack(context_size=context_size))
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
      worker_buffer_size=worker_buffer_size,
  )
  return batched_dataloader


def get_dataset(shuffle_rng, batch_size, eval_batch_size=None, hps=None):
  """Data generators for Nanodo."""

  shuffle_seed = data_utils.convert_jax_to_tf_random_seed(shuffle_rng)
  tfds_name = FINEWEB_EDU_TFDS_NAME
  eval_split_name = EVAL_SPLIT

  path = epath.Path(f'{TFDS_HUGGINGFACE_PATH_PREFIX}/{tfds_name}')

  train_splits = [
      filepath.name
      for filepath in path.iterdir()
      if filepath.name != 'default' and filepath.name != eval_split_name
  ]

  eval_splits = [eval_split_name]

  train_ds = py_batched_tfds(
      tfds_name=tfds_name,
      splits=train_splits,
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
      tfds_name=tfds_name,
      splits=eval_splits,
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
      inputs, targets, weights = data_loader.get_in_out(example)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights
      }
      yield batch

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(eval_ds)
    for example in itertools.islice(eval_train_iter, num_batches):
      inputs, targets, weights = data_loader.get_in_out(example)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights
      }

      yield batch

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for example in itertools.islice(valid_iter, num_batches):
      inputs, targets, weights = data_loader.get_in_out(example)
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
