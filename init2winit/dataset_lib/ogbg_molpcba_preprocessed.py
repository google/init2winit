# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Data generator for preprocessed ogbg_molpcba."""

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import ogbg_molpcba
import jax
from ml_collections.config_dict import config_dict
import tensorflow as tf

# Reuse default hparams from ogbg_molpcba, but add dataset_path
DEFAULT_HPARAMS = config_dict.ConfigDict(ogbg_molpcba.DEFAULT_HPARAMS)
DEFAULT_HPARAMS.dataset_path = ''


METADATA = ogbg_molpcba.METADATA


def get_ogbg_molpcba_preprocessed(
    shuffle_rng, batch_size, eval_batch_size, hps=None
):
  """Get preprocessed ogbg_molpcba dataset.

  The saved batches must have the correct per-host batch size (batch_size /
  process_count).

  Args:
    shuffle_rng: RNG key for shuffling (used for fallback).
    batch_size: Global batch size.
    eval_batch_size: Global eval batch size.
    hps: Hyperparameters.

  Returns:
    Dataset object.
  """
  if not hps.dataset_path:
    raise ValueError(
        'hps.dataset_path must be provided for ogbg_molpcba_preprocessed'
    )

  # Fallback to original dataset for validation/test splits
  original_ds = ogbg_molpcba.get_ogbg_molpcba(
      shuffle_rng,
      batch_size,
      eval_batch_size,
      hps,
      override_process_count=None,
  )

  def train_iterator_fn():
    ds = tf.data.Dataset.load(hps.dataset_path, compression='GZIP')

    process_count = jax.process_count()
    process_index = jax.process_index()

    if process_count > 1:
      ds = ds.shard(process_count, process_index)

    iterator = iter(ds.as_numpy_iterator())
    return iterator

  return data_utils.Dataset(
      train_iterator_fn,
      original_ds.eval_train_epoch,
      original_ds.valid_epoch,
      original_ds.test_epoch,
  )


def generate_and_save_dataset(
    hps,
    output_path,
    num_steps,
    batch_size,
    seed=0,
    progress_bar_fn=None,
    target_num_processes=None,
):
  """Generates and saves the ogbg_molpcba dataset.

  Args:
    hps: Hyperparameters.
    output_path: Path to save the dataset.
    num_steps: Number of steps (batches) to generate.
    batch_size: Batch size.
    seed: Random seed.
    progress_bar_fn: Optional function to wrap the iterator for progress
      tracking. Should accept an iterable and return an iterable.
    target_num_processes: If set, the dataset will be sharded for this number of
      processes (=number of TPUs usually).
  """
  rng = jax.random.PRNGKey(seed)

  dataset_builder = ogbg_molpcba.get_ogbg_molpcba(
      shuffle_rng=rng,
      batch_size=batch_size,
      eval_batch_size=batch_size,
      hps=hps,
      override_process_count=target_num_processes,
  )

  train_iter = dataset_builder.train_iterator_fn()

  def generator():
    iterator = range(num_steps)
    if progress_bar_fn:
      iterator = progress_bar_fn(iterator)

    for _ in iterator:
      try:
        batch = next(train_iter)
        yield batch
      except StopIteration:
        return

  # Determine output signature from the first batch
  peek_iter = dataset_builder.train_iterator_fn()
  first_batch = next(peek_iter)

  output_signature = tf.nest.map_structure(
      lambda x: tf.TensorSpec(shape=x.shape, dtype=tf.as_dtype(x.dtype)),
      first_batch,
  )

  ds = tf.data.Dataset.from_generator(
      generator, output_signature=output_signature
  )

  ds.save(output_path, compression='GZIP')
