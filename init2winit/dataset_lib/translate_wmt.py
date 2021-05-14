# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

"""WMT15 en->de MT input pipeline."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import mt_pipeline
from init2winit.dataset_lib.data_utils import Dataset
import jax

from ml_collections.config_dict import config_dict

VOCAB_SIZE = 32000  # Typical vocab_size for MT models.

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        tfds_dataset_key='wmt15_translate/de-en',
        tfds_eval_dataset_key='wmt14_translate/de-en',
        reverse_translation=False,
        # If vocab_path is None, dataset_lib.mt_pipeline generates one with spm.
        # Right now, they have been generated offline and set in
        # experiments.translate dir.
        # TODO(ankugarg): Generate vocab with determinism.
        vocab_path=None,
        vocab_size=VOCAB_SIZE,
        max_corpus_chars=10**7,
        max_target_length=256,  # filter on (spm tokenized, not raw) train data.
        max_eval_target_length=256,  # similarly filter on eval data.
        max_predict_length=256,  # and filter on test examples for BLEU scores.
        train_split='train',
        eval_split='test',
        pack_examples=False,
        output_shape=(VOCAB_SIZE,),
        train_size=4522998,  # raw data size, update with filtered data size.
        input_shape=[(100,), (100,)],  # dummy small values to init_by_shape.
        # If None, datasets.get_dataset_hparams() adds input_shape:
        # max_len = max(max_target_length,
        #               max_eval_target_length,
        #               max_predict_length)
        # input_shape = [(max_len,), (max_len,)]
    ))

METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_outputs': True,
    'causal': True,
}


def get_translate_wmt(shuffle_rng, batch_size, eval_batch_size=None, hps=None):
  """Wrapper to conform to the general dataset API."""

  per_host_batch_size = batch_size // jax.host_count()
  per_host_eval_batch_size = eval_batch_size // jax.host_count()
  return _get_translate_wmt(per_host_batch_size,
                            per_host_eval_batch_size,
                            hps,
                            shuffle_rng)


def _get_translate_wmt(per_host_batch_size,
                       per_host_eval_batch_size,
                       hps,
                       shuffle_rng):
  """Data generators for wmt translate task."""
  n_devices = jax.local_device_count()
  if per_host_batch_size % n_devices != 0:
    raise ValueError('n_devices={} must divide per_host_batch_size={}.'.format(
        n_devices, per_host_batch_size))
  if per_host_eval_batch_size % n_devices != 0:
    raise ValueError(
        'n_devices={} must divide per_host_eval_batch_size={}.'.format(
            n_devices, per_host_eval_batch_size))

  vocab_path = hps.vocab_path
  train_ds, eval_ds, _ = mt_pipeline.get_wmt_datasets(
      hps,
      shuffle_seed=shuffle_rng[0],
      n_devices=jax.local_device_count(),
      per_host_batch_size=per_host_batch_size,
      per_host_eval_batch_size=per_host_eval_batch_size,
      vocab_path=vocab_path)

  def train_iterator_fn():
    for batch in iter(train_ds):
      yield data_utils.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_batch_size,
          data_format=None,
          mask_key='targets')

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      yield data_utils.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_batch_size,
          data_format=None,
          mask_key='targets')

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch in itertools.islice(valid_iter, num_batches):
      yield data_utils.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_eval_batch_size,
          data_format=None,
          mask_key='targets')

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.

  # pylint: enable=unreachable

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
