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

"""WMT15 en->de MT input pipeline."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib import mt_pipeline
from init2winit.dataset_lib.data_utils import Dataset
import jax
from ml_collections.config_dict import config_dict
import numpy as np

VOCAB_SIZE = 32000  # Typical vocab_size for MT models.

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        # one of tfds_dataset_key and tfds_dataset_keys should have values.
        # tfds_dataset_key needs to be a string: (used for bilingual settings)
        tfds_dataset_key=None,
        # tfds_dataset_keys needs to be a list of keys:
        # (used for multilingual settings)
        tfds_dataset_keys=[],
        tfds_eval_dataset_key='',  # tfds_eval_dataset_key needs to be a string.
        # If 'tfds_predict_dataset_key' is None,
        # 'tfds_eval_dataset_key' is used.
        tfds_predict_dataset_key=None,
        reverse_translation=False,
        # If vocab_path is None, dataset_lib.mt_pipeline generates one with spm.
        # Right now, they have been generated offline and set in
        # experiments.translate dir.
        # TODO(ankugarg): Generate vocab with determinism.
        # If 'rates' is None defaults to uniform.
        # Otherwise, 'rates' is sampling rates to use when we have multiple
        # keys, i.e. 'rates' is a list of the same size as 'tfds_dataset_keys'.
        rates=None,
        loss_weights=None,
        add_language_token=False,
        vocab_path=None,
        vocab_size=VOCAB_SIZE,
        max_corpus_chars=10**7,
        max_target_length=256,  # filter on (spm tokenized, not raw) train data.
        max_eval_target_length=256,  # filter on eval data.
        max_predict_length=256,  # filter on test data.
        train_split='train',
        eval_split='validation',
        predict_split='test',
        pack_examples=True,
        output_shape=(VOCAB_SIZE,),
        train_size=4522998,  # raw data size, update with filtered data size.
        input_shape=[(100,), (100,)],  # dummy small values to init_by_shape.
        character_coverage=1.0,
        byte_fallback=False,
        split_digits=False,
        # If None, datasets.get_dataset_hparams() adds input_shape:
        # max_len = max(max_target_length,
        #               max_eval_target_length,
        #               max_predict_length)
        # input_shape = [(max_len,), (max_len,)]
    ))

METADATA = {
    'apply_one_hot_in_loss': True,
    'causal': True,
}


def get_translate_wmt(shuffle_rng, batch_size, eval_batch_size=None, hps=None):
  """Wrapper to conform to the general dataset API."""

  per_host_batch_size = batch_size // jax.process_count()
  per_host_eval_batch_size = eval_batch_size // jax.process_count()
  return _get_translate_wmt(per_host_batch_size,
                            per_host_eval_batch_size,
                            hps,
                            shuffle_rng)


def validate_hparams(hps):
  """This function checks the type-validity of different hyperparameters."""

  if not hps.tfds_dataset_key:  # trains bilingual model if tfds_dataset_key
    if hps.tfds_dataset_keys:  # trains multilingual model
      assert len(hps.tfds_dataset_keys) >= 1
    else:
      raise ValueError('Either set tfds_dataset_key to train bilingual model' +
                       'or set tfds_dataset_keys to train multilingual model')
    if hps.rates:
      assert len(hps.tfds_dataset_keys) == len(hps.rates)


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

  validate_hparams(hps)

  vocab_path = hps.vocab_path
  shuffle_seed, sample_seed = jax.random.split(shuffle_rng, 2)
  train_ds, eval_ds, predict_ds = mt_pipeline.get_wmt_datasets(
      hps,
      shuffle_seed=data_utils.convert_jax_to_tf_random_seed(shuffle_seed),
      sample_seed=data_utils.convert_jax_to_tf_random_seed(sample_seed),
      n_devices=jax.local_device_count(),
      per_host_batch_size=per_host_batch_size,
      per_host_eval_batch_size=per_host_eval_batch_size,
      vocab_path=vocab_path)

  def train_iterator_fn():
    for batch in iter(train_ds):
      yield mt_pipeline.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_batch_size,
          mask_key='targets')

  def eval_train_epoch(num_batches=None):
    eval_train_iter = iter(train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      yield mt_pipeline.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_batch_size,
          mask_key='targets')

  def valid_epoch(num_batches=None):
    valid_iter = iter(eval_ds)
    for batch in itertools.islice(valid_iter, num_batches):
      yield mt_pipeline.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_eval_batch_size,
          mask_key='targets')

  def test_epoch(num_batches=None):
    predict_iter = iter(predict_ds)
    for batch in itertools.islice(predict_iter, num_batches):
      yield mt_pipeline.maybe_pad_batch(
          data_utils.tf_to_numpy(batch),
          per_host_eval_batch_size,
          mask_key='targets')

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)


def get_fake_batch(hps):
  """Build fake batch for translate_wmt."""
  batch = {
      'inputs':
          np.ones((hps.batch_size, hps.max_target_length),
                  dtype=np.int32),
      'targets':
          np.ones((hps.batch_size, hps.max_target_length),
                  dtype=np.int32),
      'weights':
          np.ones((hps.batch_size, hps.max_target_length),
                  dtype=np.int64),
  }

  if hps.pack_examples:
    batch.update({
        'inputs_position':
            np.ones((hps.batch_size, hps.max_target_length),
                    dtype=np.int32),
        'inputs_segmentation':
            np.ones((hps.batch_size, hps.max_target_length),
                    dtype=np.int32),
        'targets_position':
            np.ones((hps.batch_size, hps.max_target_length),
                    dtype=np.int32),
        'targets_segmentation':
            np.ones((hps.batch_size, hps.max_target_length),
                    dtype=np.int32),
    })

    return batch
