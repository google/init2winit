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

"""Data generators for init2winit."""

import collections

from init2winit.dataset_lib import criteo_terabyte_input_pipeline
from init2winit.dataset_lib import fake_dataset
from init2winit.dataset_lib import fastmri_dataset
from init2winit.dataset_lib import imagenet_dataset
from init2winit.dataset_lib import librispeech
from init2winit.dataset_lib import lm1b_v2
from init2winit.dataset_lib import mlperf_imagenet_dataset
from init2winit.dataset_lib import nqm_noise
from init2winit.dataset_lib import ogbg_molpcba
from init2winit.dataset_lib import proteins
from init2winit.dataset_lib import small_image_datasets
from init2winit.dataset_lib import translate_wmt
from init2winit.dataset_lib import wikitext2

_Dataset = collections.namedtuple(
    'Dataset', ('getter', 'hparams', 'meta_data', 'fake_batch_getter'))

_ALL_DATASETS = {
    'mnist':
        _Dataset(small_image_datasets.get_mnist,
                 small_image_datasets.MNIST_HPARAMS,
                 small_image_datasets.MNIST_METADATA, None),
    'mnist_autoencoder':
        _Dataset(small_image_datasets.get_mnist_autoencoder,
                 small_image_datasets.MNIST_AUTOENCODER_HPARAMS,
                 small_image_datasets.MNIST_AUTOENCODER_METADATA, None),
    'fashion_mnist':
        _Dataset(small_image_datasets.get_fashion_mnist,
                 small_image_datasets.FASHION_MNIST_HPARAMS,
                 small_image_datasets.FASHION_MNIST_METADATA, None),
    'cifar10':
        _Dataset(small_image_datasets.get_cifar10,
                 small_image_datasets.CIFAR10_DEFAULT_HPARAMS,
                 small_image_datasets.CIFAR10_METADATA, None),
    'cifar100':
        _Dataset(small_image_datasets.get_cifar100,
                 small_image_datasets.CIFAR100_DEFAULT_HPARAMS,
                 small_image_datasets.CIFAR100_METADATA, None),
    'criteo1tb':
        _Dataset(criteo_terabyte_input_pipeline.get_criteo1tb,
                 criteo_terabyte_input_pipeline.CRITEO1TB_DEFAULT_HPARAMS,
                 criteo_terabyte_input_pipeline.CRITEO1TB_METADATA, None),
    'fake':
        _Dataset(fake_dataset.get_fake, fake_dataset.DEFAULT_HPARAMS,
                 fake_dataset.METADATA, None),
    'fastmri':
        _Dataset(fastmri_dataset.get_fastmri, fastmri_dataset.DEFAULT_HPARAMS,
                 fastmri_dataset.METADATA, None),
    'imagenet':
        _Dataset(imagenet_dataset.get_imagenet,
                 imagenet_dataset.DEFAULT_HPARAMS, imagenet_dataset.METADATA,
                 None),
    'translate_wmt':
        _Dataset(translate_wmt.get_translate_wmt, translate_wmt.DEFAULT_HPARAMS,
                 translate_wmt.METADATA, None),
    'librispeech':
        _Dataset(librispeech.get_librispeech, librispeech.DEFAULT_HPARAMS,
                 librispeech.METADATA, None),
    'lm1b_v2':
        _Dataset(lm1b_v2.get_lm1b, lm1b_v2.DEFAULT_HPARAMS, lm1b_v2.METADATA,
                 None),
    'mlperf_imagenet':
        _Dataset(mlperf_imagenet_dataset.get_mlperf_imagenet,
                 mlperf_imagenet_dataset.DEFAULT_HPARAMS,
                 mlperf_imagenet_dataset.METADATA, None),
    'svhn_no_extra':
        _Dataset(small_image_datasets.get_svhn_no_extra,
                 small_image_datasets.SVHN_NO_EXTRA_DEFAULT_HPARAMS,
                 small_image_datasets.SVHN_NO_EXTRA_METADATA, None),
    'nqm_noise':
        _Dataset(nqm_noise.get_nqm_noise, nqm_noise.NQM_HPARAMS,
                 nqm_noise.NQM_METADATA, None),
    'ogbg_molpcba':
        _Dataset(ogbg_molpcba.get_ogbg_molpcba, ogbg_molpcba.DEFAULT_HPARAMS,
                 ogbg_molpcba.METADATA, None),
    'uniref50':
        _Dataset(proteins.get_uniref, proteins.DEFAULT_HPARAMS,
                 proteins.METADATA, None),
    'wikitext2':
        _Dataset(wikitext2.get_wikitext2, wikitext2.DEFAULT_HPARAMS,
                 wikitext2.METADATA, None),
}


def get_dataset(dataset_name):
  """Maps dataset name to a dataset_builder."""
  try:
    return _ALL_DATASETS[dataset_name].getter
  except KeyError:
    raise ValueError('Unrecognized dataset: {}'.format(dataset_name)) from None


def get_dataset_hparams(dataset_name):
  """Maps dataset name to default_hps."""
  try:
    hparams = _ALL_DATASETS[dataset_name].hparams
    # TODO(mbadura): Refactor to explicitly support different input specs
    if 'input_shape' not in hparams or hparams.input_shape is None:
      if 'input_edge_shape' in hparams and 'input_node_shape' in hparams:
        hparams.input_shape = (hparams.input_node_shape,
                               hparams.input_edge_shape)
      elif dataset_name == 'lm1b_v2':
        max_len = max(hparams.max_target_length, hparams.max_eval_target_length)
        hparams.input_shape = (max_len,)
      elif dataset_name == 'translate_wmt':
        max_len = max(hparams.max_target_length, hparams.max_eval_target_length,
                      hparams.max_predict_length)
        hparams.input_shape = [(max_len,), (max_len,)]
      else:
        raise ValueError(
            'Undefined input shape for dataset: {}'.format(dataset_name))
    return hparams
  except KeyError:
    raise ValueError('Unrecognized dataset: {}'.format(dataset_name)) from None


def get_dataset_meta_data(dataset_name):
  """Maps dataset name to a dict of dataset meta_data.

  meta_data is a dictionary where the keys will depend on the dataset.
  Required keys - All models: num_classes, is_one_hot
  This dictionary will be referenced by the model constructor.
  New datasets may have new keys, this will work as long as the model
  references the key properly.

  Args:
    dataset_name: (str) the name of the dataset.

  Returns:
    A dict of dataset metadata as described above.
  Raises:
    ValueError, if the dataset_name is not in _ALL_DATASETS.
  """
  try:
    return _ALL_DATASETS[dataset_name].meta_data
  except KeyError:
    raise ValueError('Unrecognized dataset: {}'.format(dataset_name)) from None


def get_fake_batch(dataset_name):
  """Maps dataset name to fake_batch_getter.

  fake_batch_getter is a function that takes a `ConfigDict` of fully formed hps
  (e.g., the output from `hyperparameters.build_hparams`)

  Args:
    dataset_name: (str) the name of the dataset.

  Returns:
    A function that takes a `ConfigDict` and returns a fake batch of the same
    shape as a real batch from the dataset.
  Raises
    ValueError, if `dataset_name` is not in `_ALL_DATASETS` or the
    `fake_batch_getter` is `None`.
  """
  try:
    getter = _ALL_DATASETS[dataset_name].fake_batch_getter
    if getter is None:
      raise ValueError(
          f'Fake batch getter not defined for dataset {dataset_name}') from None
  except KeyError:
    raise ValueError('Unrecognized dataset: {}'.format(dataset_name)) from None
