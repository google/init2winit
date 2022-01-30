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

"""init2winit datasets."""

from init2winit import import_utils

# By lazily importing, we do not need to import the entire library even if we
# are only using a few models/datasets/optimizers, which substantially cuts down
# on import times.
_IMPORTS = [
    'datasets',
    'data_utils',
    'fake_dataset',
    'imagenet_dataset',
    'image_preprocessing',
    'lm1b_input_pipeline',
    'lm1b',
    'mlperf_imagenet_dataset',
    'mlperf_input_pipeline',
    'mt_pipeline',
    'mt_pipeline_test',
    'mt_tokenizer',
    'nqm_noise',
    'ogbg_molpcba',
    'proteins',
    'protein_vocab',
    'small_image_datasets',
    'test_datasets',
    'test_data_utils',
    'test_ogbg_molpcba',
    'test_small_image_datasets',
    'translate_wmt',
]

__getattr__ = import_utils.lazy_import_fn('dataset_lib', _IMPORTS)

