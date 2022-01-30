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

"""init2winit models."""

from init2winit import import_utils

# By lazily importing, we do not need to import the entire library even if we
# are only using a few models/datasets/optimizers, which substantially cuts down
# on import times.
_IMPORTS = [
    'adabelief_densenet',
    'adabelief_resnet',
    'adabelief_vgg',
    'autoencoder',
    'base_model',
    'convolutional_autoencoder',
    'fully_connected',
    'gnn',
    'losses',
    'max_pooling_cnn',
    'metrics',
    'mlperf_resnet',
    'models',
    'model_utils',
    'normalization',
    'nqm',
    'resnet',
    'simple_cnn',
    'test_losses',
    'test_models',
    'test_normalization',
    'transformer_lm',
    'wide_resnet',
    'xformer_translate',
]

__getattr__ = import_utils.lazy_import_fn('model_lib', _IMPORTS)
