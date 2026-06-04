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

"""Imagenette input pipeline with ImageNet-style preprocessing.

Imagenette is a subset of 10 easily classified classes from ImageNet,
created by Jeremy Howard / fastai. It uses the same preprocessing as ImageNet
(JPEG decode, random crop, resize to 224x224, ImageNet mean/std normalization).

TFDS dataset: imagenette/full-size-v2
  - Train: 9,469 images
  - Validation: 3,925 images
  - No test split
"""

from init2winit.dataset_lib import imagenet_dataset
from ml_collections.config_dict import config_dict

NUM_CLASSES = 10

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        input_shape=(224, 224, 3),
        output_shape=(NUM_CLASSES,),
        train_size=9469,
        valid_size=3925,
        test_size=0,
        crop='random',  # options are: {"random", "inception", "center"}
        random_flip=True,
        use_mixup=False,
        mixup={'alpha': 0.5},
        use_randaug=False,
        randaug={'magnitude': 15, 'num_layers': 2},
        use_grain=False,
    )
)

METADATA = {
    'apply_one_hot_in_loss': False,
}

TFDS_DATASET_NAME = 'imagenette/full-size-v2:1.*.*'

get_fake_batch = imagenet_dataset.get_fake_batch


def get_imagenette(
    shuffle_rng, batch_size, eval_batch_size, hps, global_step=0
):
  """Data generators for Imagenette."""
  return imagenet_dataset.get_imagenet(
      shuffle_rng=shuffle_rng,
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      hps=hps,
      global_step=global_step,
      tfds_dataset_name=TFDS_DATASET_NAME,
      num_classes=NUM_CLASSES,
  )
