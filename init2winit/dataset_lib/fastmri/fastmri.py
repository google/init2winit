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

"""fastmri dataset."""

import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
See https://fastmri.org.

NOTE: This tensorflow dataset ONLY supports the knee singlecoil challenge.
"""

_CITATION = """\
@article{DBLP:journals/corr/abs-1811-08839,
  author    = {Jure Zbontar and
               Florian Knoll and
               Anuroop Sriram and
               Matthew J. Muckley and
               Mary Bruno and
               Aaron Defazio and
               Marc Parente and
               Krzysztof J. Geras and
               Joe Katsnelson and
               Hersh Chandarana and
               Zizhao Zhang and
               Michal Drozdzal and
               Adriana Romero and
               Michael Rabbat and
               Pascal Vincent and
               James Pinkerton and
               Duo Wang and
               Nafissa Yakubova and
               Erich Owens and
               C. Lawrence Zitnick and
               Michael P. Recht and
               Daniel K. Sodickson and
               Yvonne W. Lui},
  title     = {fastMRI: An Open Dataset and Benchmarks for Accelerated {MRI}},
  journal   = {CoRR},
  volume    = {abs/1811.08839},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.08839},
  archivePrefix = {arXiv},
  eprint    = {1811.08839},
  timestamp = {Mon, 26 Nov 2018 12:52:45 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-08839},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


def _generate_example(hf, split, fname, slice_num):
  """Generate a single example (slice from mri image).

  Args:
    hf: file handle.
    split: which split.
    fname: file path.
    slice_num: slice number.

  Returns:
    UNetSample object for U-Net reconstruction.
  """
  kspace = hf.get('kspace')[slice_num]
  target = hf.get('reconstruction_esc',
                  np.zeros((kspace.shape[0], 320, 320),
                           dtype=np.float32))[slice_num]

  # randomness
  seed = tuple(map(ord, fname))
  rng = np.random.RandomState(seed)
  state = rng.get_state()
  rng.seed(seed)

  if split != 'test':
    # sample_mask
    num_cols = kspace.shape[1]

    # choose_acceleration
    center_fraction, acceleration = 0.8, 4

    num_low_frequencies = round(num_cols * center_fraction)

    # calculate_center_mask
    mask = np.zeros(num_cols, dtype=np.float32)
    pad = (num_cols - num_low_frequencies + 1) // 2
    mask[pad:pad + num_low_frequencies] = 1

    # reshape_mask
    center_mask = mask.reshape(1, num_cols)

    # calculate_acceleration_mask
    prob = (num_cols / acceleration - num_low_frequencies) / (
        num_cols - num_low_frequencies
    )

    mask = np.array(rng.uniform(size=num_cols) < prob, dtype=np.float32)
    acceleration_mask = mask.reshape(1, num_cols)

    rng.set_state(state)

    mask = np.maximum(center_mask, acceleration_mask)

    # apply_mask
    masked_kspace = kspace * mask + 0.0
  else:
    masked_kspace = kspace

  # ifft2c
  shifted_kspace = tf.signal.ifftshift(masked_kspace, axes=(0, 1))
  shifted_image = tf.signal.ifft2d(shifted_kspace)
  image = tf.signal.fftshift(shifted_image, axes=(0, 1))
  scaling_norm = tf.cast(
      tf.math.sqrt(
          tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')),
      kspace.dtype)
  image = (scaling_norm * image).numpy()
  image = np.stack((image.real, image.imag), axis=-1)

  # complex_center_crop
  w_from = (image.shape[-3] - target.shape[0]) // 2
  h_from = (image.shape[-2] - target.shape[1]) // 2
  w_to = w_from + target.shape[0]
  h_to = h_from + target.shape[1]

  image = image[..., w_from:w_to, h_from:h_to, :]

  # complex_abs
  abs_image = np.sqrt((image ** 2).sum(axis=-1))

  # normalize_instance
  mean, std = abs_image.mean(), abs_image.std()
  norm_image = (abs_image - mean) / std

  # clip_image
  image = np.clip(norm_image, -6, 6)

  # process target
  norm_target = (target - mean) / std
  target = np.clip(norm_target, -6, 6)

  return {'image': image, 'target': target}


class FastMRI(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for fastmri dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  - Apply for access here: https://fastmri.med.nyu.edu/.
  - You will receive an email with links to download.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(320, 320), dtype=tf.float32),
            'target': tfds.features.Tensor(shape=(320, 320), dtype=tf.float32),
        }),
        supervised_keys=('image', 'target'),
        homepage='https://fastmri.med.nyu.edu/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    paths = dl_manager.extract({
        'train':
            os.path.join(dl_manager.manual_dir, 'knee_singlecoil_train.tar.gz'),
        'test':
            os.path.join(dl_manager.manual_dir,
                         'knee_singlecoil_test_v2.tar.gz'),
        'val':
            os.path.join(dl_manager.manual_dir, 'knee_singlecoil_val.tar.gz'),
    })

    return {
        'train':
            self._generate_examples(
                'train', os.path.join(paths['train'], 'singlecoil_train')),
        'test':
            self._generate_examples(
                'test', os.path.join(paths['test'], 'singlecoil_test_v2')),
        'val':
            self._generate_examples(
                'val', os.path.join(paths['val'], 'singlecoil_val')),
    }

  def _generate_examples(self, split, path):
    """Yields examples."""
    for image_file in tf.io.gfile.listdir(path):
      fname = os.path.join(path, image_file)
      with h5py.File(fname, 'r') as hf:
        num_slices = hf['kspace'].shape[0]
        for slice_num in range(num_slices):
          key = f'{fname}-{slice_num}'
          example = _generate_example(hf, split, fname, slice_num)
          yield key, example
