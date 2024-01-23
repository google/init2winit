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

"""Preprocessing for image datasets."""

from jax import lax
from jax import random
from jax import vmap
import jax.numpy as jnp
import tensorflow as tf


def _crop(key, image, hps):
  """Randomly shifts the window viewing the image."""
  pixpad = (hps.crop_num_pixels, hps.crop_num_pixels)
  zero = (0, 0)
  padded_image = jnp.pad(image, (pixpad, pixpad, zero),
                         mode='constant', constant_values=0.0)
  corner = random.randint(key, (2,), 0, 2 * hps.crop_num_pixels)
  corner = jnp.concatenate((corner, jnp.zeros((1,), jnp.int32)))
  cropped_image = lax.dynamic_slice(padded_image, corner, image.shape)

  return cropped_image


def crop(key, images, hps):
  """Randomly crops a batch of images.

  For each image the batch we translate the window which views the image. Pixels
  in the view the of the new window which are undefined are shown in black.

  Args:
    key: PRNGKey controlling the randomness.
    images: A batch of images with shape [batch, height, width, channels].
    hps: An HParam object, we use hps.crop_num_pixels to determine the shift
      amount.

  Returns:
    A batch of cropped images.
  """
  vcrop = vmap(_crop, (0, 0, None), 0)
  key = random.split(key, images.shape[0])
  return vcrop(key, images, hps)


def mixup_tf(key, inputs, targets, alpha=0.1):
  """Perform mixup https://arxiv.org/abs/1710.09412.

  NOTE: Code taken from https://github.com/google/big_vision with variables
  renamed to match `mixup` in this file and logic to synchronize globally.

  Args:
    key: The random key to use.
    inputs: inputs to mix.
    targets: targets to mix.
    alpha: the beta/dirichlet concentration parameter, typically 0.1 or 0.2.

  Returns:
    A new key key. A list of mixed *things. A dict of mixed **more_things.
  """
  # NOTE(dsuo): we don't use split because it's not allowed in Graph execution.
  key_a = tf.random.experimental.stateless_fold_in(key, 0)
  key_b = tf.random.experimental.stateless_fold_in(key_a, 0)

  # Compute beta using gamma functions.
  gamma_a = tf.random.stateless_gamma((1,), key_a, alpha)
  gamma_b = tf.random.stateless_gamma((1,), key_b, alpha)
  weight = tf.squeeze(gamma_a / (gamma_a + gamma_b))

  inputs = weight * inputs + (1.0 - weight) * tf.roll(inputs, 1, axis=0)
  targets = weight * targets + (1.0 - weight) * tf.roll(targets, 1, axis=0)

  return inputs, targets


def mixup(key, alpha, images, labels):
  """Mixes images and labels within a single batch.

  Results in a batch with:
    mixed_images[idx] = weight * images[idx] + (1-weight) * images[-(idx+1)].

  Args:
    key: Rng key.
    alpha: Used to control the beta distribution that weight is sampled from.
    images: Array of shape [batch, height, width, channels]
    labels: Array fo shape [batch, num_classes]

  Returns:
    Tuple (mixed_images, mixed_labels).
  """
  assert len(labels.shape) == 2, 'Mixup requires one hot targets.'
  image_format = 'NHWC'
  batch_size = labels.shape[0]

  weight = random.beta(key, alpha, alpha) * jnp.ones((batch_size, 1))
  mixed_labels = weight * labels + (1.0 - weight) * labels[::-1]

  weight = jnp.reshape(weight, (1,) * 0 + (batch_size,) + (1,) * (3))
  reverse = tuple(slice(images.shape[i]) if d != 'N' else slice(-1, None, -1)
                  for i, d in enumerate(image_format))

  mixed_images = weight * images + (1.0 - weight) * images[reverse]

  return mixed_images, mixed_labels


def augment_cifar10(key, images, labels, hps):
  """Applies a random flip, crop and mixup to the image.

  Args:
    key: Rng key.
    images: A batch of images with shape [batch, height, width, channels].
    labels: A batch of labels with shape [batch, ...]
    hps: HParams object. hps.alpha parameterizes
      the beta distribution sampling the mixup probabilities.
      hps.crop_num_pixels determines the max amount of pixels for which the
      viewing window will be shifted. hps.flip_probability determines the
      probability of applying a random flip. hps.use_mixup determines whether
      or not mixup is applied.

  Returns:
    A tuple containing the augmented images and labels.
  """
  mixup_rng, flip_rng, crop_rng = random.split(key, 3)
  assert images.shape[1:] == (32, 32, 3)

  batch_size = labels.shape[0]

  # Random flip
  flip_mask = random.uniform(flip_rng, (1,) * (-1) + (batch_size,) + (1,) * (3))
  images = jnp.where(
      flip_mask < hps.flip_probability, images[tuple(
          slice(images.shape[i]) if d != 'W' else slice(-1, None, -1)
          for i, d in enumerate('NHWC'))], images)

  images = crop(crop_rng, images, hps)

  if hps.use_mixup:
    images, labels = mixup(mixup_rng, hps.alpha, images, labels)
  return images, labels


def identity_augment(key, images, labels):
  del key
  return images, labels
