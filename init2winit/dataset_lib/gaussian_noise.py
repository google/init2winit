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

"""Data generators for init2winit.
"""

import itertools
from init2winit.dataset_lib import data_utils
import jax.numpy as jnp
import jax.random
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf


GAUSSIAN_HPARAMS = config_dict.ConfigDict(
    dict(
        rngs=[12345, 23456],
        model_rngs=[1, 2],
        train_size=2**13,
        train_sizes=[2**13, 2**9],
        rates=[0.5, 0.5],
        locs=[0.0, 0.0],
        scales=[1.0, 1.0],
        sample_seed=123,
        shuffle_seed=246,
        input_shape=(100,),  # This determines the dimension.
        output_shape=(1,),
    ))

METADATA = {
    "apply_one_hot_in_loss": False,
}


class GaussianLinearModel():
  """Gaussian weights sampled."""

  def __init__(self, rng_seed, loc, scale, input_shape, output_shape):
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.loc, self.scale = loc, scale
    self.rng = np.random.RandomState(seed=rng_seed)
    self.params = {
        "w": self.rng.normal(loc=loc, scale=scale, size=(input_shape)),
        "b": self.rng.normal(loc=loc, scale=scale, size=(output_shape))
    }

  def forward(self, dataset):
    new_shape = list(dataset.shape)
    new_shape[-1] = 1
    return jnp.reshape(jnp.dot(dataset, self.params["w"]) + self.params["b"],
                       new_shape)


def create_sampled_data(linear_models, train_rngs, hps):
  """Get sampled datasets of gaussians."""
  train_datasets = []
  for lm, train_rng, loc, scale, train_size in zip(
      linear_models, train_rngs, hps.locs,
      hps.scales, hps.train_sizes):

    # Generate training inputs and targets.
    train_data = train_rng.normal(
        loc=loc, scale=scale, size=(train_size, *hps.input_shape))
    train_targets = lm.forward(train_data)
    train_datasets.append(tf.data.Dataset.from_tensor_slices(
        {"inputs": train_data,
         "targets": train_targets}))

  # Combine datasets through sampling.and then batch.
  train_ds = data_utils.get_sampled_dataset(
      train_datasets, hps.rates,
      is_training=True, sample_seed=hps.sample_seed,
      shuffle_seed=hps.shuffle_seed)

  # Shard dataset across hosts.
  index = jax.process_index()
  num_hosts = jax.process_count()
  train_ds = train_ds.shard(num_hosts, index)
  return train_ds


def get_gaussian_noise(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Returns the noise seed for the nqm model.

  NOTE: This dataset is only meant to be used with the nqm model.
  This just generates isotropic Gaussian noise of the desired dimension.
  The nqm model will then multiple this noise by a matrix D, with the properly
  that D^T D = C. This yields noise with gradient covariance C.

  Args:
    shuffle_rng: Not used.
    batch_size: The global train batch size, used to determine the batch size
      yielded from train_epoch().
    eval_batch_size: Not used.
    hps: Hparams object. We only refer to hps.input_shape to determine the
      dimension of the noise.
  Returns:
    train_epoch, eval_train_epoch, valid_epoch, test_epoch: three generators.
      Only train_epoch is used.
  """
  del eval_batch_size
  del shuffle_rng

  per_host_batch_size = batch_size // jax.process_count()

  n_devices = jax.local_device_count()
  if per_host_batch_size % n_devices != 0:
    raise ValueError("n_devices={} must divide per_host_batch_size={}.".format(
        n_devices, per_host_batch_size))

  # Create raw data.
  train_rngs = [np.random.RandomState(seed=rng) for rng in hps.rngs]
  eval_train_rngs = [np.random.RandomState(seed=rng) for rng in hps.rngs]

  # Wrap in tf.data.Dataset and batch.
  linear_models = []
  for model_rng, loc, scale in zip(hps.model_rngs, hps.locs, hps.scales):
    # Create linear model with guassian sampled weights.
    lm = GaussianLinearModel(model_rng, loc, scale,
                             hps.input_shape, hps.output_shape)
    linear_models.append(lm)

  # Create sampled dataset.
  train_ds = create_sampled_data(linear_models, train_rngs,
                                 hps).batch(per_host_batch_size)
  eval_train_ds = create_sampled_data(linear_models, eval_train_rngs,
                                      hps).batch(per_host_batch_size)

  def train_iterator_fn():
    for batch in iter(train_ds):
      numpy_batch = data_utils.tf_to_numpy(batch)
      yield {
          "inputs": numpy_batch["inputs"],
          "targets": numpy_batch["targets"]
      }

  def eval_train_epoch(num_batches):
    eval_train_iter = iter(eval_train_ds)
    for batch in itertools.islice(eval_train_iter, num_batches):
      numpy_batch = data_utils.tf_to_numpy(batch)
      yield {
          "inputs": numpy_batch["inputs"],
          "targets": numpy_batch["targets"]
      }

  # pylint: disable=unreachable
  def valid_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  # pylint: disable=unreachable
  def test_epoch(*args, **kwargs):
    del args
    del kwargs
    return
    yield  # This yield is needed to make this a valid (null) iterator.
  # pylint: enable=unreachable

  return data_utils.Dataset(train_iterator_fn, eval_train_epoch,
                            valid_epoch, test_epoch)
