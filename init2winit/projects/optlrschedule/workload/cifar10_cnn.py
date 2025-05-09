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

r"""JAX-based Python script for defining a cifar10-cnn workload structure.

This Python file defines an class of Cifar10 Training on CNN,
which outlines the structure for training a machine learning model
using JAX and Flax libraries.

For the config dictionary is used to specify the training parameters,
please refer to base_workload.py.
"""

import math
from typing import Dict, Tuple

from absl import logging
from flax import linen as nn
from flax.training import train_state
from init2winit.projects.optlrschedule.workload import base_workload
from init2winit.projects.optlrschedule.workload import optimizers
import jax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds


class CNN(nn.Module):
  """A simple CNN model for CIFAR-10 classification."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply the CNN layers to input data.

    Args:
        x (jnp.ndarray): Input image batch of shape (batch_size, height, width,
          channels).

    Returns:
        jnp.ndarray: The logits for each class.
    """
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


def _loss_fn(
    params,
    apply_fn,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
  """Compute the loss for the current batch.

  Args:
      params: Model parameters.
      apply_fn: Model apply function.
      batch: A tuple containing a batch of images and corresponding labels.

  Returns:
      jnp.ndarray: The loss value computed using softmax cross-entropy.
  """
  images, labels = batch
  logits = apply_fn(params, images)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
  return loss


def _train_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    lr: float,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
  """Perform a single training step.

  Args:
      state (train_state.TrainState): The current training state containing
        model parameters and optimizer.
      batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing a batch of
        images and corresponding labels.
      lr (float): The learning rate for this step.

  Returns:
      Tuple[train_state.TrainState, jnp.ndarray]:
          - Updated training state after applying gradients.
          - The computed loss for the current batch.
  """
  loss, grads = jax.value_and_grad(_loss_fn, argnums=0)(
      state.params, state.apply_fn, batch
  )
  state.opt_state.hyperparams['learning_rate'] = lr  # pytype: disable=attribute-error
  state = state.apply_gradients(grads=grads)
  return state, loss


def _pad_batch(
    batch_x: jnp.ndarray, batch_y: jnp.ndarray, padded_batch_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Pad a batch of data to a fixed size.

  Args:
      batch_x: The input image batch.
      batch_y: The input label batch.
      padded_batch_size: The desired padded batch size.

  Returns:
      Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
          - The padded image batch.
          - The padded label batch.
          - A mask indicating which elements are valid in the padded batch.
  """
  pad_size = padded_batch_size - batch_x.shape[0]
  x = jnp.pad(batch_x, ((0, pad_size),) + ((0, 0),) * (batch_x.ndim - 1))
  y = jnp.pad(batch_y, ((0, pad_size),) + ((0, 0),) * (batch_y.ndim - 1))
  mask = jnp.arange(padded_batch_size) < batch_x.shape[0]
  return x, y, mask


def _evaluate_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Single evaluation step with padding support.

  Args:
      state (train_state.TrainState): The current training state.
      batch (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): A tuple containing:

  - batch_x: A batch of images
  - batch_y: Corresponding labels
  - mask: A boolean array of shape (batch_size,) indicating which elements
    in the
        batch are real data (True) versus padding (False). This mask is used to
        ensure padded elements don't contribute to the loss or metrics. The mask
        should be of dtype jnp.bool_. When working with where= parameter in
        optax.softmax_cross_entropy_with_integer_labels, True values keep the
        corresponding elements while False values exclude them from
        computations.

  Returns:
      Tuple of (loss, error, num_samples).
  """
  batch_x, batch_y, mask = batch

  # Forward pass
  logits = state.apply_fn(state.params, batch_x)

  # Compute loss
  losses = optax.softmax_cross_entropy_with_integer_labels(
      logits, batch_y, where=mask[:, None]
  )
  sum_loss = jnp.sum(losses, where=mask)

  # Compute errors (using original mask)
  predicted_classes = jnp.argmax(logits, axis=-1)
  sum_error = jnp.sum(predicted_classes != batch_y, where=mask)
  num_sample = jnp.sum(mask)

  return sum_loss, sum_error, num_sample


class Cifar10Training(base_workload.BaseWorkload):
  """CIFAR-10 training workload."""

  def __init__(self, config: base_workload.ConfigType) -> None:
    """Initialize CIFAR-10 training workload."""
    super().__init__(config)

    # vmapped train step
    self.vmapped_train_step = jax.vmap(_train_step, in_axes=(0, None, 0))

    # jitted vmapped train step
    self.jitted_vmapped_train_step = jax.jit(
        self.vmapped_train_step,
        in_shardings=(
            self.replicated_sharding,
            self.sharding,
            self.replicated_sharding,
        ),
        donate_argnames=('state',),
    )

    self.vmapped_evaluate_step = jax.vmap(_evaluate_step, in_axes=(0, None))

    self.jitted_vmapped_evaluate_step = jax.jit(
        self.vmapped_evaluate_step,
        in_shardings=(
            self.replicated_sharding,
            self.sharding,
        ),
    )

    if self.config['compute_option'] == 'jit(vmap)':
      print('compute_option = jit(vmap)')
      self.train_step = self.jitted_vmapped_train_step
      self.evaluate_step = self.jitted_vmapped_evaluate_step
    elif self.config['compute_option'] == 'vanilla':
      print('compute_option = vanilla')
      self.train_step = _train_step
      self.evaluate_step = _evaluate_step
    else:
      raise ValueError(
          f"Unknown compute option: {self.config['compute_option']}"
      )

    # load dataset
    if self.config['use_dummy_data']:
      self.x_train, self.y_train, self.x_test, self.y_test = (
          self._load_dummy_data()
      )
    else:
      self.x_train, self.y_train, self.x_test, self.y_test = self._load_data()

  def _load_dummy_data(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess dummy dataset.

    Generate dummy data to mimic the CIFAR-10 dataset structure CIFAR-10 has
    32x32 images with 3 channels (RGB), and 10 classes for labels Define the
    shape of the dummy data: 4 training samples, 4 test samples

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
          - x_train: Training images (NumPy array)
          - y_train: Training labels (NumPy array)
          - x_test: Test images (NumPy array)
          - y_test: Test labels (NumPy array)
    """
    print('use dummy data')

    num_train_samples = 64
    num_test_samples = 64
    img_height, img_width, num_channels = 32, 32, 3
    num_classes = 10

    # Create dummy training and test datasets
    x_train = np.random.randint(
        0,
        256,
        size=(num_train_samples, img_height, img_width, num_channels),
        dtype=np.uint8,
    )
    y_train = np.random.randint(
        0, num_classes, size=(num_train_samples,), dtype=np.int32
    )

    x_test = np.random.randint(
        0,
        256,
        size=(num_test_samples, img_height, img_width, num_channels),
        dtype=np.uint8,
    )
    y_test = np.random.randint(
        0, num_classes, size=(num_test_samples,), dtype=np.int32
    )

    # Normalize images to the range [0, 1]
    x_train = x_train.astype(jnp.float32) / 255.0
    x_test = x_test.astype(jnp.float32) / 255.0

    return x_train, y_train, x_test, y_test

  def _load_data(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess CIFAR-10 dataset.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - x_train: Training images (NumPy array)
            - y_train: Training labels (NumPy array)
            - x_test: Test images (NumPy array)
            - y_test: Test labels (NumPy array)
    """
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()

    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1)
    )
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    x_train = train_ds['image']
    y_train = train_ds['label']
    x_test = test_ds['image']
    y_test = test_ds['label']

    # Normalize images to the range [0, 1]
    x_train = x_train.astype(jnp.float32) / 255.0
    x_test = x_test.astype(jnp.float32) / 255.0

    return x_train, y_train, x_test, y_test

  def create_train_state(
      self, init_param_rng: jax.Array
  ) -> train_state.TrainState:
    """Create the initial training state with the model and optimizer.

    Args:
        init_param_rng (jax.Array): JAX random key for initializing model
          parameters.

    Returns:
        train_state.TrainState: Training state containing the initialized model
        and optimizer.
    """
    model = CNN()
    params = model.init(init_param_rng, np.ones([1, 32, 32, 3]))

    tx = optimizers.get_optimizer_from_config(self.config)

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return state

  def evaluate_models(
      self,
      states: train_state.TrainState,  # Array of TrainState
      x: jnp.ndarray,
      y: jnp.ndarray,
  ) -> dict[str, jnp.ndarray]:
    """Evaluates multiple models on training set using JAX arrays.

    Args:
        states: Array of model states to evaluate (shape: [num_models])
        x: Training images
        y: Training labels

    Returns:
        dict[str, jnp.ndarray]: A dictionary containing the following keys and
        values.
          - The computed loss for the current batch.
          - The computed error for the current batch.
    """
    if self.config['batch_size'] % jax.process_count() != 0:
      raise ValueError(
          'Batch size must be divisible by the number of hosts (processes).'
      )
    batch_size = self.config['batch_size']
    per_host_batch_size = batch_size // jax.process_count()
    num_eval_batches = int(np.ceil(x.shape[0] / batch_size))

    first_param = jax.tree_util.tree_leaves(states.params)[0]
    num_models = first_param.shape[0]

    # Initialize metrics for all models
    total_losses = jnp.zeros(num_models)
    total_errors = jnp.zeros(num_models)
    total_samples = jnp.zeros(num_models)

    # Evaluate entire dataset batch by batch
    for i in range(0, num_eval_batches * batch_size, batch_size):
      batch_images = x[i : i + batch_size]
      batch_labels = y[i : i + batch_size]
      batch = _pad_batch(batch_images, batch_labels, batch_size)
      start = jax.process_index() * per_host_batch_size
      end = start + per_host_batch_size
      batch = (
          self.make_global_array(batch[0][start:end]),
          self.make_global_array(batch[1][start:end]),
          self.make_global_array(batch[2][start:end]),
      )

      batch_sum_loss, batch_sum_error, batch_num_samples = self.evaluate_step(
          states,
          batch,
      )

      total_losses += batch_sum_loss
      total_errors += batch_sum_error
      total_samples += batch_num_samples

    # Calculate final average metrics
    results = {}
    results['losses'] = total_losses / total_samples
    results['errors'] = total_errors / total_samples

    return results

  def init_eval_results_vmap(
      self,
      num_schedule: int,
  ) -> Dict[str, jnp.ndarray]:
    """Initialize evaluation results dictionary.

    Args:
        num_schedule (int): Number of schedules to evaluate.

    Returns:
        Dict[str, jnp.ndarray]:
          A dictionary of initialized evaluation results.
    """
    results = {}
    for split in ['train', 'test']:
      metric_names = ['losses', 'errors']
      for metric_name in metric_names:
        key = f'{split}_{metric_name}'
        results[f'final/{key}'] = jnp.ones(num_schedule) * np.inf
        results[f'best/{key}'] = jnp.ones(num_schedule) * np.inf
        results[f'best/{key}_step'] = jnp.zeros(num_schedule)
    return results

  def train_and_evaluate_models(
      self,
      schedules: np.ndarray,
      params_rngs: jax.Array,
      data_rng: jax.Array,
  ) -> Tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Train the model using a specific schedule.

    Args:
        schedules (np.ndarray): Array of learning rate schedule.
        params_rngs (np.ndarray): Array of PRNGs for different model
          initializations
        data_rng (jax.Array): JAX random key for randomness during training.

    Returns:
        Tuple[
          train_state.TrainState,  # vmap_states: Array of model states after
          training.
          dict[str, jnp.ndarray] : A dictionary containing the results keys and
          values.
      ]:


    results: A dictionary containing the following keys and values.
      final/train_errors: The final error rate of the model on the test dataset.
      final/train_losses: The final loss of the model on the
      training dataset.
      best/train_losses: The average loss of the model on the training
        dataset for the best step.
      best/train_errors: The average error of the model on the training
        dataset for the best step.
      best/train_losses_step: The index of the best step regarding the loss.
      best/train_errors_step: The index of the best step regarding the error.

      final/test_errors: The final error rate of the model on the test dataset.
      final/test_losses: The final loss of the model on the
      training dataset.
      best/test_losses: The average loss of the model on the training
        dataset for the best step.
      best/test_errors: The average error of the model on the training
        dataset for the best step.
      best/test_losses_step: The index of the best step regarding the loss.
      best/test_errors_step: The index of the best step regarding the error
    """

    # num schedule should be match num of param_rngs
    num_schedule = schedules.shape[0]
    assert num_schedule == params_rngs.shape[0]

    if self.config['batch_size'] % jax.process_count() != 0:
      raise ValueError(
          'Batch size must be divisible by the number of hosts (processes).'
      )

    # Create models with different initializations
    vmap_create_state = jax.vmap(self.create_train_state)
    vmap_states = vmap_create_state(params_rngs)

    num_training_steps = len(schedules[0])
    per_host_batch_size = self.config['batch_size'] // jax.process_count()

    steps_per_epoch = (
        math.floor(self.x_train.shape[0] / self.config['batch_size']) + 1
    )
    num_epochs = math.ceil(num_training_steps / steps_per_epoch)

    returns_results = self.init_eval_results_vmap(num_schedule)
    global_step = 0

    for epoch in range(num_epochs):
      epoch_key = jax.random.fold_in(data_rng, epoch)

      # Shuffle data for each epoch
      permutation = jax.random.permutation(epoch_key, self.x_train.shape[0])
      x_train_shuffled = self.x_train[permutation]
      y_train_shuffled = self.y_train[permutation]

      for i in range(0, self.x_train.shape[0], self.config['batch_size']):
        if global_step >= self.config['total_steps']:
          break

        batch_images = x_train_shuffled[i : i + self.config['batch_size']]
        batch_labels = y_train_shuffled[i : i + self.config['batch_size']]
        # Instead of skipping the last partial batch, we could pad it the way we
        # do in evaluate_models. That would require updating the train_step
        # code.
        if batch_images.shape[0] != self.config['batch_size']:
          logging.info(
              'Skipping last partial batch. Shapes: %s, %s.',
              batch_images.shape,
              batch_labels.shape,
          )
          continue
        start = jax.process_index() * per_host_batch_size
        end = start + per_host_batch_size
        batch = (
            self.make_global_array(batch_images[start:end]),
            self.make_global_array(batch_labels[start:end]),
        )
        vmap_states, _ = self.train_step(
            vmap_states, batch, schedules[:, global_step]
        )

        # Evaluate based on configuration
        if self.should_evaluate(global_step, epoch):

          for split in ['train', 'test']:

            if split == 'train':
              results = self.evaluate_models(
                  vmap_states, self.x_train, self.y_train
              )
            else:
              results = self.evaluate_models(
                  vmap_states, self.x_test, self.y_test
              )

            for metric_name, metric_value in results.items():
              key = f'{split}_{metric_name}'
              returns_results[f'final/{key}'] = metric_value
              returns_results[f'best/{key}'] = jnp.minimum(
                  metric_value, returns_results[f'best/{key}']
              )
              returns_results[f'best/{key}_step'] = jnp.where(
                  metric_value < returns_results[f'best/{key}'],
                  global_step,
                  returns_results[f'best/{key}_step'],
              )

        global_step += 1

    return vmap_states, returns_results
