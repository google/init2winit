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

r"""JAX-based Python script for defining a workload for training a Transformer model on the WikiText-103 dataset.

For the config dictionary is used to specify the training parameters,
please refer to base_workload.py.
"""

import datetime
import math
from typing import Dict, Tuple

from absl import logging
from flax.training import train_state
from init2winit.model_lib import losses
from init2winit.model_lib import transformer_lm
from init2winit.projects.optlrschedule.workload import base_workload
from init2winit.projects.optlrschedule.workload import optimizers
from init2winit.projects.optlrschedule.workload.datasets import wikitext_103
import jax
from jax import numpy as jnp
import numpy as np


PAD_ID = wikitext_103.PAD_ID
VOCAB_SIZE = wikitext_103.SPM_TOKENIZER_VOCAB_SIZE
TransformerLM = transformer_lm.TransformerLM
BaseWorkload = base_workload.BaseWorkload

weighted_cross_entropy_loss = losses.get_loss_fn('cross_entropy')


def maybe_pad_batch(batch, batch_size):
  """Pad a batch of data to a fixed size.

  Args:
      batch: A batch of data to be padded.
      batch_size: The desired size of the padded batch.

  Returns:
      batch: The padded batch of data.
  """
  if len(batch) != batch_size:
    batch = np.pad(
        batch,
        ((0, batch_size - len(batch)), (0, 0)),
        'constant',
        constant_values=PAD_ID,
    )
  return batch


def get_weights(x: np.ndarray) -> np.ndarray:
  """Gets per token weights.

  Assigns 0 weight to padding tokens and 1 weight to all other tokens.

  Args:
    x: The input array.

  Returns:
    The per token weights.
  """
  return np.where(np.array(x) != PAD_ID, 1.0, 0.0)


def _loss_fn(
    params,
    apply_fn,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    dropout_rng: jax.Array,
) -> jnp.ndarray:
  """Compute the mean-reduced cross entropy loss for the current batch for training.

  Weighs the loss for each token based on the weights array of the batch.

  Args:
      params: Model parameters.
      apply_fn: Model apply function.
      batch: A tuple containing a batch of images and corresponding labels.
      dropout_rng: JAX random key for dropout.

  Returns:
      jnp.ndarray: The sum of unnormalized cross entropy losses over all tokens.
  """
  inputs, targets, weights = batch
  logits = apply_fn(
      {'params': params},
      inputs,
      train=True,
      rngs={'dropout': dropout_rng},
  )
  # Apply one-hot encoding to targets
  one_hot_targets = jax.nn.one_hot(targets, VOCAB_SIZE)
  # returns sum of unnormalized cross entropy loss and number of tokens
  loss, num_tokens = weighted_cross_entropy_loss(
      logits, one_hot_targets, weights
  )

  return loss / num_tokens


def _train_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    lr: float,
    dropout_rng: jax.Array,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
  """Perform a single training step.

  Args:
      state (train_state.TrainState): The current training state containing
        model parameters and optimizer.
      batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing a batch of
        images and corresponding labels.
      lr (float): The learning rate for this step.
      dropout_rng (jax.Array): JAX random key for dropout.

  Returns:
      Tuple[train_state.TrainState, jnp.ndarray]:
          - Updated training state after applying gradients.
          - The computed loss for the current batch.
  """
  dropout_train_rng = jax.random.fold_in(key=dropout_rng, data=state.step)
  loss, grads = jax.value_and_grad(_loss_fn, argnums=0)(
      state.params, state.apply_fn, batch, dropout_train_rng
  )
  state.opt_state.hyperparams['learning_rate'] = lr  # pytype: disable=attribute-error
  state = state.apply_gradients(grads=grads)
  return state, loss


def _evaluate_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Single evaluation step for one batch.

  Args:
      state (train_state.TrainState): The current training state containing
        model parameters and optimizer.
      batch (Tuple[jnp.ndarray, jnp.ndarray, int]): A tuple containing a batch
        of images and corresponding labels.

  Returns:
      Tuple of (loss, error, num_sample)
  """
  inputs, targets, weights = batch
  logits = state.apply_fn({'params': state.params}, inputs, train=False)

  # Apply one-hot encoding to targets
  one_hot_targets = jax.nn.one_hot(targets, VOCAB_SIZE)
  loss, num_tokens = weighted_cross_entropy_loss(
      logits, one_hot_targets, weights
  )

  return loss, num_tokens


class Wikitext103Transformer(BaseWorkload):
  """Wikitext-103 Transformer training workload."""

  def __init__(self, config: base_workload.ConfigType) -> None:
    """Initialize Wikitext-103 Transformer training workload."""
    super().__init__(config)

    # vmapped train step
    self.vmapped_train_step = jax.vmap(_train_step, in_axes=(0, None, 0, None))

    # jitted vmapped train step
    self.jitted_vmapped_train_step = jax.jit(
        self.vmapped_train_step,
        in_shardings=(
            self.replicated_sharding,
            self.sharding,
            self.replicated_sharding,
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
    else:
      raise ValueError(
          f"Unknown compute option: {self.config['compute_option']}"
      )

    # Set vocab size
    self.vocab_size = VOCAB_SIZE

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

    Generate dummy data to mimic the WikiText-103 dataset:
    4 training samples, 4 test samples

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
          - x_train: Training inputs (NumPy array)
          - y_train: Training targets (NumPy array)
          - x_test: Test inputs (NumPy array)
          - y_test: Test targets (NumPy array)
    """
    print('use dummy data')

    num_train_samples = 64
    num_validation_samples = 64
    sequence_length = 128

    # Create dummy training and test datasets
    x_train = np.random.randint(
        0,
        self.vocab_size,
        size=(num_train_samples, sequence_length),
    )
    y_train = x_train

    x_test = np.random.randint(
        0,
        self.vocab_size,
        size=(num_validation_samples, sequence_length),
    )
    y_test = x_test

    return x_train, y_train, x_test, y_test

  def _load_data(
      self,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess WikiText-103 dataset.

    For each split, the input and target arrays are identical arrays of shape
    (num_sequences, sequence_length).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - Training sequence inputs (NumPy array)
            - Training sequence inputs(NumPy array)
            - Test sequence targets (NumPy array)
            - Test sequence targets (NumPy array)
    """

    train_dataset, validation_dataset, _ = (
        wikitext_103.get_wikitext103_dataset()
    )

    return (
        train_dataset['inputs'].astype(jnp.float32),
        train_dataset['targets'].astype(jnp.float32),
        validation_dataset['inputs'].astype(jnp.float32),
        validation_dataset['targets'].astype(jnp.float32),
    )

  def create_train_state(
      self,
      init_param_rng: jax.Array,
  ) -> train_state.TrainState:
    """Create the initial training state with the model and optimizer.

    Args:
        init_param_rng (jax.Array): JAX random key for initializing model
          parameters.

    Returns:
        train_state.TrainState: Training state containing the initialized model
        and optimizer.
    """
    model_dtype = 'float32'
    params_rng, dropout_rng = jax.random.split(init_param_rng, 2)

    model_config = self.config['model_config']
    transformer = TransformerLM(
        vocab_size=VOCAB_SIZE,
        emb_dim=model_config['emb_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        qkv_dim=model_config['qkv_dim'],
        mlp_dim=model_config['mlp_dim'],
        max_len=model_config['max_len'],
        dropout_rate=model_config['dropout_rate'],
        model_dtype=model_dtype,
    )

    # Init model
    fake_input_batch = [jnp.ones((1, 128), model_dtype)]
    variables = transformer.init(
        {'params': params_rng, 'dropout': dropout_rng},
        *fake_input_batch,
        train=False,
    )

    tx = optimizers.get_optimizer_from_config(self.config)

    state = train_state.TrainState.create(
        apply_fn=transformer.apply, params=variables['params'], tx=tx
    )
    return state

  def evaluate_models(
      self,
      state: train_state.TrainState,
      x: jnp.ndarray,
      y: jnp.ndarray,
  ) -> dict[str, jnp.ndarray]:
    """Evaluates multiple models on a dataset split using JAX arrays.

    Args:
        state: Train state where each element in tree_leaves(state.params)
          returns the model is of shape (num_models, ...).
        x: Input sequences
        y: Target sequences

    Returns:
        dict[str, jnp.ndarray]: A dictionary containing the following keys and
        values.
          - The computed loss for the current batch.
          - The computed perplexities for the current batch.
    """
    batch_size = self.config['batch_size']
    per_host_batch_size = batch_size // jax.process_count()

    num_train_batches = int(np.ceil(self.x_train.shape[0] / batch_size))

    # Get number of models from the first parameter's shape
    # num_models = states.params['Conv_0']['kernel'].shape[0]
    first_param = jax.tree_util.tree_leaves(state.params)[0]
    num_models = first_param.shape[0]

    # Initialize metrics for all models
    total_losses = jnp.zeros(num_models)
    num_tokens = 0

    # Evaluate entire dataset batch by batch
    for i in range(0, num_train_batches * batch_size, batch_size):
      # Get batches of data
      batch_inputs = x[i : i + batch_size]
      batch_targets = y[i : i + batch_size]

      # Pad batches if necessary and get weights
      batch_inputs = maybe_pad_batch(batch_inputs, batch_size)
      batch_targets = maybe_pad_batch(batch_targets, batch_size)
      batch_weights = get_weights(batch_inputs)

      # Convert the numpy arrays to jax global arrays with process-local
      # sharding on multi-host runs, since "passing non-trivial shardings
      # for numpy inputs is not allowed".
      start = jax.process_index() * per_host_batch_size
      end = start + per_host_batch_size
      batch_inputs = self.make_global_array(batch_inputs[start:end])
      batch_targets = self.make_global_array(batch_targets[start:end])
      batch_weights = self.make_global_array(batch_weights[start:end])

      batch = (batch_inputs, batch_targets, batch_weights)
      batch_sum_loss, batch_num_tokens = self.evaluate_step(state, batch)

      total_losses += batch_sum_loss
      num_tokens += batch_num_tokens

    # Calculate final average metrics
    results = {}
    results['losses'] = total_losses / num_tokens
    results['perplexities'] = jnp.exp(total_losses / num_tokens)

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
      metric_names = ['losses', 'perplexities']
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
      train_eval_fraction: float = 0.1,
  ) -> Tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Train the model using a specific schedule.

    Args:
        schedules (np.ndarray): Array of learning rate schedule.
        params_rngs (np.ndarray): Array of PRNGs for different model
          initializations
        data_rng (jax.Array): JAX random key for randomness during training.
        train_eval_fraction (float): Fraction of training sequences to evaluate.
          The train sequences are included from the unshuffled training
          sequences.

    Returns:
        Tuple[
          train_state.TrainState,  # vmap_states: Contains model states after
          training.
          dict[str, jnp.ndarray] : A dictionary containing the results keys and
          values.
      ]:


    results: A dictionary containing the following keys and values.
      final/train_perplexities: The final error rate of the model on the
      test dataset.
      final/train_losses: The final loss of the model on the
      training dataset.
      best/train_losses: The average loss of the model on the training
        dataset for the best step.
      best/train_perplexities: The average error of the model on the training
        dataset for the best step.
      best/train_perplexities_step: The index of the best step regarding the
      loss.
      best/train_perplexities_step: The index of the best step regarding the
      error.

      final/test_perplexities: The final error rate of the model on the
      test dataset.
      final/test_losses: The final loss of the model on the
      training dataset.
      best/test_losses: The average loss of the model on the training
        dataset for the best step.
      best/test_perplexities: The average error of the model on the training
        dataset for the best step.
      best/test_losses_step: The index of the best step regarding the loss.
      best/test_perplexities_step: The index of the best step regarding the
      error
    """
    # Get dropout rng
    data_rng, dropout_rng = jax.random.split(data_rng, 2)

    batch_size = self.config['batch_size']
    per_host_batch_size = batch_size // jax.process_count()

    # num schedule should be match num of param_rngs
    num_schedule = schedules.shape[0]
    assert num_schedule == params_rngs.shape[0]

    # Create models with different initializations
    vmap_create_state = jax.vmap(self.create_train_state)
    vmap_states = vmap_create_state(params_rngs)

    num_training_steps = len(schedules[0])

    steps_per_epoch = math.floor(self.x_train.shape[0] / batch_size) + 1
    num_epochs = math.ceil(num_training_steps / steps_per_epoch)

    returns_results = self.init_eval_results_vmap(num_schedule)
    global_step = 0

    for epoch in range(num_epochs):
      epoch_key = jax.random.fold_in(data_rng, epoch)

      # Shuffle data for each epoch
      permutation = jax.random.permutation(epoch_key, self.x_train.shape[0])
      x_train_shuffled = self.x_train[permutation]
      y_train_shuffled = self.y_train[permutation]

      for i in range(0, self.x_train.shape[0], batch_size):
        if global_step >= self.config['total_steps']:
          break

        batch_inputs = x_train_shuffled[i : i + batch_size]
        batch_targets = y_train_shuffled[i : i + batch_size]

        # Pad batches if necessary and get weights
        batch_inputs = maybe_pad_batch(batch_inputs, batch_size)
        batch_targets = maybe_pad_batch(batch_targets, batch_size)
        batch_weights = get_weights(batch_inputs)

        # Convert the numpy arrays to jax global arrays with process-local
        # sharding on multi-host runs, since "passing non-trivial shardings
        # for numpy inputs is not allowed".
        start = jax.process_index() * per_host_batch_size
        end = start + per_host_batch_size
        batch_inputs = self.make_global_array(batch_inputs[start:end])
        batch_targets = self.make_global_array(batch_targets[start:end])
        batch_weights = self.make_global_array(batch_weights[start:end])

        batch = (batch_inputs, batch_targets, batch_weights)

        vmap_states, loss = self.train_step(
            vmap_states, batch, schedules[:, global_step], dropout_rng
        )
        # TODO(kasimbeg): remove this after debugging
        if global_step % 100 == 0:
          current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          logging.info(
              '%s: global_step: %d --- Loss: %s',
              current_time,
              global_step,
              loss,
          )

        # Evaluate based on configuration
        if self.should_evaluate(global_step, epoch):

          for split in ['train', 'test']:
            if split == 'train':
              try:
                num_train_eval_examples = int(
                    train_eval_fraction * self.x_train.shape[0]
                )
              except Exception as e:
                print(type(train_eval_fraction), type(self.x_train.shape[0]))
                raise e
              results = self.evaluate_models(
                  vmap_states,
                  self.x_train[:num_train_eval_examples],
                  self.y_train[:num_train_eval_examples],
              )
            else:
              results = self.evaluate_models(
                  vmap_states, self.x_test, self.y_test
              )

            for metric_name, metric_value in results.items():
              key = f'{split}_{metric_name}'
              returns_results[f'final/{key}'] = metric_value
              try:
                returns_results[f'best/{key}'] = jnp.minimum(
                    metric_value, returns_results[f'best/{key}']
                )
              except Exception as e:
                print(type(metric_value), type(returns_results[f'best/{key}']))
                raise e

              returns_results[f'best/{key}_step'] = jnp.where(
                  metric_value <= returns_results[f'best/{key}'],
                  global_step,
                  returns_results[f'best/{key}_step'],
              )

        global_step += 1

    return vmap_states, returns_results
