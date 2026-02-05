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

"""Implements checkpointing of nested python containers of numpy arrays.

This is useful for training neural networks with stax, where model parameters
are nested numpy arrays.
"""
from absl import flags
from absl import logging
import jax
# pylint: disable=g-importing-member
from jax.experimental.multihost_utils import process_allgather
import orbax.checkpoint as ocp

FLAGS = flags.FLAGS


def load_pytree(pytree_file, orbax_checkpoint_manager=None):
  """Loads a checkpointed pytree."""
  if not orbax_checkpoint_manager:
    orbax_checkpoint_manager = ocp.CheckpointManager(pytree_file)
  latest = load_latest_checkpoint(
      target=None, orbax_checkpoint_manager=orbax_checkpoint_manager
  )
  if latest:
    # Because we pass target=None, flax checkpointing will return the raw
    # state dict, where 'state' will be a dict with keys ['0', '1', ...]
    # instead of a list.
    return latest['pytree']
  return None


def maybe_restore_checkpoint(
    unreplicated_optimizer_state,
    unreplicated_params,
    unreplicated_batch_stats,
    unreplicated_training_metrics_state,
    orbax_checkpoint_manager=None,
    orbax_checkpoint_manager_external=None):
  """Optionally restores from a checkpoint.

  The checkpoint logic is as follows: if `orbax_checkpoint_manager` contains
  a latest checkpoint, restore it. Otherwise, don't restore any checkpoint,
  and just return the passed-in optimizer_state, params, batch_stats, and
  metrics_grabber.

  Args:
    unreplicated_optimizer_state: unreplicated optimizer state
    unreplicated_params: unreplicated params
    unreplicated_batch_stats: unreplicated batch stats
    unreplicated_training_metrics_state: unreplicated metrics state
    orbax_checkpoint_manager: orbax.CheckpointManager
    orbax_checkpoint_manager_external: orbax.CheckpointManager

  Returns:
    unreplicated_optimizer_state
    unreplicated_params
    unreplicated_batch_stats
    unreplicated_training_metrics_state
    global_step (int)
    sum_train_cost (float)
    preemption_count (int)
    is_restored (bool): True if we've restored the latest checkpoint
                        in train_dir.
  """
  uninitialized_global_step = -1
  unreplicated_checkpoint_state = dict(
      params=unreplicated_params,
      optimizer_state=unreplicated_optimizer_state,
      batch_stats=unreplicated_batch_stats,
      training_metrics_grabber=unreplicated_training_metrics_state,
      global_step=uninitialized_global_step,
      preemption_count=0,
      sum_train_cost=0.0,
  )
  logging.info('Loading latest checkpoint')
  latest_ckpt = load_latest_checkpoint(
      target=unreplicated_checkpoint_state,
      orbax_checkpoint_manager=orbax_checkpoint_manager,
  )
  logging.info('Loading checkpoint from complete.')
  # Load_latest_checkpoint() will return unreplicated_checkpoint_state if
  # train_dir does not exist or if it exists and contains no checkpoints.
  # Note that we could likely change the below line to:
  # found_checkpoint = latest_ckpt != unreplicated_checkpoint_state
  found_checkpoint = (latest_ckpt['global_step'] != uninitialized_global_step)

  # If there's a latest checkpoint in the train_dir, restore from that.
  if found_checkpoint:
    ckpt_to_return = latest_ckpt
    is_restored = True  # We do want trainer to increment preemption_count.
    logging.info(
        'Restoring checkpoint from ckpt_%d', latest_ckpt['global_step']
    )
  elif not found_checkpoint and orbax_checkpoint_manager_external:
    logging.info('Restoring checkpoint from external checkpoint.')
    ckpt_to_return = load_latest_checkpoint(
        target=unreplicated_checkpoint_state,
        orbax_checkpoint_manager=orbax_checkpoint_manager_external,
    )
    is_restored = False  # We don't want trainer to increment preemption_count.

    # Handle failure to load from external_checkpoint_path.
    if ckpt_to_return['global_step'] == -1:
      return (
          unreplicated_optimizer_state,
          unreplicated_params,
          unreplicated_batch_stats,
          unreplicated_training_metrics_state,
          0,  # global_step
          0,  # sum_train_cost
          0,  # preemption_count
          False)  # is_restored
  else:  # Else, don't restore from any checkpoint.
    return (
        unreplicated_optimizer_state,
        unreplicated_params,
        unreplicated_batch_stats,
        unreplicated_training_metrics_state,
        0,  # global_step
        0,  # sum_train_cost
        0,  # preemption_count
        False)  # is_restored

  return (
      ckpt_to_return['optimizer_state'],
      ckpt_to_return['params'],
      ckpt_to_return['batch_stats'],
      ckpt_to_return['training_metrics_grabber'],
      ckpt_to_return['global_step'],  # global_step
      ckpt_to_return['sum_train_cost'],
      ckpt_to_return['preemption_count'],  # preemption_count
      is_restored)  # is_restored


def unreplicate_and_save_checkpoint(
    optimizer_state,
    params,
    batch_stats,
    training_metrics_state,
    global_step,
    preemption_count,
    sum_train_cost,
    orbax_checkpoint_manager):
  """Saves pytree, step, preemption_count, and sum_train_cost to train_dir."""
  logging.info('Saving checkpoint to ckpt_%d', global_step)
  # jax.device_get doesn't work if jax.Array lives on multiple hosts.
  # So we first all_gather it to the host and then call jax.device_get
  if jax.process_count() > 1:
    unreplicated_optimizer_state = jax.device_get(
        process_allgather(optimizer_state, tiled=True))
    unreplicated_params = jax.device_get(process_allgather(params, tiled=True))
  else:
    unreplicated_optimizer_state = jax.device_get(optimizer_state)
    unreplicated_params = jax.device_get(params)
  unreplicated_batch_stats = jax.device_get(batch_stats)
  unreplicated_training_metrics_state = jax.device_get(
      training_metrics_state)
  unreplicated_sum_train_cost = jax.device_get(sum_train_cost)
  state = dict(global_step=global_step,
               preemption_count=preemption_count,
               sum_train_cost=unreplicated_sum_train_cost,
               optimizer_state=unreplicated_optimizer_state,
               params=unreplicated_params,
               batch_stats=unreplicated_batch_stats,
               training_metrics_grabber=unreplicated_training_metrics_state)
  save_checkpoint(global_step,
                  state,
                  orbax_checkpoint_manager=orbax_checkpoint_manager)
  logging.info('Done saving checkpoint.')


def save_checkpoint(step,
                    state,
                    orbax_checkpoint_manager):
  """Saves checkpoint to train_dir.

  A list of checkpoints will be stored in train_dir/step.
  If the step folder already exists, the checkpoint will not be saved and a
  warning will be logged.

  Args:
    step: (int) Step of the checkpoint.
    state: (pytree)The state to save.
    orbax_checkpoint_manager: orbax.CheckpointManager

  Returns:
    The path of the checkpoint directory.
  """
  saved = orbax_checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
  if not saved:
    logging.warning(
        'Checkpoint at step %d was not saved! Perhaps it already exists?', step
    )
  return orbax_checkpoint_manager.directory


def load_checkpoint(
    checkpoint_path=None,
    target=None,
    step=0,
    orbax_checkpoint_manager=None,
):
  """Loads the specified checkpoint."""
  # for backwards compatibility
  if checkpoint_path and not orbax_checkpoint_manager:
    orbax_checkpoint_manager = ocp.CheckpointManager(checkpoint_path)
  restored = orbax_checkpoint_manager.restore(
      step,
      args=ocp.args.StandardRestore(target),
  )
  return restored


def load_latest_checkpoint(target=None, orbax_checkpoint_manager=None):
  """Loads the most recent checkpoint listed in train_dir.

  Args:
    target: used for checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    orbax_checkpoint_manager: An orbax.CheckpointManager instance.
  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67. If
    the directory doesn't exist, it will return the original target.
  """
  restore_step = orbax_checkpoint_manager.latest_step()
  try:
    restored = orbax_checkpoint_manager.restore(
        restore_step, args=ocp.args.StandardRestore(target)
    )
    return restored
  except FileNotFoundError:
    return target
