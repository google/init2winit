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
import os
import sys

from absl import flags
from absl import logging
from flax.training import checkpoints as flax_checkpoints
import jax
# pylint: disable=g-importing-member
from jax.experimental.multihost_utils import process_allgather

FLAGS = flags.FLAGS


def load_pytree(pytree_file, orbax_checkpointer=None):
  """Loads the checkpointed pytree."""
  latest = load_latest_checkpoint(pytree_file,
                                  target=None,
                                  orbax_checkpointer=orbax_checkpointer)
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
    train_dir,
    external_checkpoint_path=None,
    orbax_checkpointer=None):
  """Optionally restores from a checkpoint.

  The checkpoint logic is as follows: if there is a checkpoint in `train_dir`,
  restore it.  Else, if `external_checkpoint_path` is set, restore the
  checkpoint found there.  Else, don't restore any checkpoint, and just
  return the passed-in optimizer_state, params, batch_stats, and
  metrics_grabber.

  Args:
    unreplicated_optimizer_state: unreplicated optimizer state
    unreplicated_params: unreplicated params
    unreplicated_batch_stats: unreplicated batch stats
    unreplicated_training_metrics_state: unreplicated metrics state
    train_dir: (str) The training directory where we will look for a checkpoint.
    external_checkpoint_path: (str) If this argument is set, then we will load
      the external checkpoint stored there.
    orbax_checkpointer: orbax.Checkpointer

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
      sum_train_cost=0.0)
  latest_ckpt = load_latest_checkpoint(train_dir,
                                       target=unreplicated_checkpoint_state,
                                       orbax_checkpointer=orbax_checkpointer)
  # Load_latest_checkpoint() will return unreplicated_checkpoint_state if
  # train_dir does not exist or if it exists and contains no checkpoints.
  # Note that we could likely change the below line to:
  # found_checkpoint = latest_ckpt != unreplicated_checkpoint_state
  found_checkpoint = (latest_ckpt['global_step'] != uninitialized_global_step)

  # If there's a latest checkpoint in the train_dir, restore from that.
  if found_checkpoint:
    ckpt_to_return = latest_ckpt
    is_restored = True  # We do want trainer to increment preemption_count.
    logging.info('Restoring checkpoint from ckpt_%d',
                 latest_ckpt['global_step'])
  # Else, if external_checkpoint_path is non-null, restore from that checkpoint.
  elif external_checkpoint_path is not None:
    # TODO(jeremycohen) This code will crash if we try to load an external
    # checkpoint which was trained with a different num_train_steps.  The issue
    # is that some of the fields in the training metrics state are arrays of
    # shape [num_train_steps].  In the future we may want to handle these
    # arrays explicitly, in order to avoid this crash.
    logging.info(
        'Restoring checkpoint from external_checkpoint_path %s',
        external_checkpoint_path,
    )
    ckpt_to_return = load_checkpoint(
        external_checkpoint_path,
        target=unreplicated_checkpoint_state,
        orbax_checkpointer=orbax_checkpointer,
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


def save_unreplicated_checkpoint(
    train_dir,
    optimizer_state,
    params,
    batch_stats,
    training_metrics_state,
    global_step,
    preemption_count,
    sum_train_cost,
    orbax_checkpointer,
    max_to_keep=1):
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
  save_checkpoint(train_dir,
                  global_step,
                  state,
                  max_to_keep=max_to_keep,
                  orbax_checkpointer=orbax_checkpointer)
  logging.info('Done saving checkpoint.')


def save_checkpoint(train_dir,
                    step,
                    state,
                    prefix='ckpt_',
                    max_to_keep=None,
                    orbax_checkpointer=None):
  """Saves checkpoint to train_dir/{prefix}{step}.

  A list of checkpoints will be stored in train_dir. The user
  is responsible for using unique checkpoint names when calling save_checkpoint
  repeatedly. If the same train_dir and checkpoint name are used more than once,
  the latest file will become corrupt. This may become an issue if max_to_keep
  is not None.

  Args:
    train_dir: (str) Directory to create the checkpoint directory in.
    step: (int) Step of the checkpoint.
    state: (dict) The state to save.
    prefix: (str) Prefix of the checkpoint name.
    max_to_keep: (int) Checkpoints older than the max_to_keep'th will be
      deleted. Defaults to never deleting.
    orbax_checkpointer: orbax.Checkpointer

  Returns:
    The path of the checkpoint directory.
  """
  if max_to_keep is None:
    max_to_keep = sys.maxsize
  flax_checkpoints.save_checkpoint_multiprocess(
      train_dir,
      target=state,
      step=step,
      prefix=prefix,
      keep=max_to_keep,
      overwrite=True,
      orbax_checkpointer=orbax_checkpointer,
  )
  save_dir = os.path.join(train_dir, prefix + str(step))
  return save_dir


def load_checkpoint(
    checkpoint_path, target=None, prefix='ckpt_', orbax_checkpointer=None
):
  """Loads the specified checkpoint."""
  restored = flax_checkpoints.restore_checkpoint(
      checkpoint_path,
      target=target,
      prefix=prefix,
      orbax_checkpointer=orbax_checkpointer,
  )
  return restored


def load_latest_checkpoint(
    train_dir, target=None, prefix='ckpt_', orbax_checkpointer=None
):
  """Loads the most recent checkpoint listed in train_dir.

  Args:
    train_dir: the directory to read checkpoints from.
    target: used for Flax checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    prefix: the prefix of the names of checkpoint files.
    orbax_checkpointer: orbax.Checkpointer
  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67. If
    the directory doesn't exist, it will return the original target.
  """
  try:
    restored = flax_checkpoints.restore_checkpoint(
        train_dir, target=target, prefix=prefix,
        orbax_checkpointer=orbax_checkpointer
    )
    return restored
  except ValueError:
    return target
