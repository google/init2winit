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

"""Implements checkpointing of nested python containers of numpy arrays.

This is useful for training neural networks with stax, where model parameters
are nested numpy arrays.
"""

import codecs
import copy
import os
import sys
import threading

from absl import flags
from absl import logging
from flax import serialization
from flax.training import checkpoints as flax_checkpoints

import msgpack
from tensorflow.io import gfile


FLAGS = flags.FLAGS

_save_checkpoint_background_thread = None
_save_checkpoint_background_error = None
_save_checkpoint_background_lock = threading.RLock()




class CheckpointState(object):
  """Object to package up the state we load and restore.

  We save generic python state using msgpack. We assume any kwarg other than
  pytree is just extra python state and not a pytree. pytree contains
  all state that must be replicated across devices (presumably only Jax arrays).
  """

  def __init__(self, pytree, **kwargs):
    self.pytree = pytree
    # TODO(znado): rename to be more descriptive/distinct.
    self.pystate = kwargs


def _ckpt_state_dict(checkpoint_state):
  return serialization.to_state_dict({
      'pytree': serialization.to_state_dict(
          checkpoint_state.pytree),
      'pystate': serialization.to_state_dict(checkpoint_state.pystate),
  })


def _ckpt_restore_state(checkpoint_state, state_dict):
  """Restore the state from the state dict.

  Allows for checkpointing the class object.

  Args:
    checkpoint_state: an instance of a CheckpointState.
    state_dict: a state dict containing the desired new state of the object.

  Returns:
    The restored class object.
  """
  checkpoint_state.pytree = serialization.from_state_dict(
      checkpoint_state.pytree, state_dict['pytree'])
  checkpoint_state.pystate = serialization.from_state_dict(
      checkpoint_state.pystate, state_dict['pystate'])
  return checkpoint_state


# Note that this will only be used if we call
# `flax_checkpoints.restore_checkpoint` with the `target` arg set to a
# CheckpointState object to be filled in with values.
serialization.register_serialization_state(
    CheckpointState, _ckpt_state_dict, _ckpt_restore_state, override=True)


def _save_checkpoint_background_catch_error(*args, **kwargs):
  """Call save_checkpoint with provided args, store exception if any."""
  global _save_checkpoint_background_error
  try:
    save_checkpoint(*args, **kwargs)
    _save_checkpoint_background_error = None
  except BaseException as err:  # pylint: disable=broad-except
    logging.exception('Error while saving checkpoint in background.')
    _save_checkpoint_background_error = err


def wait_for_checkpoint_save():
  """Wait until last checkpoint save (if any) to finish."""
  with _save_checkpoint_background_lock:
    if _save_checkpoint_background_thread:
      _save_checkpoint_background_thread.join()
    if _save_checkpoint_background_error:
      raise _save_checkpoint_background_error


def save_checkpoint_background(*args, **kwargs):
  """Saves checkpoint to train_dir/checkpoint_name in a background thread.

  Args:
    *args:
    **kwargs: See save_checkpoint for a descrition of the arguments.

  The process is prevented from exiting until the last checkpoint as been saved.

  At most one checkpoint can be saved simultaneously. If the function is called
  while a previous checkpoint is being saved, the function will block until that
  previous checkpoint saving finishes.

  The provided state can be mutated after this function returns.

  Raises error raised by save_checkpoint during the previous call, if any.
  """
  with _save_checkpoint_background_lock:
    wait_for_checkpoint_save()
    global _save_checkpoint_background_thread
    # Copy everything for state, rest is negligeable, do it to keep it simple.
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    _save_checkpoint_background_thread = threading.Thread(
        target=_save_checkpoint_background_catch_error,
        args=args, kwargs=kwargs, daemon=False)
    _save_checkpoint_background_thread.start()


def save_checkpoint(train_dir,
                    checkpoint_name,
                    state,
                    max_to_keep=None,
                    recents_filename='latest',
                    use_deprecated_checkpointing=True):
  """Saves checkpoint to train_dir/checkpoint_name.

  A list of checkpoints will be stored in train_dir/recents_filename. The user
  is responsible for using unique checkpoint names when calling save_checkpoint
  repeatedly. If the same train_dir and checkpoint name are used more than once,
  the latest file will become corrupt. This may become an issue if max_to_keep
  is not None.

  Args:
    train_dir: (str) Directory to create the checkpoint directory in.
    checkpoint_name: (str) Name of the checkpoint.
    state: (CheckpointState) The state to save.
    max_to_keep: (int) Checkpoints older than the max_to_keep'th will be
      deleted. Defaults to never deleting.
    recents_filename: (str) Filename to keep track of the most recent
      checkpoints. All calls of save_checkpoint with the same recents_filename
      and train_dir will be treated as different snapshots of the same state and
      cleaned up together, even though their checkpoint_name will generally
      differ.
    use_deprecated_checkpointing: Whether to use deprecated checkpointing.

  Returns:
    The path of the checkpoint directory.
  """
  checkpoint_type_str = 'deprecated' if use_deprecated_checkpointing else 'Flax'
  logging.info('Using %s checkpointing format.', checkpoint_type_str)
  save_dir = os.path.join(train_dir, checkpoint_name)
  if max_to_keep is None:
    max_to_keep = sys.maxsize
  flax_checkpoints.save_checkpoint(
      train_dir,
      target=state,
      step='',
      prefix=checkpoint_name,
      keep=max_to_keep,
      overwrite=True)
  return save_dir


def load_checkpoint(
    checkpoint_path, target=None, use_deprecated_checkpointing=True):
  """Loads the specified checkpoint."""
  checkpoint_type_str = 'deprecated' if use_deprecated_checkpointing else 'Flax'
  logging.info('Using %s checkpointing format.', checkpoint_type_str)
  restored = flax_checkpoints.restore_checkpoint(checkpoint_path, target=target)
  return restored


def load_latest_checkpoint(
    train_dir,
    recents_filename='latest',
    target=None,
    use_deprecated_checkpointing=True):
  """Loads the most recent checkpoint listed in train_dir/recents_filename.

  Args:
    train_dir: the directory to read checkpoints from.
    recents_filename: used for deprecated checkpointing, specifies the filename
      inside train_dir that contains the path to the most recent checkpoint.
    target: used for Flax checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    use_deprecated_checkpointing: whether or not to use Flax checkpointing, or
      the previous deprecated checkpointing.

  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67.
  """
  checkpoint_type_str = 'deprecated' if use_deprecated_checkpointing else 'Flax'
  logging.info('Using %s checkpointing format.', checkpoint_type_str)
  restored = flax_checkpoints.restore_checkpoint(train_dir, target=target)
  return restored
