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
import copy
import os
import sys
import threading

from absl import flags
from absl import logging
from flax.training import checkpoints as flax_checkpoints


FLAGS = flags.FLAGS

_save_checkpoint_background_thread = None
_save_checkpoint_background_error = None
_save_checkpoint_background_lock = threading.RLock()


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
                    step,
                    state,
                    prefix='ckpt_',
                    max_to_keep=None):
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

  Returns:
    The path of the checkpoint directory.
  """
  if max_to_keep is None:
    max_to_keep = sys.maxsize
  flax_checkpoints.save_checkpoint(
      train_dir,
      target=state,
      step=step,
      prefix=prefix,
      keep=max_to_keep,
      overwrite=True)
  save_dir = os.path.join(train_dir, prefix + str(step))
  return save_dir


def load_checkpoint(
    checkpoint_path,
    target=None,
    prefix='ckpt_'):
  """Loads the specified checkpoint."""
  restored = flax_checkpoints.restore_checkpoint(
      checkpoint_path, target=target, prefix=prefix)
  return restored


def load_latest_checkpoint(train_dir, target=None, prefix='ckpt_'):
  """Loads the most recent checkpoint listed in train_dir.

  Args:
    train_dir: the directory to read checkpoints from.
    target: used for Flax checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    prefix: the prefix of the names of checkpoint files.

  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67.
  """
  restored = flax_checkpoints.restore_checkpoint(
      train_dir, target=target, prefix=prefix)
  return restored
