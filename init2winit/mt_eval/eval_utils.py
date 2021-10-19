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

"""BLEU calculation utilities."""

import os
import pathlib

from flax import jax_utils
from init2winit import checkpoint
from init2winit import trainer
import jax
import sacrebleu
from tensorflow.io import gfile


def compute_bleu_from_predictions(predictions, references, name):
  """Computes BLEU score given predictions and references."""
  bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
  return {name: bleu_score}


def get_eval_fpath(ckpt_dir, ckpt_step, eval_split):
  output_dir = str(pathlib.Path(ckpt_dir).parents[0])
  return os.path.join(output_dir, 'bleu_' + eval_split + '_' + str(ckpt_step))


def load_evals(ckpt_dir, ckpt_step, eval_split):
  """Loads results if already available, else return None."""
  ckpt_eval_fpath = get_eval_fpath(ckpt_dir, ckpt_step, eval_split)
  if not gfile.exists(ckpt_eval_fpath):
    return None
  else:
    with gfile.GFile(ckpt_eval_fpath, 'r') as f:
      bleu_score = f.readlines()[-1]
      return float(bleu_score.strip())


def save_evals(ckpt_dir, ckpt_step, eval_split, bleu_score):
  ckpt_eval_fpath = get_eval_fpath(ckpt_dir, ckpt_step, eval_split)
  with gfile.GFile(ckpt_eval_fpath, 'w') as f:
    f.write(str(bleu_score))


def load_checkpoint(checkpoint_path, optimizer, batch_stats,
                    replicate=True):
  """Load model (and batch stats) from checkpoint."""
  ckpt = checkpoint.load_checkpoint(
      checkpoint_path,
      target=(optimizer, batch_stats),
      use_deprecated_checkpointing=True
  )
  results = trainer.restore_checkpoint(
      ckpt,
      (optimizer, batch_stats),
      replicate=replicate,
      use_deprecated_checkpointing=True
  )
  optimizer, batch_stats = results[0]
  return optimizer, batch_stats


def average_checkpoints(checkpoint_paths, optimizer, batch_stats):
  """Averages a set of checkpoints in input checkpoints."""
  assert len(checkpoint_paths) >= 1
  # Sum parameters of separate models together.
  optimizer, batch_stats = load_checkpoint(checkpoint_paths[0], optimizer,
                                           batch_stats, replicate=False)
  for checkpoint_path in checkpoint_paths[1:]:
    optimizer_update, _ = load_checkpoint(
        checkpoint_path, optimizer, batch_stats, replicate=False
    )
    # TODO(dxin): Make this averaging process more numerically stable.
    target_params_update = jax.tree_map(
        lambda x, y: x + y, optimizer.target.params,
        optimizer_update.target.params)
    state_params_update = jax.tree_map(
        lambda x, y: x + y, optimizer.state.param_states.params,
        optimizer_update.state.param_states.params)
    optimizer.target.replace(params=target_params_update)
    optimizer.state.param_states.replace(params=state_params_update)

  # Average checkpoints.
  optimizer.target.replace(params=jax.tree_map(
      lambda x: x / float(len(checkpoint_paths)),
      optimizer.target.params))
  optimizer.state.param_states.replace(params=jax.tree_map(
      lambda x: x / float(len(checkpoint_paths)),
      optimizer.state.param_states.params))

  # Replicate across devices.
  optimizer, batch_stats = jax_utils.replicate((optimizer, batch_stats))

  return optimizer, batch_stats


def get_checkpoints_in_range(checkpoint_dir, lower_bound, upper_bound):
  """Get checkpoint paths in step range [lower_bound, upper_bound]."""
  checkpoint_paths = []
  for checkpoint_path in gfile.glob(
      os.path.join(checkpoint_dir, 'ckpt_*')):
    ckpt_step = int(checkpoint_path.split('_')[-1])
    if ckpt_step >= lower_bound and ckpt_step <= upper_bound:
      checkpoint_paths.append(os.path.join(checkpoint_dir, checkpoint_path))
  return checkpoint_paths
