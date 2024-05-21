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

"""BLEU calculation utilities."""

import os
import pathlib

from init2winit import checkpoint
import jax
import sacrebleu
from tensorflow.io import gfile

exists = gfile.exists
glob = gfile.glob


def compute_bleu_from_predictions(predictions, references, language_code, name):
  """Computes BLEU score given predictions and references."""
  sacrebleu_tokenizer = 'zh' if language_code == 'zh' else sacrebleu.DEFAULT_TOKENIZER
  bleu_score = sacrebleu.corpus_bleu(
      predictions, [references], tokenize=sacrebleu_tokenizer).score
  return {name: bleu_score}


def get_eval_fpath(ckpt_dir, ckpt_step, eval_split):
  output_dir = str(pathlib.Path(ckpt_dir).parents[0])
  return os.path.join(output_dir, 'bleu_' + eval_split + '_' + str(ckpt_step))


def load_evals(ckpt_dir, ckpt_step, eval_split):
  """Loads results if already available, else return None."""
  ckpt_eval_fpath = get_eval_fpath(ckpt_dir, ckpt_step, eval_split)
  if not exists(ckpt_eval_fpath):
    return None
  else:
    with gfile.GFile(ckpt_eval_fpath, 'r') as f:
      bleu_score = f.readlines()[-1]
      return float(bleu_score.strip())


def save_evals(ckpt_dir, ckpt_step, eval_split, bleu_score):
  ckpt_eval_fpath = get_eval_fpath(ckpt_dir, ckpt_step, eval_split)
  with gfile.GFile(ckpt_eval_fpath, 'w') as f:
    f.write(str(bleu_score))


def _load_checkpoint(checkpoint_path, params, replicate=True):
  """Load model (and batch stats) from checkpoint."""
  target = dict(
      params=params,
      global_step=-1,
      preemption_count=0,
      sum_train_cost=0.0)
  ckpt = checkpoint.load_checkpoint(
      checkpoint_path,
      target=target,
  )
  results = checkpoint.replicate_checkpoint(
      ckpt,
      pytree_keys=['params'],
      replicate=replicate,
  )
  params = results[0]['params']
  return params


def average_checkpoints(checkpoint_paths, params):
  """Averages a set of checkpoints in input checkpoints."""
  assert len(checkpoint_paths) >= 1
  # Sum parameters of separate models together.
  params = _load_checkpoint(checkpoint_paths[0], params, replicate=False)
  for checkpoint_path in checkpoint_paths[1:]:
    params_update = _load_checkpoint(checkpoint_path, params, replicate=False)
    # TODO(dxin): Make this averaging process more numerically stable.
    params = jax.tree.map(
        lambda x, y: x + y, params, params_update)

  # Average checkpoints.
  params = jax.tree.map(lambda x: x / float(len(checkpoint_paths)), params)
  return params


def get_checkpoints_in_range(checkpoint_dir, lower_bound, upper_bound):
  """Get checkpoint paths in step range [lower_bound, upper_bound]."""
  checkpoint_paths = []
  for checkpoint_path in glob(os.path.join(checkpoint_dir, 'ckpt_*')):
    ckpt_step = int(checkpoint_path.split('_')[-1])
    if ckpt_step >= lower_bound and ckpt_step <= upper_bound:
      checkpoint_paths.append(os.path.join(checkpoint_dir, checkpoint_path))
  return checkpoint_paths
