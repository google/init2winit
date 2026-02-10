# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Registry of eval metrics and whether they should be minimized or not.

This file is useful when writing configs that perform tuning studies. In
general, users should call is_minimized on eval metric column names.
"""
import collections
import itertools


MIN_EVAL_METRICS = [
    'ce_loss',
    'error_rate',
    'ctc_loss',
    'wer',
    'l1_loss',
    'perplexity',
]
MAX_EVAL_METRICS = ['average_precision', 'ssim', 'bleu_score']


def generate_eval_cols(metrics: collections.abc.Iterable[str]) -> list[str]:
  splits = ['train', 'valid', 'test']
  return [f'{split}/{col}' for split, col in itertools.product(splits, metrics)]


MINIMIZE_REGISTRY = {k: True for k in generate_eval_cols(MIN_EVAL_METRICS)}
MINIMIZE_REGISTRY.update(
    {k: False for k in generate_eval_cols(MAX_EVAL_METRICS)})
MINIMIZE_REGISTRY['train_cost'] = True
MINIMIZE_REGISTRY['callback/wmt14_translate/de-en/valid/bleu_score'] = False
MINIMIZE_REGISTRY['callback/wmt14_translate/de-en/test/bleu_score'] = False


def is_minimized(col_name: str) -> bool:
  """Guess if the eval metric column name should be minimized or not."""
  for prefix in ['best_', 'final_']:
    col_name = col_name.replace(prefix, '')

  for col in MINIMIZE_REGISTRY:
    if col in col_name:
      return MINIMIZE_REGISTRY[col]

  raise ValueError(f'Column {col_name} not found in `MINIMIZE_REGISTRY` as '
                   'either a column name or a substring of a column name.')
