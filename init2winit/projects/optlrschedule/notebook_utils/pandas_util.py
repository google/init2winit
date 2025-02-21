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

r"""Utility functions for pandas dataframes."""

from typing import Any, Dict, List
import pandas as pd


def reduce_to_best_base_lrs(input_df: pd.DataFrame):
  """Reduce to the best base_lr for each set of parameters.

  this function retains only the records that have the base_lr with the lowest
  median score,
  among the multiple base_lr candidates.

  Args:
      input_df (pd.DataFrame): Input DataFrame containing configurations and
        scores.

  Returns:
      pd.DataFrame: DataFrame containing the best base_lr for each set of
        parameters.

  [Process Flow]
    1. Calculate the median score for each (param + base_lr) combination.
    2. For each set of parameters, select the base_lr with the lowest median
    score.
    3. Extract from the original DataFrame only the records that match the
    selected (param, base_lr) combination.
  """

  # Exclude columns: 'score' and 'generation' are not used for aggregation.
  # Note: 'base_lr' is used as a candidate, so it is not excluded.
  excluded_cols = ['generation', 'score']
  # param_cols are the parameter columns excluding 'base_lr'
  param_cols = [
      col
      for col in input_df.columns
      if col not in (excluded_cols + ['base_lr'])
  ]

  # Step 1: Calculate the median score for each (param + base_lr) combination
  candidate_medians = (
      input_df.groupby(param_cols + ['base_lr'])['score'].median().reset_index()
  )
  candidate_medians.rename(columns={'score': 'score_median'}, inplace=True)

  # Step 2: For each set of parameters, select the base_lr with the lowest
  # median score.
  best_candidates = (
      candidate_medians.sort_values('score_median')
      .groupby(param_cols)
      .first()
      .reset_index()
  )

  # Step 3: Extract from the original DataFrame only the records that match the
  # selected (param, base_lr) combination.
  result = input_df.merge(best_candidates, on=param_cols + ['base_lr'])

  return result.drop(columns=['score_median'])


def add_seed_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Add seed stats columns to a DataFrame.

  Args:
      df (pd.DataFrame): Input DataFrame containing configurations and scores.

  Returns:
      pd.DataFrame: DataFrame containing configurations and scores, with
      additional columns for seed stats.
  """
  # Ensure no leftover param_tuple column
  if 'param_tuple' in df.columns:
    df = df.drop(columns=['param_tuple'])

  # Columns to exclude from parameter analysis
  exclude_cols = {'score', 'rank', 'index', 'generation'}
  param_cols = [col for col in df.columns if col not in exclude_cols]

  # Group by parameter columns and calculate statistics
  stats_df = (
      df.groupby(param_cols)
      .agg(
          score_mean=('score', 'mean'),
          score_median=('score', 'median'),
          score_std=('score', 'std'),
          score_min=('score', 'min'),
          score_max=('score', 'max'),
          group_size=(
              'score',
              'size',
          ),  # Count the number of samples in each group
      )
      .reset_index()
  )
  return stats_df


def filter_top_k_schedules(
    df: pd.DataFrame, top_k: int, metric: str, ascending: bool = True
) -> pd.DataFrame:
  """Filter schedules to keep only top k performing ones.

  Args:
      df (pd.DataFrame): DataFrame containing schedule data.
      top_k (int): Number of top performing schedules to retain.
      metric (str): Metric to use for ranking.
      ascending (bool): If True, rank in ascending order (e.g., smallest median
        score). If False, rank in descending order (e.g., largest median score).

  Returns:
      pd.DataFrame: DataFrame containing top k performing schedules.
  """
  return (
      df.sort_values(metric, ascending=ascending)
      .head(top_k)
      .reset_index(drop=True)
  )


# TODO(hnaganuma): Remove this function future CLs.
def get_top_n_schedule_params(
    df: pd.DataFrame, n_samples=5, metric='score_median', ascending=True
) -> List[Dict[str, Any]]:
  """Get top N schedules from a DataFrame by mean score or variance.

  Args:
      df (pd.DataFrame): Input DataFrame containing configurations and scores.
      n_samples (int): Number of top configurations to return.
      metric (str): Metric to use for ranking.
      ascending (bool): If True, rank in ascending order (e.g., smallest median
        score). If False, rank in descending order (e.g., largest std or median
        score).

  Returns:
      list[dict]: List of configuration dictionaries with statistics, including
          group size.
  """
  # Ensure no leftover param_tuple column
  if 'param_tuple' in df.columns:
    df = df.drop(columns=['param_tuple'])

  # Columns to exclude from parameter analysis
  exclude_cols = {'score', 'rank', 'index', 'generation'}
  param_cols = [col for col in df.columns if col not in exclude_cols]

  # Group by parameter columns and calculate statistics
  stats_df = (
      df.groupby(param_cols)
      .agg(
          score_mean=('score', 'mean'),
          score_median=('score', 'median'),
          score_std=('score', 'std'),
          score_min=('score', 'min'),
          score_max=('score', 'max'),
          group_size=(
              'score',
              'size',
          ),  # Count the number of samples in each group
      )
      .reset_index()
  )

  # Determine ranking column
  rank_col = metric

  # Sort configurations and select top or worst N
  sorted_stats = stats_df.sort_values(by=rank_col, ascending=ascending).head(
      n_samples
  )

  # Convert rows to dictionaries
  result_dicts = sorted_stats.to_dict('records')

  return result_dicts
