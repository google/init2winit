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

r"""Utility functions for pandas dataframes."""

from typing import Any, Dict, List
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np
import pandas as pd
from scipy import stats

bootstrap = stats.bootstrap


# Compute bootrap standard deviation of sample median
def med_boot_std(
    values,
    method='BCa',
    n_resamples=1000,
    return_ci=False,
    confidence_level=0.95,
):
  """Compute bootrap standard deviation of sample median."""
  boot_obj = bootstrap(
      (values,),
      np.median,
      method=method,
      n_resamples=n_resamples,
      confidence_level=confidence_level,
  )
  if return_ci:
    return (
        boot_obj.standard_error,
        boot_obj.confidence_interval.low,
        boot_obj.confidence_interval.high,
    )
  else:
    return boot_obj.standard_error


def quantile_ci_from_dkw(
    values: np.typing.ArrayLike,
    quantile: float,
    confidence_level: float = 0.95,
    nan_val: float | None = None,
) -> tuple[float, float]:
  """Compute quantile confidence interval upper bound using DKW inequality.

  Uses the Dvoretzky–Kiefer–Wolfowitz–Massart (DKW) inequality to compute a
  conservative confidence interval and variance estimate for the median of a
  population. Only relies on the assumption that values are bounded. Similar in
  spirit to Kolmogorov–Smirnov (KS) statistic.

  Args:
    values: List or array of values for median confidence interval computation.
    quantile: Quantile to compute the confidence interval for, between 0 and 1.
    confidence_level: Confidence level for the confidence interval.
    nan_val: Value to use for NaN values. If none, all elements in values must
      be finite.

  Returns:
    Tuple of (lower bound, upper bound) of the confidence interval.
  """
  if quantile < 0 or quantile > 1:
    raise ValueError('quantile must be between 0 and 1.')
  values = np.array(values)
  if nan_val is None:
    if np.any(np.isnan(values)):
      raise ValueError('All values must be finite if nan_val is None.')
  else:
    values[~np.isfinite(values)] = nan_val
  alpha = 1 - confidence_level
  num_values = len(values)
  eps = np.sqrt(
      np.log(2 / alpha) / (2 * num_values)
  )  # band size, in probability

  # Upper and lower bound CDF values
  F_ub = quantile + eps  # pylint: disable=invalid-name
  F_lb = quantile - eps  # pylint: disable=invalid-name

  # If bounds are outside of [0, 1], then CI estimate is unbounded in at least
  # one direction.
  if F_ub >= 1:
    raise ValueError(f'No upper bound for quantile {quantile}')
  if F_lb <= 0:
    raise ValueError('No lower bound for quantile {quantile}')

  # Get CDF values corresponding to band limits
  sorted_values = np.sort(values)
  if F_ub > 1:
    ub_val = float('inf')
  else:
    ub_idx = int(np.floor(F_ub * num_values))  # Smallest value which is > F_ub
    ub_val = sorted_values[ub_idx]
  if F_lb < 0:
    lb_val = float('-inf')
  else:
    lb_idx = int(np.floor(F_lb * num_values))  # Largest value which is < F_lb
    lb_val = sorted_values[lb_idx]

  return (lb_val, ub_val)  # CI values


def med_ci_from_dkw(values, confidence_level=0.95, nan_val=None):
  """Compute median confidence interval using DKW inequality."""
  return quantile_ci_from_dkw(values, 0.5, confidence_level, nan_val)


def med_dkw_std(values, return_ci=False, confidence_level=0.95, nan_val=None):
  """Compute median statistics using DKW inequality.

  Uses DKW inequality to compute confidence interval for the population median.
  Length of interval is returned as standard error for analysis and plotting.

  Args:
    values: List of values to compute the median confidence interval for.
    return_ci: If True, return the confidence interval as a tuple of (lower,
      upper) bounds. Otherwise, return the standard error.
    confidence_level: Confidence level for the confidence interval.
    nan_val: Value to use for NaN values. If none, all elements in values must
      be finite.

  Returns:
    If return_ci is True, returns a tuple of (lower bound, upper bound) of the
      confidence interval. Otherwise, returns the standard error. Standard error
      is computed as (upper bound - lower bound)/2.
  """
  lb, ub = quantile_ci_from_dkw(values, 0.5, confidence_level, nan_val)
  std_err = (ub - lb) / 2
  if return_ci:
    return (std_err, lb, ub)
  else:
    return std_err


def reduce_to_best_base_lrs(input_df: pd.DataFrame,
                            group_on_xid_history: bool = False,
                            reduction_statistic: str = 'median'):
  """Reduce to the best base_lr for each set of parameters.

  This function retains only the records that have the base_lr with the lowest
  median score, among the multiple base_lr candidates. It does NOT perform any
  reduction across different seeds for the same (param, base_lr) combination.

  Args:
      input_df (pd.DataFrame): Input DataFrame containing configurations and
        scores.
      group_on_xid_history (bool): If True, group by xid history as well.
      reduction_statistic (str): Statistic to use for reduction. Currently,
        only 'median' and 'mean' are supported.

  Returns:
      pd.DataFrame: DataFrame containing the best base_lr for each set of
        parameters.

  [Process Flow]
    1. Calculate the median score for each (param + base_lr) combination.
    2. For each set of parameters, select the base_lr with the lowest
    reduction_statistic score (median by default).
    3. Extract from the original DataFrame only the records that match the
    selected (param, base_lr) combination.
  """
  distinct_shape_cols = [
      col
      for col in input_df.columns
      if base_schedule_family.is_schedule_param(col)
  ]
  if group_on_xid_history:
    # Same params, different xid history treated as distinct shapes
    distinct_shape_cols.append('xid_history')

  # Step 1: Calculate the median score for each
  # (param + base_lr + [optional] xid_history) combination.
  groupby_cols = distinct_shape_cols + ['base_lr']
  if reduction_statistic == 'median':
    candidate_summary_scores = (
        input_df.groupby(groupby_cols)['score'].median().reset_index()
    )
  elif reduction_statistic == 'mean':
    candidate_summary_scores = (
        input_df.groupby(groupby_cols)['score'].mean().reset_index()
    )
  else:
    raise ValueError(
        f'Unsupported reduction statistic: {reduction_statistic}. Currently,'
        ' only "median" and "mean" are supported.'
    )
  candidate_summary_scores.rename(
      columns={'score': 'score_reduced'}, inplace=True)

  # Step 2: For each set of parameters (and optionally, xid history),
  # select the base_lr with the lowest median score.
  best_candidates = (
      candidate_summary_scores.sort_values('score_reduced')
      .groupby(distinct_shape_cols)
      .first()
      .reset_index()
  )

  # Step 3: Extract from the original DataFrame only the records that match the
  # selected (param, [optional] xid_history, base_lr) combination.
  result = input_df.merge(best_candidates, on=groupby_cols)
  return result.drop(columns=['score_reduced'])


def add_seed_stats_columns(
    df: pd.DataFrame, ci_config: dict[str, Any] | None = None
) -> pd.DataFrame:
  """Add seed stats columns to a DataFrame.

  Args:
      df (pd.DataFrame): Input DataFrame containing configurations and scores.
      ci_config (dict): Configuration for bootstrap sample median standard
        deviation computation. If None, use normal approximation.

  Returns:
      pd.DataFrame: DataFrame containing configurations and scores, with
      additional columns for seed stats.
  """
  param_cols = [
      col for col in df.columns if base_schedule_family.is_schedule_param(col)
  ]
  group_cols = ['base_lr'] + param_cols
  if 'xid_history' in df.columns:  # Group on xid history if present
    group_cols.append('xid_history')

  if ci_config is None:
    agg_dict = dict(
        score_mean=('score', 'mean'),
        score_median=('score', 'median'),
        score_std=('score', 'std'),  # Sample standard deviation, 1 DDOF
        score_min=('score', 'min'),
        score_max=('score', 'max'),
        group_size=(
            'score',
            'size',
        ),
    )
  else:
    med_bound_type = ci_config.pop('median_bound_type', 'bootstrap')
    if med_bound_type == 'bootstrap':
      # Boostrap standard deviation function
      def med_std_apply_fn(values):
        return med_boot_std(values, **ci_config)

    elif med_bound_type == 'dkw':
      # DKW standard deviation function
      def med_std_apply_fn(values):
        return med_dkw_std(values, **ci_config)

    else:
      raise ValueError(f'Unsupported median type: {med_bound_type}')

    agg_dict = dict(
        score_mean=('score', 'mean'),
        score_median=('score', 'median'),
        score_median_error=('score', med_std_apply_fn),
        score_std=('score', 'std'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
        group_size=(
            'score',
            'size',
        ),
    )
  stats_df = df.groupby(group_cols).agg(**agg_dict).reset_index()

  # Standard deviation of sample mean
  stats_df['score_std_error'] = stats_df['score_std'] / np.sqrt(
      stats_df['group_size']  # score_std already uses 1 DDOF
  )

  # Alternative computation of median error using normal approximation
  stats_df['score_median_error_normal'] = (
      np.sqrt(np.pi / 2)
      * stats_df['score_std_error']
  )
  if ci_config is None:
    stats_df['score_median_error'] = stats_df['score_median_error_normal']
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

  # Standard deviation of sample mean (score_std already uses 1 DDOF)
  stats_df['score_std_error'] = stats_df['score_std'] / np.sqrt(
      stats_df['group_size']
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


def get_scores_from_schedule_shapes(
    df: pd.DataFrame,
    schedule_param_df: pd.DataFrame,
    reduce_seeds: bool = False,
    ci_config: dict[str, Any] | None = None,
):
  """Get scores for schedule shapes whose parameters are stored in a DataFrame.

  Args:
      df: DataFrame containing scores from training runs.
      schedule_param_df: DataFrame containing schedule parameters to select.
      reduce_seeds: If True, compute statistics over seeds.
      ci_config: Configuration for computing confidence intervals of median if
        reduce_seeds is True. If None, use normal approximation.

  Returns:
      Dataframe containing all scores for the given schedules.
  """

  scores_df = pd.merge(
      df, schedule_param_df, how='right', validate='many_to_one'
  )
  # Optionally, reduce over seeds
  if reduce_seeds:
    scores_df = add_seed_stats_columns(scores_df, ci_config)
  # Return scores
  return scores_df.reset_index(drop=True)


def extract_sweeps_from_results(
    df,
    initial_params_df,
    ci_config='default',
):
  """Process data from coordinate descent experiment into individual sweeps.

  Given a dataframe of scores and a dataframe of initial schedule space
  parameters, this function computes statistics and organizes dataframes
  into a list of sweep dictionaries. For the purposes of this code, a sweep is
  the result of coordinate descent on a single parameter, from a single initial
  condition, including the median score, a corresponding confidence interval,
  and the base_lr that achieves the median score. Sweeps are sorted by the swept
  parameter value.

  The function returns a list where each element corresponds to an initial
  condition. Each element of the list is a dictionary containing the parameter
  sweep dataframes for that initial condition, keyed by the parameter name.

  Args:
    df: Dataframe of scores over all seeds.
    initial_params_df: Dataframe of initial parameters for each sweep.
    ci_config: Dictionary of confidence interval configuration parameters.
      Default value is string 'default', which uses boostrapped 95% CI.

  Returns:
    stats_df: Dataframe of statistics (no other conditioning).
    sweep_dfs_per_point: List of dictionaries, one for each initial parameter.
      Each dictionary maps a parameter name to a sweep dataframe of scores
      varying that parameter and leaving others fixed. base_lr is
      optimized for each parameter value.
  """
  if ci_config == 'default':
    ci_config = {
        'return_ci': True,
        'confidence_level': 0.95,
    }
  elif not isinstance(ci_config, dict):
    raise ValueError('ci_config must be a dictionary or "default".')
  if not ci_config.get('return_ci', False):
    raise ValueError('Median calculation must return CI for plotting.')

  param_list = list(initial_params_df.columns)
  # Reduce to best base lrs per parameter
  base_lr_opt_df = reduce_to_best_base_lrs(df)
  # Reduce over seeds and compute statistics
  stats_df = add_seed_stats_columns(base_lr_opt_df, ci_config=ci_config)

  # Extract confidence interval to its own column
  stats_df['ci_lower'] = stats_df['score_median_error'].apply(lambda x: x[1])
  stats_df['ci_upper'] = stats_df['score_median_error'].apply(lambda x: x[2])

  # For each starting point, return list of sub-frames of coordinate descent
  sweep_dfs_per_point = []
  for _, initial_param in initial_params_df.iterrows():
    # For each point, sweep dataframes are organized into dict by param name
    sweep_dfs_per_param = {}
    for param in param_list:
      fixed_conditions_df = pd.DataFrame([initial_param.drop(param).to_dict()])
      sweep_dfs_per_param[param] = pd.merge(
          stats_df, fixed_conditions_df, how='right', validate='many_to_one'
      )
      sweep_dfs_per_param[param] = (
          sweep_dfs_per_param[param]
          .sort_values(by=param, ascending=True)
          .reset_index(drop=True)
      )
    sweep_dfs_per_point.append(sweep_dfs_per_param)

  return stats_df, sweep_dfs_per_point
