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

"""Utility functions for plotting data in notebooks."""

import collections
from init2winit.projects.optlrschedule.notebook_utils import pandas_util
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Global font sizes for plots (change these values to update font sizes
# globally)
BAR_TEXT_FONT_SIZE = 16
BAR_ALPHA = 0.5
AXIS_LABEL_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 16
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
LINE_WIDTH = 3.0  # Unified line width for all plots

LR_SCHEDULE_NAME_MAP = {
    'constant': 'Constant',
    'cosine': 'Generalized Cosine',
    'cosine_y': 'Cosine w/non-zero decay',
    'cosine_standard': 'Cosine w/exponent=1.0',
    'twopointslinear': 'Two-Point Linear',
    'twopointsspline': 'Two-Point Spline',
    'twopointsspline_y': 'Two-Point Spline w/non-zero decay',
    'smoothnonmonotonic': 'Smooth Non-Monotonic',
    'sqrt': 'Square Root',
    'rex': 'Generalized REX',
}

LR_SCHEDULE_SHORT_NAME_MAP = {
    'constant': 'con',
    'cosine': 'cos-gen',
    'cosine_y': 'cos-y',
    'cosine_standard': 'cos-std',
    'twopointslinear': 'tpl',
    'twopointsspline': 'tps',
    'twopointsspline_y': 'tps-y',
    'smoothnonmonotonic': 'snm',
    'sqrt': 'sqrt',
    'rex': 'rex',
}

DEFAULT_CMAP = plt.cm.Set2
DEFAULT_CYCLE_LENGTH = 9


def get_colors_from_cmap(
    num_colors: int, cmap=DEFAULT_CMAP, cycle_length: int = DEFAULT_CYCLE_LENGTH
):
  """Returns a list of colors from a colormap.

  Args:
    num_colors: The number of colors to return.
    cmap: The colormap to use.
    cycle_length: The number of colors before repeating the color cycle.

  Returns:
    A list of colors.
  """
  if cycle_length == 1:
    index_delta = 0
  else:
    index_delta = 1.0 / (cycle_length - 1)
  index_list = [(i % cycle_length) * index_delta for i in range(num_colors)]
  return cmap(index_list)


def map_schedule_name_to_label(
    schedule_type: str, short_name: bool = False
) -> str:
  """Maps the given lr schedule type name to its human-readable string.

  Args:
      schedule_type (str): The lr schedule type name (e.g., 'constant',
        'cosine', etc.)
      short_name (bool, optional): If True, return the short name instead of the
        full name.

  Returns:
      str: The corresponding human-readable lr schedule name. If the given
      schedule_type is not found,
           the original schedule_type is returned.
  """
  # Convert the input to lowercase to ensure case-insensitive matching
  # and return the mapped name.
  if short_name:
    return LR_SCHEDULE_SHORT_NAME_MAP.get(schedule_type.lower(), schedule_type)
  else:
    return LR_SCHEDULE_NAME_MAP.get(schedule_type.lower(), schedule_type)


def plot_best_base_lr_histgram(
    df: pd.DataFrame,
    exp_name: str,
    title: str,
    key_metric: str,
    ax: plt.Axes = None,
) -> plt.Axes:
  """Plots the histogram of the best base_lr values determined by the minimum key_metric value.

  Args:
      df (pd.DataFrame): DataFrame containing columns for base_lr and score.
      exp_name (str): Experiment name to be used in the plot title.
      title (str): Additional title string.
      key_metric (str): Metric to use for selecting the best base_lr (e.g.,
        'score').
      ax (plt.Axes, optional): Existing matplotlib Axes. If None, a new one is
        created.

  Returns:
      plt.Axes: The matplotlib Axes object on which the plot is drawn.
  """
  param_key_list = [
      col for col in df.columns if base_schedule_family.is_schedule_param(col)
  ]

  # Compute indices of minimum key_metric values per group
  min_score_indices = df.groupby(param_key_list)[key_metric].idxmin()
  base_lr_values = df.loc[min_score_indices, 'base_lr']

  # Get all unique base_lr values (sorted)
  all_base_lr = sorted(df['base_lr'].unique())

  # Count frequency of each base_lr among the selected best values
  value_counts = pd.Series(0, index=all_base_lr)
  counts = base_lr_values.value_counts()
  value_counts[counts.index] = counts

  # Create a new axis if not provided
  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # Create a bar plot for the histogram
  bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=BAR_ALPHA)

  # Annotate each bar with its count
  for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=BAR_TEXT_FONT_SIZE,
    )

  # Set x-axis ticks and labels
  ax.set_xticks(range(len(value_counts)))
  ax.set_xticklabels(
      [f'{x:.6f}' for x in value_counts.index],
      rotation=45,
      ha='right',
      fontsize=AXIS_TICK_FONT_SIZE,
  )

  # Set labels and title
  ax.set_xlabel('Base LR', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_ylabel('Count', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_title(f'{exp_name} / {title}', fontsize=TITLE_FONT_SIZE)

  # Set y-axis tick label font size
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  # Add legend showing the experiment name
  ax.legend([exp_name], prop={'size': LEGEND_FONT_SIZE})

  return ax


def plot_multiple_best_base_lr_histgrams(
    df_list: list[pd.DataFrame],
    exp_names: list[str],
    title: str,
    key_metric: str,
    ax: plt.Axes = None,
) -> plt.Axes:
  """Overlays multiple best base_lr histograms from a list of DataFrames.

  Args:
      df_list (list): List of DataFrames, each containing columns for base_lr
        and score.
      exp_names (list): List of experiment names corresponding to each
        DataFrame.
      title (str): Plot title.
      key_metric (str): Metric to use for selecting the best base_lr (e.g.,
        'score').
      ax (plt.Axes, optional): Existing matplotlib Axes. If None, a new one is
        created.

  Returns:
      plt.Axes: The matplotlib Axes object on which the overlaid histograms are
      drawn.
  """
  if len(df_list) != len(exp_names):
    raise ValueError('The lengths of df_list and exp_names must match.')

  # Compute the union of all unique base_lr values from all DataFrames
  all_base_lr = sorted(
      set().union(*(set(df['base_lr'].unique()) for df in df_list))
  )

  # Create a new axis if not provided
  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # Allocate a total bar width (e.g., 0.8) for each bin and compute
  # individual bar width
  n = len(df_list)
  total_width = 0.8
  bar_width = total_width / n
  # Compute offsets so that the histograms are centered within each bin:
  offsets = [i * bar_width - total_width / 2 + bar_width / 2 for i in range(n)]

  # Define common x positions based on the union of base_lr values
  x_positions = np.arange(len(all_base_lr))

  # Use a color cycle for differentiating experiments
  colors = get_colors_from_cmap(n)

  # For each DataFrame, compute and plot its best base_lr histogram
  for i, (df, exp_name) in enumerate(zip(df_list, exp_names)):
    # Determine parameter columns by excluding specific columns
    param_key_list = [
        col for col in df.columns if base_schedule_family.is_schedule_param(col)
    ]

    # Compute indices of minimum key_metric values per group
    min_score_indices = df.groupby(param_key_list)[key_metric].idxmin()
    base_lr_values = df.loc[min_score_indices, 'base_lr']

    # Count frequency of each base_lr among the selected best values
    # using the union bins
    value_counts = pd.Series(0, index=all_base_lr)
    counts = base_lr_values.value_counts()
    value_counts[counts.index] = counts

    # Plot the histogram for this experiment with a computed offset
    offset = offsets[i]
    bars = ax.bar(
        x_positions + offset,
        value_counts.values,
        width=bar_width,
        alpha=BAR_ALPHA,
        color=colors[i],
        label=exp_name,
    )

    # Annotate each bar with its count
    for bar in bars:
      height = bar.get_height()
      ax.text(
          bar.get_x() + bar.get_width() / 2.0,
          height,
          f'{int(height)}',
          ha='center',
          va='bottom',
          fontsize=BAR_TEXT_FONT_SIZE,
          color=colors[i],
      )

  # Set x-axis ticks and labels based on the union of base_lr values
  ax.set_xticks(x_positions)
  ax.set_xticklabels(
      [f'{x:.6f}' for x in all_base_lr],
      rotation=45,
      ha='right',
      fontsize=AXIS_TICK_FONT_SIZE,
  )

  # Set labels and title
  ax.set_xlabel('Base LR', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_ylabel('Count', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_title(f'{title}', fontsize=TITLE_FONT_SIZE)

  # Set y-axis tick label font size
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  # Add legend outside the figure to the right
  ax.legend(
      loc='center left',
      bbox_to_anchor=(1.02, 0.5),
      prop={'size': LEGEND_FONT_SIZE},
  )

  return ax


def plot_multiple_best_base_lr_distribution(
    df_list: list[pd.DataFrame],
    exp_names: list[str],
    title: str,
    key_metric: str,
    ax: plt.Axes = None,
) -> plt.Axes:
  """Overlays multiple best base_lr probability distributions (PMFs) from a list of DataFrames.

  For each DataFrame, the best base_lr values are determined by selecting the
  row with the minimum key_metric for each group (grouped by all non-excluded
  parameters). The frequency counts are then normalized to represent
  probabilities. Since base_lr is discrete, the distributions are plotted as
  line plots with markers, and the x-axis is set to log scale.

  Args:
      df_list (list): List of DataFrames, each containing columns for base_lr
        and score.
      exp_names (list): List of experiment names corresponding to each
        DataFrame.
      title (str): Plot title.
      key_metric (str): Metric to use for selecting the best base_lr (e.g.,
        'score').
      ax (plt.Axes, optional): Existing matplotlib Axes. If None, a new one is
        created.

  Returns:
      plt.Axes: The matplotlib Axes object on which the probability
      distributions are drawn.
  """
  if len(df_list) != len(exp_names):
    raise ValueError('The lengths of df_list and exp_names must match.')

  # Compute the union of all unique base_lr values across all DataFrames
  all_base_lr = sorted(
      set().union(*(set(df['base_lr'].unique()) for df in df_list))
  )

  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # Use a color cycle for differentiating experiments
  n = len(df_list)
  colors = get_colors_from_cmap(n)

  for i, (df, exp_name) in enumerate(zip(df_list, exp_names)):
    # Exclude specific columns to determine grouping parameters
    param_key_list = [
        col for col in df.columns if base_schedule_family.is_schedule_param(col)
    ]

    # Compute indices of the minimum key_metric values per group
    min_score_indices = df.groupby(param_key_list)[key_metric].idxmin()
    best_base_lr = df.loc[min_score_indices, 'base_lr']

    # Count the frequency of each base_lr among the selected best values
    value_counts = best_base_lr.value_counts().reindex(
        all_base_lr, fill_value=0
    )

    # Normalize counts to get probabilities
    total = value_counts.sum()
    if total > 0:
      probabilities = value_counts / total
    else:
      probabilities = value_counts

    # Plot the probability distribution as a line plot with markers
    ax.plot(
        all_base_lr,
        probabilities,
        marker='o',
        linestyle='-',
        color=colors[i],
        linewidth=LINE_WIDTH,
        label=exp_name,
    )

  # Set x-axis to log scale (since base_lr spans multiple orders of magnitude)
  ax.set_xscale('log')

  # Configure labels and title
  ax.set_xlabel('Base LR', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_ylabel('Probability', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_title(f'{title}', fontsize=TITLE_FONT_SIZE)

  # Set tick label font sizes
  plt.setp(ax.get_xticklabels(), fontsize=AXIS_TICK_FONT_SIZE)
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  # Place the legend outside the figure to the right
  ax.legend(
      loc='center left',
      bbox_to_anchor=(1.02, 0.5),
      prop={'size': LEGEND_FONT_SIZE},
  )

  return ax


def plot_multiple_schedules(
    schedules_data, title='Learning Rate Schedule Comparison', ax=None
) -> plt.Axes:
  """Plot multiple learning rate schedules with improved legend formatting.

  Args:
      schedules_data: List of dictionaries containing: - schedule: array of
        learning rates. - params: dictionary of parameters. - color: color for
        the plot (optional).
      title: Plot title.
      ax: Optional existing axis to plot on.

  Returns:
      The matplotlib axis on which the plot is plotted.
  """
  # Colors for the schedules
  colors = get_colors_from_cmap(len(schedules_data))

  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # Plot schedules
  for i, (data, color) in enumerate(zip(schedules_data, colors)):
    schedule = data['schedule']
    if 'color' in data:
      color = data['color']

    # Create legend label with schedule number and score
    score = data['params'].get('score', None)
    label = f'Schedule {i+1}: {score:.4f}'

    ax.plot(
        np.arange(len(schedule)),
        schedule,
        linewidth=2,
        label=label,
        color=color,
        alpha=0.8,
    )

  # Configure plot
  ax.set_xlabel('Steps', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_ylabel('Learning Rate', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_title(title, pad=20, fontsize=TITLE_FONT_SIZE)
  ax.grid(True, alpha=0.3)
  ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

  # Set tick label font sizes
  plt.setp(ax.get_xticklabels(), fontsize=AXIS_TICK_FONT_SIZE)
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  ax.legend(
      loc='center left',
      bbox_to_anchor=(1.02, 0.5),
      prop={'size': LEGEND_FONT_SIZE},
  )

  return ax


def plot_ecdf_with_legend(
    schedule_df_list,
    legend_names,
    key_metric,
    is_x_log_scale=False,
    is_y_log_scale=False,
    title='',
    zoom_x_threshold_ratio=None,
    ax=None,
) -> plt.Axes:
  """Calculates and plots the ECDF of the values corresponding to key_metric.

  Uses enhanced line widths and colors for better visibility.

  Args:
      schedule_df_list: A list of reduced_exp_df (pandas.DataFrame) for each
        experiment.
      legend_names: A list of legend names corresponding to each experiment.
      key_metric: The column name used for calculating the ECDF (e.g.,
        'score_median').
      is_x_log_scale: Whether to use a log scale for the x-axis.
      is_y_log_scale: Whether to use a log scale for the y-axis.
      title: The string to be appended to the plot title.
      zoom_x_threshold_ratio: If not None, the x-axis will be zoomed to fit the
        data within a threshold ratio of the range.
      ax: An existing matplotlib Axes. If None, a new one is created.

  Returns:
      The matplotlib Axes object on which the plot is drawn.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=(9, 6))

  if len(schedule_df_list) != len(legend_names):
    raise ValueError(
        'The lengths of schedule_df_list and legend_names do not match.'
    )

  # Create a color cycle for better visibility using a distinct colormap
  colors = get_colors_from_cmap(len(schedule_df_list))

  # For each DataFrame, calculate and plot the ECDF
  for i, (df, legend) in enumerate(zip(schedule_df_list, legend_names)):
    # Convert the data from the specified column into a list
    x_list = df[key_metric].tolist()
    # Sort the data and calculate the ECDF
    sorted_x_list = np.sort(x_list)
    ecdf = np.arange(1, len(sorted_x_list) + 1) / len(sorted_x_list)
    # Plot the ECDF using a step plot with unified line width and assigned color
    ax.step(
        sorted_x_list,
        ecdf,
        where='post',
        label=legend,
        linewidth=LINE_WIDTH,
        color=colors[i],
    )

  # Decorate the plot
  if is_x_log_scale:
    ax.set_xscale('log')
  if is_y_log_scale:
    ax.set_yscale('log')
  ax.set_title(f'{title}', fontsize=TITLE_FONT_SIZE)
  ax.set_xlabel(key_metric, fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_ylabel(
      'Cumulative Probability - Density', fontsize=AXIS_LABEL_FONT_SIZE
  )
  ax.grid(True)

  # Set tick label font sizes
  plt.setp(ax.get_xticklabels(), fontsize=AXIS_TICK_FONT_SIZE)
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  ax.legend(
      loc='center left',
      bbox_to_anchor=(1.02, 0.5),
      prop={'size': LEGEND_FONT_SIZE},
  )

  if zoom_x_threshold_ratio is not None:
    all_x = np.concatenate(
        [np.sort(df[key_metric].tolist()) for df in schedule_df_list]
    )
    x_min, x_max = np.min(all_x), np.max(all_x)
    x_threshold = x_min + zoom_x_threshold_ratio * (x_max - x_min)
    ax.set_xlim(x_min, x_threshold)
    ax.set_ylim(0, 0.5)

  return ax


def plot_multiple_schedules_with_metadata(
    schedule_array,
    schedule_metadata,
    legend_names,
    key_metric='score_mean',
    title='Learning Rate Schedule Comparison',
    ax=None,
    notation='floating_point',
    score_digits=None,
    error_digits=None,
    normalized_x_axis=False,
    normalized_y_axis=False,
    **plot_kwargs,
) -> plt.Axes:
  """Plots multiple learning rate schedules with enhanced line widths and a clear legend.

  using metadata information. Colors and line thickness have been adjusted for
  better clarity.

  Args:
      schedule_array: List of arrays, each containing learning rates.
      schedule_metadata: List of dictionaries, each containing metadata such as
        'params' and an optional 'color'.
      legend_names: List of names for each schedule.
      key_metric: Metric to use for the legend ('score_mean' or 'score_median').
      title: Plot title.
      ax: Optional existing axis to plot on.
      notation: Notation to use for reported scores ('floating_point' or 'sci').
      score_digits: Number of digits to display for scores.
      error_digits: Number of digits to display for errors.
      normalized_x_axis: Whether to normalize the x-axis to [0, 1].
      normalized_y_axis: Whether to normalize the y-axis to [0, 1].
      **plot_kwargs: Additional keyword arguments to pass to the plot function.

  Returns:
      The matplotlib Axes object on which the plot is plotted.
  """
  # Ensure schedule_array and schedule_metadata have the same length
  if len(schedule_array) != len(schedule_metadata):
    raise ValueError(
        'schedule_array and schedule_metadata must have the same length.'
    )
  if score_digits is None:
    if notation == 'floating_point':
      score_digits = 1
    elif notation == 'sci':
      score_digits = 2
  if error_digits is None:
    if notation == 'floating_point':
      error_digits = 2
    elif notation == 'sci':
      error_digits = 0

  if not legend_names:
    legend_names = [f'Schedule {i+1}' for i in range(len(schedule_array))]

  # Create a color cycle using a distinct colormap for better visibility
  colors = get_colors_from_cmap(len(schedule_array))

  if ax is None:
    _, ax = plt.subplots(figsize=(8, 6))

  # Plot schedules
  for i, (schedule, metadata, color) in enumerate(
      zip(schedule_array, schedule_metadata, colors)
  ):
    if 'color' in metadata:
      color = metadata['color']

    # Create legend label with schedule number and score
    metrics = metadata.get(key_metric, None)
    if key_metric == 'score_median':
      score_std = metadata.get('score_median_error', None)
    elif key_metric == 'score_mean':
      score_std = metadata.get('score_std_error', None)
    else:
      score_std = None
    if metrics is None:
      label = legend_names[i]
    elif score_std is None:
      if notation == 'floating_point':
        label = f'{legend_names[i]}: {metrics:.{score_digits}f}'
      elif notation == 'sci':
        label = f'{legend_names[i]}: {metrics:.{score_digits}e}'
      else:
        raise ValueError(f'Unsupported notation: {notation}')
    else:
      if notation == 'floating_point':
        label = (
            f'{legend_names[i]}: {metrics:.{score_digits}f} ±'
            f' {score_std:.{error_digits}f}'
        )
      elif notation == 'sci':
        label = (
            f'{legend_names[i]}: {metrics:.{score_digits}e} ±'
            f' {score_std:.{error_digits}e}'
        )
      else:
        raise ValueError(f'Unsupported notation: {notation}')
    if normalized_x_axis:
      xs = np.linspace(0, 1, len(schedule))
    else:
      xs = np.arange(len(schedule))
    if normalized_y_axis:
      ys = schedule/np.max(schedule)
    else:
      ys = schedule
    ax.plot(
        xs,
        ys,
        linewidth=LINE_WIDTH,
        label=label,
        color=color,
        alpha=0.8,
        **plot_kwargs,
    )

  # Configure plot
  if normalized_x_axis:
    ax.set_xlabel('Training fraction', fontsize=AXIS_LABEL_FONT_SIZE)
  else:
    ax.set_xlabel('Steps', fontsize=AXIS_LABEL_FONT_SIZE)
  if normalized_y_axis:
    ax.set_ylabel('Relative learning rate', fontsize=AXIS_LABEL_FONT_SIZE)
  else:
    ax.set_ylabel('Learning Rate', fontsize=AXIS_LABEL_FONT_SIZE)
  ax.set_title(title, pad=20, fontsize=TITLE_FONT_SIZE)
  ax.grid(True, alpha=0.3)
  ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

  # Set tick label font sizes
  plt.setp(ax.get_xticklabels(), fontsize=AXIS_TICK_FONT_SIZE)
  plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONT_SIZE)

  ax.legend(
      loc='center left',
      bbox_to_anchor=(1.02, 0.5),
      prop={'size': LEGEND_FONT_SIZE},
  )

  return ax


def plot_base_lr_heatmap(
    exp_names,
    sched_records,
    score_dfs,
    base_lrs,
    fig=None,
    ax=None,
    schedules_per_family=1,
    metric='score_median',
    plot_config=None,
    metric_label=None,
):
  """Plots a heatmap of scores vs. base_lr values for multiple schedules.

  Assumes that the same base_lr values are used for all schedules.

  Args:
      exp_names: List of experiment names corresponding to each DataFrame.
      sched_records: Dictionary of schedule param dfs, keyed by experiment name.
      score_dfs: Dictionary of score DataFrames, keyed by experiment name.
      base_lrs: List of base_lr values to plot.
      fig: Optional existing figure to plot on.
      ax: Optional existing axis to plot on.
      schedules_per_family: Number of schedules per family.
      metric: Metric to use for plotting (default: 'score_median').
      plot_config: Optional dictionary containing plot configuration parameters.
        If None, default values are used.
      metric_label: Optional label for colorbar. If None, the metric name is
        used.

  Returns:
      The matplotlib Axes object on which the plot is drawn
      and matrix of scores.
  """
  if plot_config is None:
    plot_config = {}
  tick_period = plot_config.get('tick_period', 4)
  tick_offset = plot_config.get('tick_offset', 1)
  figsize = plot_config.get('figsize', (8, 5))
  cmap = plot_config.get('cmap', 'magma_r')
  cmap_min = plot_config.get('cmap_min', 0.05)
  cmap_max = plot_config.get('cmap_max', 1.0)
  cmap_bad_color = plot_config.get('cmap_bad_color', 'black')

  if metric_label is None:
    metric_label = metric
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  # Label schedules
  if schedules_per_family == 1:
    sched_names = [exp_name for exp_name in exp_names]
  else:
    sched_names_list = [
        [exp_name + '_' + str(i) for i in range(schedules_per_family)]
        for exp_name in exp_names
    ]
    sched_names = []
    for sublist in sched_names_list:
      sched_names.extend(sublist)

  # Extract scores into matrix
  score_mat = np.zeros((
      len(exp_names) * schedules_per_family,
      len(base_lrs),
  ))

  score_idx = 0
  for exp_name in exp_names:
    for i in range(len(sched_records[exp_name])):
      params = sched_records[exp_name].iloc[[i]]  # Single param set as DF
      sub_df = pandas_util.get_scores_from_schedule_shapes(
          score_dfs[exp_name], params
      )
      score_mat[score_idx, :] = sub_df.sort_values(by='base_lr')[metric]
      score_idx += 1
  # Plot heatmap
  cmap = plt.get_cmap(cmap)
  cmap.set_bad(cmap_bad_color)
  image = ax.imshow(
      score_mat,
      norm=plt_colors.LogNorm(vmin=cmap_min, vmax=cmap_max),
      aspect='auto',
      cmap=cmap,
  )
  ax.grid(False)  # Turn off grid lines on heatmap

  cbar = fig.colorbar(image, ax=ax)
  cbar.set_label(metric_label)

  ax.set_yticks(np.arange(len(sched_names)))
  ax.set_yticklabels(sched_names)

  ax.set_xticks(np.arange(tick_offset, len(base_lrs), tick_period))
  ax.set_xticklabels(
      [f'{x_val:.1e}' for x_val in base_lrs[tick_offset::tick_period]]
  )

  plt.xlabel('Base learning rate')
  plt.ylabel('Schedule')

  return ax, score_mat


def _order_keys_by_prefix(d, prefix_order):
  """Partially order dict keys based on an ordering over prefixes."""
  ordered = collections.OrderedDict(d)
  for prefix in reversed(prefix_order):
    for k in d:
      if k.startswith(prefix):
        ordered.move_to_end(k, last=False)
  return ordered


def canonicalize_dict_keys(d):
  """Canonicalize schedule dictionary key order."""
  prefix_order = ['con', 'cos', 'tps', 'tpl', 'sqrt', 'rex', 'snm']
  return _order_keys_by_prefix(d, prefix_order)


def make_single_descent_plot(
    sorted_sweep_df, sweep_param, init_param_val, ax=None, plot_config=None
):
  """Plot coordinate descent for a single parameter sweep.

  Args:
    sorted_sweep_df: Dataframe of scores for a single parameter sweep, sorted by
      the parameter being swept.
    sweep_param: Name of the parameter being swept.
    init_param_val: Initial value of the parameter being swept.
    ax: Optional existing axis to plot on.
    plot_config: Optional dictionary of plotting configuration parameters. If
      None, default values are used.

  Returns:
    The matplotlib Axes object on which the plot is drawn.
  """
  # Set up plotting configurations
  if plot_config is None:
    plot_config = {}
  figsize = plot_config.get('figsize', (6, 4))
  legend_on = plot_config.get('legend_on', False)
  x_label_fontsize = plot_config.get('x_label_fontsize', 17)
  y_label_fontsize = plot_config.get('y_label_fontsize', 17)
  major_tick_size = plot_config.get('major_tick_size', 15)
  minor_tick_size = plot_config.get('minor_tick_size', 14)
  title_fontsize = plot_config.get('title_fontsize', 18)
  legend_fontsize = plot_config.get('legend_fontsize', 11)
  score_name = plot_config.get('score_name', 'Train error')

  param_name = sweep_param[2:]
  # New axis if needed
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)

  color = 'C0'
  ax.set_xlabel(param_name, fontsize=x_label_fontsize)
  ax.set_ylabel(f'{score_name} (median)',
                color=color, fontsize=y_label_fontsize)
  yerr = [
      sorted_sweep_df['score_median'] - sorted_sweep_df['ci_lower'],
      sorted_sweep_df['ci_upper'] - sorted_sweep_df['score_median'],
  ]
  ax.errorbar(
      sorted_sweep_df[sweep_param],
      sorted_sweep_df['score_median'],
      yerr=yerr,
      marker='o',
      capsize=5,
      color=color,
  )
  ax.tick_params(axis='y', labelcolor=color)

  ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
  ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
  # Create second y-axis for base_lr

  ax2 = ax.twinx()
  color = 'C1'
  ax2.set_ylabel(
      'Base learning rate',
      color=color,
      fontsize=y_label_fontsize,
  )
  ax2.tick_params(axis='both', which='major', labelsize=major_tick_size)
  ax2.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
  ax2.plot(
      sorted_sweep_df[sweep_param],
      sorted_sweep_df['base_lr'],
      color=color,
      marker='s',
      zorder=0,
  )
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_yscale('log')

  # Add vertical dashed line at base_param (median of x_param)
  base_param = init_param_val
  ax.axvline(
      x=base_param,
      color='lime',
      linestyle='--',
      linewidth=5,
      alpha=0.3,
      label='Original param',
  )

  # Add legend
  lines1, labels1 = ax.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  if legend_on:
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',
        fontsize=legend_fontsize,
    )

  # Set title
  title = f'Coordinate descent on {param_name}'

  ax.set_title(title, fontsize=title_fontsize, y=1.03)
  ax.grid(True)
  return ax


def make_coordinate_descent_plots(
    point_df_dict,
    initial_param_dict,
    fig=None,
    plot_config=None,
    subplot_cols=None,
):
  """Plot coordinate descent for all parameter sweeps from one initialization.

  Args:
    point_df_dict: Dictionary of dataframes of scores for each parameter sweep,
      keyed by the parameter being swept.
    initial_param_dict: Dictionary of initial parameter values for each sweep.
    fig: Optional existing figure to plot on.
    plot_config: Optional dictionary of plotting configuration parameters. If
      None, default values are used.
    subplot_cols: Number of columns for subplots. If None, plots are not
      subdivided into subplots.

  Returns:
    If subplot_cols is None (sequence of independent figures), returns a list of
    matplotlib Axes objects on which
    the plots are drawn.

    If subplot_cols is not None, returns the matplotlib Figure object on which
    the grid of subplots is drawn.
  """
  if fig is not None and subplot_cols is None:
    raise ValueError('If fig is provided, subplot_cols must also be provided.')
  if plot_config is None:
    single_figsize = (6, 4)
  else:
    single_figsize = plot_config.get('single_figsize', (6, 4))
  param_list = list(initial_param_dict.keys())

  n_cols = subplot_cols
  ax_list = []  # Use only if subplot_cols is None
  n_plots = len(param_list)

  if subplot_cols is not None:
    # Dimensions of subplot
    n_rows = (n_plots - 1) // n_cols + 1
    if fig is None:
      # Create subplots
      fig, axes = plt.subplots(
          n_rows,
          n_cols,
          figsize=(single_figsize[0] * n_cols, single_figsize[1] * n_rows),
      )
    else:
      axes = fig.subplots(
          n_rows,
          n_cols,
      )
    axes = np.atleast_2d(axes)  # Ensure 2D array for consistent indexing
  else:
    axes = None

  if not param_list:  # No parameters to plot
    return None

  # One plot per parameter sweep.
  for plot_idx, sweep_param in enumerate(param_list):
    init_param_val = initial_param_dict[sweep_param]
    if subplot_cols is not None:
      row, col = divmod(plot_idx, n_cols)
      ax = axes[row, col]
    else:
      ax = None
    ax = make_single_descent_plot(
        point_df_dict[sweep_param], sweep_param, init_param_val, ax, plot_config
    )
    if subplot_cols is None:
      ax_list.append(ax)
  if subplot_cols is None:
    return ax_list
  else:
    for plot_idx in range(n_plots, n_rows * n_cols):
      row, col = divmod(plot_idx, n_cols)
      fig.delaxes(axes[row, col])
    fig.tight_layout()
    return fig


def make_multiple_coordinate_descent_plots(
    per_point_dfs, initial_params_df, plot_config=None, subplot_cols=None
):
  """Plot coordinate descent results for multiple initialization points.

  Args:
    per_point_dfs: List of dictionaries of dataframes of scores for each
      parameter sweep, keyed by the parameter being swept. Each dictionary
      corresponds to one initialization point. Same as format returned by
      pandas_util.extract_sweeps_from_results.
    initial_params_df: Dataframe of initial parameter values for each sweep.
    plot_config: Optional dictionary of plotting configuration parameters. If
      None, default values are used.
    subplot_cols: Number of columns for subplots. If None, plots are not
      subdivided into subplots.
  """
  for i, point_df_dict in enumerate(per_point_dfs):
    initial_param_dict = dict(initial_params_df.iloc[i])
    make_coordinate_descent_plots(
        point_df_dict,
        initial_param_dict,
        fig=None,
        plot_config=plot_config,
        subplot_cols=subplot_cols,
    )
