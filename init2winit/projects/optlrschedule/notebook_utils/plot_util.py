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

"""Utility functions for plotting data in notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Global font sizes for plots (change these values to update font sizes
# globally)
BAR_TEXT_FONT_SIZE = 16
BAR_ALPHA = 0.5
AXIS_LABEL_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 16
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
LINE_WIDTH = 3.0  # Unified line width for all plots

LR_SCHEDULE_NAME_MAP = {
    'constant': 'Constant',
    'cosine': 'Generalized Cosine',
    'twopointslinear': 'Two-Point Linear',
    'twopointsspline': 'Two-Point Spline',
    'smoothnonmonotonic': 'Smooth Non-Monotonic',
    'sqrt': 'Square Root',
    'rex': 'Generalized REX',
}


def map_schedule_name_to_label(schedule_type: str) -> str:
  """Maps the given lr schedule type name to its human-readable string.

  Args:
      schedule_type (str): The lr schedule type name (e.g., 'constant',
        'cosine', etc.)

  Returns:
      str: The corresponding human-readable lr schedule name. If the given
      schedule_type is not found,
           the original schedule_type is returned.
  """
  # Convert the input to lowercase to ensure case-insensitive matching
  # and return the mapped name.
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
  # Determine parameter columns by excluding specific columns
  excluded_cols = ['generation', 'base_lr', 'score', 'group_size']
  param_key_list = [
      col
      for col in df.columns
      if col not in excluded_cols
      and not col.startswith('score')
      and not col.startswith('original')
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
  colors = plt.cm.tab10(np.linspace(0, 1, n))

  # For each DataFrame, compute and plot its best base_lr histogram
  for i, (df, exp_name) in enumerate(zip(df_list, exp_names)):
    # Determine parameter columns by excluding specific columns
    excluded_cols = ['generation', 'base_lr', 'score', 'group_size']
    param_key_list = [
        col
        for col in df.columns
        if col not in excluded_cols
        and not col.startswith('score')
        and not col.startswith('original')
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
  colors = plt.cm.tab10(np.linspace(0, 1, n))

  for i, (df, exp_name) in enumerate(zip(df_list, exp_names)):
    # Exclude specific columns to determine grouping parameters
    excluded_cols = ['generation', 'base_lr', 'score', 'group_size']
    param_key_list = [
        col
        for col in df.columns
        if col not in excluded_cols
        and not col.startswith('score')
        and not col.startswith('original')
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
  colors = plt.cm.tab20(np.linspace(0, 1, len(schedules_data)))

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
  colors = plt.cm.Set1(np.linspace(0, 1, len(schedule_df_list)))

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

  Returns:
      The matplotlib Axes object on which the plot is plotted.
  """
  # Ensure schedule_array and schedule_metadata have the same length
  if len(schedule_array) != len(schedule_metadata):
    raise ValueError(
        'schedule_array and schedule_metadata must have the same length.'
    )

  if not legend_names:
    legend_names = [f'Schedule {i+1}' for i in range(len(schedule_array))]

  # Create a color cycle using a distinct colormap for better visibility
  colors = plt.cm.Set2(np.linspace(0, 1, len(schedule_array)))

  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

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
    else:
      score_std = metadata.get('score_std', None)
    label = (
        f'{legend_names[i]}: {metrics:.4f} ± {score_std:.4f}'
        if metrics is not None
        else legend_names[i]
    )

    ax.plot(
        np.arange(len(schedule)),
        schedule,
        linewidth=LINE_WIDTH,
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
