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


def plot_best_base_lr_histgram(
    df: pd.DataFrame,
    exp_name: str,
    title: str,
    key_metric: str,
    ax: plt.Axes = None,
) -> plt.Axes:
  """plot best base_lr histogram.

  Args:
    df: Dataframe with base_lr and score columns
    exp_name: Experiment name for plot title
    title: Plot title
    key_metric: Metric to use for selecting the best base_lr (e.g., 'score')
    ax: Optional existing axis to plot on.

  Returns:
    The matplotlib axis on which the plot is plotted.
  """
  # Extract base_lr values with minimum scores from grouped data
  key_list = df.keys()
  excluded_cols = ['generation', 'base_lr', 'score', 'group_size']
  param_key_list = [
      x
      for x in key_list
      if x not in excluded_cols
      and not x.startswith('score')
      and not x.startswith('original')
  ]
  min_score_indices = df.groupby(param_key_list)[key_metric].idxmin()
  base_lr_values = df.loc[min_score_indices, 'base_lr']

  # Get all unique base_lr values from the dataframe
  all_base_lr = sorted(df['base_lr'].unique())

  # Count frequencies of base_lr values that actually appeared
  value_counts = pd.Series(0, index=all_base_lr)
  counts = base_lr_values.value_counts()
  value_counts[counts.index] = counts

  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # Create bar plot
  bars = ax.bar(range(len(value_counts)), value_counts.values)

  for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
    )

  # Set x-axis ticks (show all base_lr values)
  ax.set_xticks(range(len(value_counts)))
  ax.set_xticklabels(
      [f'{x:.6f}' for x in value_counts.index], rotation=45, ha='right'
  )

  # Labels
  ax.set_xlabel('base_lr')
  ax.set_ylabel('Count')
  ax.set_title(f'base_lr Distribution / {exp_name} / {title}')

  return ax


def plot_multiple_schedules(
    schedules_data, title='Learning Rate Schedule Comparison', ax=None
) -> plt.Axes:
  """Plot multiple learning rate schedules with improved legend formatting.

  Args:
      schedules_data: List of dictionaries containing: - schedule: array of
        learning rates - params: dictionary of parameters - color: color for the
        plot (optional)
      title: Plot title
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
  ax.set_xlabel('Steps')
  ax.set_ylabel('Learning Rate')
  ax.set_title(title, pad=20)
  ax.grid(True, alpha=0.3)
  ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

  ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

  return ax


def plot_multiple_schedules_with_metadata(
    schedule_array,
    schedule_metadata,
    legend_names,
    key_metric='score_mean',
    title='Learning Rate Schedule Comparison',
    ax=None,
) -> plt.Axes:
  """Plot multiple learning rate schedules with improved legend formatting.

  Args:
      schedule_array: List of arrays, each containing learning rates.
      schedule_metadata: List of dictionaries, each containing metadata such as
        'params' and optional 'color'.
      legend_names: List of names for each schedule.
      key_metric: Metric to use for legend ('score_mean' or 'score_median').
      title: Plot title.
      ax: Optional existing axis to plot on.

  Returns:
      The matplotlib axis on which the plot is plotted.
  """

  # Ensure schedule_array and schedule_metadata have the same length
  if len(schedule_array) != len(schedule_metadata):
    raise ValueError(
        'schedule_array and schedule_metadata must have the same length.'
    )

  if not legend_names:
    legend_names = [f'Schedule {i+1}' for i in range(len(schedule_array))]

  # Colors for the schedules
  colors = plt.cm.tab20(np.linspace(0, 1, len(schedule_array)))

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
    score_std = metadata.get('score_std', None)
    label = (
        f'{legend_names[i]}: {metrics:.4f} ± {score_std:.4f}'
        if metrics is not None
        else legend_names[i]
    )

    ax.plot(
        np.arange(len(schedule)),
        schedule,
        linewidth=2,
        label=label,
        color=color,
        alpha=0.8,
    )

  # Configure plot
  ax.set_xlabel('Steps')
  ax.set_ylabel('Learning Rate')
  ax.set_title(title, pad=20)
  ax.grid(True, alpha=0.3)
  ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

  ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

  return ax
