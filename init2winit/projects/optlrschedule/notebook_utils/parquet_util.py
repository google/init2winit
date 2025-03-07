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

r"""Utility functions for loading parquet files in notebooks."""

import io
from typing import List, Optional

from etils import epath
import pandas as pd


def load_parquet_file(
    path: str,
    file_name: Optional[str] = None,
    *,
    sort_by: str = 'score',
    ascending: bool = True
) -> pd.DataFrame:
  """Load a single parquet file and return it as a sorted DataFrame.

  Args:
      path: Directory path string
      file_name (optional): File name string (default: 'results.parquet')
      sort_by: Column to sort by (default: 'score')
      ascending: Sort order (default: True)

  Returns:
      pandas DataFrame
  """

  if file_name:
    path = epath.Path(path) / file_name
  else:
    path = epath.Path(path)

  # Read the file
  with path.open('rb') as in_f:
    buf = io.BytesIO(in_f.read())
    df = pd.read_parquet(buf)

  # Sort if the column exists
  if sort_by in df.columns:
    df.sort_values(by=sort_by, ascending=ascending, inplace=True)

  return df


def load_all_parquet_files(
    paths: List[str],
    file_name: Optional[str] = None,
    *,
    sort_by: str = 'score',
    ascending: bool = True
) -> pd.DataFrame:
  """Load and merge all parquet files from different paths.

  Args:
      paths: List of directory paths.
      file_name (optional): File name string (default: 'results.parquet')
      sort_by: Column to sort by (default: 'score').
      ascending: Sort order (default: True).

  Returns:
      Merged pandas DataFrame.
  """
  dfs = []

  for path in paths:
    df = load_parquet_file(
        path, file_name, sort_by=sort_by, ascending=ascending
    )
    if not df.empty:
      dfs.append(df)

  if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df
  else:
    return pd.DataFrame()
