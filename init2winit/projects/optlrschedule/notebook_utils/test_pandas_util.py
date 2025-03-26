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

"""Unit tests for pandas_util.py."""

from absl.testing import absltest
from init2winit.projects.optlrschedule.notebook_utils import pandas_util
import pandas as pd


class TestPandasUtil(absltest.TestCase):
  """Test cases for pandas_util.py."""

  def test_reduce_to_best_base_lrs_evaluate(self):
    """Test reduce_to_best_base_lrs on evaluate_experiment.py-style input."""
    # Create a sample DataFrame with multiple base_lr candidates
    df = pd.DataFrame([
        {
            'base_lr': 0.01,
            'score': 0.08060000091791153,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.01,
            'score': 0.06371999531984329,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.02,
            'score': 0.07231999933719635,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.02,
            'score': 0.07005999982357025,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.03,
            'score': 0.05663999915122986,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.1,
            'score': 0.10099999606609344,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
        {
            'base_lr': 0.1,
            'score': 0.06651999801397324,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
        {
            'base_lr': 0.2,
            'score': 0.07333999872207642,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
        {
            'base_lr': 0.2,
            'score': 0.07727999985218048,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
        {
            'base_lr': 0.3,
            'score': 0.08311999589204788,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
    ])
    reduced_df = pandas_util.reduce_to_best_base_lrs(df)
    expected_df = pd.DataFrame([
        {
            'base_lr': 0.03,
            'score': 0.05663999915122986,
            'p.exponent': 0.39098358154296875,
            'p.warmup_steps': 144.92100524902344,
            'rank': 0,
            'original_score_mean': 0.059893997758626936,
            'original_score_median': 0.057329997420310974,
            'original_score_std': 0.007927739898221917,
            'original_score_min': 0.05209999904036522,
            'original_score_max': 0.0737999975681305,
            'original_group_size': 10,
            'original_score_std_error': 0.002642579966073972,
            'original_score_median_error_normal': 0.0033119828304672243,
            'original_score_median_error': 0.0033119828304672243,
        },
        {
            'base_lr': 0.2,
            'score': 0.07333999872207642,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
        {
            'base_lr': 0.2,
            'score': 0.07727999985218048,
            'p.exponent': 0.16678667068481445,
            'p.warmup_steps': 73.85269165039062,
            'rank': 19,
            'original_score_mean': 0.07237199768424034,
            'original_score_median': 0.06612999737262726,
            'original_score_std': 0.014489456942578346,
            'original_score_min': 0.058720000088214874,
            'original_score_max': 0.10077999532222748,
            'original_group_size': 10,
            'original_score_std_error': 0.004829818980859449,
            'original_score_median_error_normal': 0.006053280409385888,
            'original_score_median_error': 0.006053280409385888,
        },
    ])
    print(reduced_df)
    print(expected_df)
    self.assertTrue(reduced_df.equals(expected_df))

  def test_reduce_to_best_base_lrs_search(self):
    """Test reduce_to_best_base_lrs function on run_search.py-style input."""
    # Create a sample DataFrame with multiple base_lr candidates
    df = pd.DataFrame([
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.25648000836372375,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.25562000274658203,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.2488199919462204,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921999871730804,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 0,
            'base_lr': 0.003,
            'score': 0.24543999135494232,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.8999999761581421,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.8999999761581421,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
        {
            'generation': 4,
            'base_lr': 0.2,
            'score': 0.94,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
        {
            'generation': 4,
            'base_lr': 0.2,
            'score': 0.89,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
        {
            'generation': 4,
            'base_lr': 0.3,
            'score': 0.7,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
    ])

    reduced_df = pandas_util.reduce_to_best_base_lrs(df)
    expected_df = pd.DataFrame([
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.2488199919462204,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921999871730804,
            'p.exponent': 0.10225892066955566,
            'p.warmup_steps': 155.4968719482422,
        },
        {
            'generation': 4,
            'base_lr': 0.3,
            'score': 0.7,
            'p.exponent': 0.0667257308959961,
            'p.warmup_steps': 209.09278869628906,
        },
    ])
    self.assertTrue(reduced_df.equals(expected_df))


if __name__ == '__main__':
  absltest.main()
