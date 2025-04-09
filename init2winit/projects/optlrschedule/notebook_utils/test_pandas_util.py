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
import numpy as np
import pandas as pd
import pandas.testing as pd_testing


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
    pd_testing.assert_frame_equal(reduced_df, expected_df, check_exact=True)

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
    pd_testing.assert_frame_equal(reduced_df, expected_df, check_exact=True)

  def test_get_all_scores_from_param_list(self):
    """Test get_all_scores_from_param_list function."""
    df = pd.DataFrame([
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.26,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.31,
            'xid_history': (12347,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.25,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.003,
            'score': 0.25,
            'xid_history': (82345,),
            'p.exponent': 0.2534,
            'p.warmup_steps': 180.93,
        },
        {
            'generation': 0,
            'base_lr': 0.003,
            'score': 0.30,
            'xid_history': (82345,),
            'p.exponent': 0.2534,
            'p.warmup_steps': 180.93,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.899,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.543,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.452,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
    ])
    # Test without reduction over seeds
    no_stats_param_df = pd.DataFrame([
        {
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
    ])
    expected_no_stats_df = pd.DataFrame([
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.26,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.001,
            'score': 0.31,
            'xid_history': (12347,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.25,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 0,
            'base_lr': 0.002,
            'score': 0.14921,
            'xid_history': (12345,),
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.899,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.543,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
        {
            'generation': 4,
            'base_lr': 0.1,
            'score': 0.452,
            'xid_history': (12345,),
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
    ])
    extracted_no_stats_df = pandas_util.get_scores_from_schedule_shapes(
        df, no_stats_param_df, reduce_seeds=False
    )
    pd_testing.assert_frame_equal(
        extracted_no_stats_df, expected_no_stats_df, check_exact=False
    )

    # Test with reduction over seeds
    stats_param_df = pd.DataFrame([{
        'p.exponent': 0.2534,
        'p.warmup_steps': 180.93,
    }])
    expected_with_stats_df = pd.DataFrame([
        {
            'base_lr': 0.003,
            'p.exponent': 0.2534,
            'p.warmup_steps': 180.93,
            'xid_history': (82345,),
            'score_mean': 0.275,
            'score_median': 0.275,
            'score_std': 0.03535533905932736,
            'score_min': 0.25,
            'score_max': 0.30,
            'group_size': 2,
            'score_std_error': 0.025,
            'score_median_error_normal': 0.03133285343288749,
            'score_median_error': 0.03133285343288748,
        },
    ])
    extracted_with_stats_df = pandas_util.get_scores_from_schedule_shapes(
        df, stats_param_df, reduce_seeds=True
    )
    print('Extracted:', extracted_with_stats_df)
    print('Expected:', expected_with_stats_df)
    pd_testing.assert_frame_equal(
        extracted_with_stats_df, expected_with_stats_df, check_exact=False
    )

  def test_extract_sweeps_from_results(self):
    """Test extract_sweeps_from_results function."""
    # Create a DataFrame with multiple parameter values
    initial_params_df = pd.DataFrame([
        {
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
        },
        {
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
        },
    ])

    # Dataframe of sweep results

    df = pd.DataFrame([
        {
            'base_lr': 0.001,
            'score': 0.26,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.001,
            'score': 0.30,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.001,
            'score': 0.28,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.002,
            'score': 0.86,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.002,
            'score': 0.99,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.005,
            'score': 0.26,
            'p.exponent': 0.0022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.005,
            'score': 0.24,
            'p.exponent': 0.0022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.005,
            'score': 0.28,
            'p.exponent': 0.0022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.005,
            'score': 0.22,
            'p.exponent': 0.0022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
        },
        {
            'base_lr': 0.004,
            'score': 0.26,
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
            'xid_history': (52345,),
        },
        {
            'base_lr': 0.004,
            'score': 0.24,
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
            'xid_history': (52345,),
        },
        {
            'base_lr': 0.004,
            'score': 0.28,
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
            'xid_history': (52345,),
        },
        {
            'base_lr': 0.004,
            'score': 0.22,
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
            'xid_history': (52345,),
        },
    ])

    # Total dataframe
    expected_total_df = pd.DataFrame([
        {
            'base_lr': 0.001,
            'p.exponent': 0.1022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
            'score_mean': 0.28,
            'score_median': 0.28,
            'score_median_error': (
                np.float64(0.014401937285648684),
                np.float64(0.26),
                np.float64(0.3),
            ),
            'score_std': 0.019999999999999976,
            'score_min': 0.26,
            'score_max': 0.3,
            'group_size': 3,
            'score_std_error': 0.011547005383792502,
            'score_median_error_normal': 0.014472025091165335,
            'ci_lower': 0.26,
            'ci_upper': 0.3,
        },
        {
            'base_lr': 0.004,
            'p.exponent': 0.06672,
            'p.warmup_steps': 209.092,
            'xid_history': (52345,),
            'score_mean': 0.25,
            'score_median': 0.25,
            'score_median_error': (
                np.float64(0.015050726608278553),
                np.float64(0.22),
                np.float64(0.28),
            ),
            'score_std': 0.025819888974716126,
            'score_min': 0.22,
            'score_max': 0.28,
            'group_size': 4,
            'score_std_error': 0.012909944487358063,
            'score_median_error_normal': 0.016180215937964166,
            'ci_lower': 0.22,
            'ci_upper': 0.28,
        },
        {
            'base_lr': 0.005,
            'p.exponent': 0.0022,
            'p.warmup_steps': 155.49,
            'xid_history': (82345,),
            'score_mean': 0.25,
            'score_median': 0.25,
            'score_median_error': (
                np.float64(0.015111278406326438),
                np.float64(0.22),
                np.float64(0.28),
            ),
            'score_std': 0.025819888974716126,
            'score_min': 0.22,
            'score_max': 0.28,
            'group_size': 4,
            'score_std_error': 0.012909944487358063,
            'score_median_error_normal': 0.016180215937964166,
            'ci_lower': 0.22,
            'ci_upper': 0.28,
        },
    ])

    # Sweep dfs
    expected_sweep_df_list = []

    expected_sweep_df_list.append({
        'p.exponent': pd.DataFrame([
            {
                'base_lr': 0.005,
                'p.exponent': 0.0022,
                'p.warmup_steps': 155.49,
                'xid_history': (82345,),
                'score_mean': 0.25,
                'score_median': 0.25,
                'score_median_error': (
                    np.float64(0.015111278406326438),
                    np.float64(0.22),
                    np.float64(0.28),
                ),
                'score_std': 0.025819888974716126,
                'score_min': 0.22,
                'score_max': 0.28,
                'group_size': 4,
                'score_std_error': 0.012909944487358063,
                'score_median_error_normal': 0.016180215937964166,
                'ci_lower': 0.22,
                'ci_upper': 0.28,
            },
            {
                'base_lr': 0.001,
                'p.exponent': 0.1022,
                'p.warmup_steps': 155.49,
                'xid_history': (82345,),
                'score_mean': 0.28,
                'score_median': 0.28,
                'score_median_error': (
                    np.float64(0.014401937285648684),
                    np.float64(0.26),
                    np.float64(0.3),
                ),
                'score_std': 0.019999999999999976,
                'score_min': 0.26,
                'score_max': 0.3,
                'group_size': 3,
                'score_std_error': 0.011547005383792502,
                'score_median_error_normal': 0.014472025091165335,
                'ci_lower': 0.26,
                'ci_upper': 0.3,
            },
        ]),
        'p.warmup_steps': pd.DataFrame([
            {
                'base_lr': 0.001,
                'p.exponent': 0.1022,
                'p.warmup_steps': 155.49,
                'xid_history': (82345,),
                'score_mean': 0.28,
                'score_median': 0.28,
                'score_median_error': (
                    np.float64(0.014401937285648684),
                    np.float64(0.26),
                    np.float64(0.3),
                ),
                'score_std': 0.019999999999999976,
                'score_min': 0.26,
                'score_max': 0.3,
                'group_size': 3,
                'score_std_error': 0.011547005383792502,
                'score_median_error_normal': 0.014472025091165335,
                'ci_lower': 0.26,
                'ci_upper': 0.3,
            },
        ]),
    })

    expected_sweep_df_list.append({
        'p.exponent': pd.DataFrame([
            {
                'base_lr': 0.004,
                'p.exponent': 0.06672,
                'p.warmup_steps': 209.092,
                'xid_history': (52345,),
                'score_mean': 0.25,
                'score_median': 0.25,
                'score_median_error': (
                    np.float64(0.015050726608278553),
                    np.float64(0.22),
                    np.float64(0.28),
                ),
                'score_std': 0.025819888974716126,
                'score_min': 0.22,
                'score_max': 0.28,
                'group_size': 4,
                'score_std_error': 0.012909944487358063,
                'score_median_error_normal': 0.016180215937964166,
                'ci_lower': 0.22,
                'ci_upper': 0.28,
            },
        ]),
        'p.warmup_steps': pd.DataFrame([
            {
                'base_lr': 0.004,
                'p.exponent': 0.06672,
                'p.warmup_steps': 209.092,
                'xid_history': (52345,),
                'score_mean': 0.25,
                'score_median': 0.25,
                'score_median_error': (
                    np.float64(0.015050726608278553),
                    np.float64(0.22),
                    np.float64(0.28),
                ),
                'score_std': 0.025819888974716126,
                'score_min': 0.22,
                'score_max': 0.28,
                'group_size': 4,
                'score_std_error': 0.012909944487358063,
                'score_median_error_normal': 0.016180215937964166,
                'ci_lower': 0.22,
                'ci_upper': 0.28,
            },
        ]),
    })

    total_df, sweep_dfs_list = pandas_util.extract_sweeps_from_results(
        df,
        initial_params_df,
        ci_config={
            'return_ci': True,
            'confidence_level': 0.95,
            'n_resamples': 10000,
        },
    )

    # Check total dataframe
    pd_testing.assert_frame_equal(
        expected_total_df, total_df, check_exact=False, rtol=1e-2, atol=1e-3
    )

    # Check sweep df list
    self.assertLen(sweep_dfs_list, len(expected_sweep_df_list))
    for expected_sweep_dict, actual_sweep_dict in zip(
        expected_sweep_df_list, sweep_dfs_list
    ):
      # Check that key lists are the same in each dictionary
      self.assertSequenceEqual(
          expected_sweep_dict.keys(), actual_sweep_dict.keys()
      )
      for param in expected_sweep_dict.keys():
        print('Checking param:', param)
        pd_testing.assert_frame_equal(
            expected_sweep_dict[param],
            actual_sweep_dict[param],
            check_exact=False,
            rtol=1e-2,
            atol=1e-3,
        )


if __name__ == '__main__':
  absltest.main()
