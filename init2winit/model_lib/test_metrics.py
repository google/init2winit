# coding=utf-8
# Copyright 2022 The init2winit Authors.
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

"""Tests for metrics.py.

"""

from absl.testing import absltest
from absl.testing import parameterized
from init2winit.model_lib import metrics
import numpy as np


class MetricsTest(parameterized.TestCase):
  """Tests for metrics.py."""

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          targets=np.array([[1., 0.], [0., 1.]]),
          logits=np.array([[0.5, 0.5], [0.5, 0.5]]),
          weights=np.array([[1., 1.], [1., 1.]]),
          result=0.5),
      dict(
          testcase_name='weights',
          targets=np.array([[1., 0.,], [0., 1.], [0., 1.]]),
          logits=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.7]]),
          weights=np.array([[1., 1.], [0., 1.], [1., 0.]]),
          result=0.5))
  def test_MeanAveragePrecision(self, logits, targets, weights, result):
    """Tests the mean average precision computation."""

    average_precision = metrics.MeanAveragePrecision.from_model_output(
        logits=logits, targets=targets, weights=weights).compute()
    self.assertAlmostEqual(average_precision, result)

if __name__ == '__main__':
  absltest.main()
