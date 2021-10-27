# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

"""Unit tests for trainer.py.

"""

import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from flax import optim as optimizers
from flax.deprecated import nn
from init2winit import utils
from jax import test_util as jtu
import numpy as np

FLAGS = flags.FLAGS


class TrainingMetricsTest(jtu.JaxTestCase):
  """Tests the logged statistics from training_metrics_grabber."""

  def setUp(self):
    super(TrainingMetricsTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(TrainingMetricsTest, self).tearDown()

  def test_grad_var(self):
    model_size = 10
    example_grads = [{
        'layer1': np.ones(model_size),
        'layer2': 3 * np.ones(model_size)
    }, {
        'layer1': 2 * np.ones(model_size),
        'layer2': np.ones(model_size)
    }]
    eval_config = {'ema_beta': 0.5}
    training_metrics_grabber = utils.TrainingMetricsGrabber.create(
        example_grads[0], eval_config)

    # For the purposes of this test, we create fake optimizers to satisfy
    # metrics grabber API.
    fake_model = nn.Model(None, example_grads[0])
    new_optimizer = optimizers.GradientDescent(
        learning_rate=None).create(fake_model)
    old_optimizer = optimizers.GradientDescent(
        learning_rate=None).create(fake_model)

    for grad in example_grads:
      training_metrics_grabber = training_metrics_grabber.update(
          grad, old_optimizer, new_optimizer)

    for layer in ['layer1', 'layer2']:
      expected_grad_ema = 1 / 4 * np.zeros(model_size) + 1 / 4 * example_grads[
          0][layer] + 1 / 2 * example_grads[1][layer]

      self.assertArraysAllClose(expected_grad_ema,
                                training_metrics_grabber.state[layer].grad_ema)

if __name__ == '__main__':
  absltest.main()
