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

"""Unit tests for trainer.py.

"""

import shutil
import tempfile

from absl.testing import absltest
import flax
from init2winit.hessian import hessian_eval
import jax.numpy as jnp
import numpy as np


class RunLanczosTest(absltest.TestCase):
  """Tests run_lanczos.py."""

  def setUp(self):
    super(RunLanczosTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(RunLanczosTest, self).tearDown()

  def test_block_hessian(self):
    """Test block_hessian code on a low rank factorization problem.

    See Example 1.2 in https://arxiv.org/abs/2202.00980.
    """
    full_dim = 10
    low_rank_dim = 3

    # Make the init unbalanced
    params = {'AA': {}, 'AB': {}}
    a_init_scale = 10.0
    b_init_scale = .1

    # We make params nested as some errors with flax.unfreeze were only
    # surfaced for nested dictionaries.
    params['AA']['inner'] = jnp.array(
        np.random.normal(scale=a_init_scale, size=(full_dim, low_rank_dim)))
    params['AB']['inner'] = jnp.array(
        np.random.normal(scale=b_init_scale, size=(full_dim, low_rank_dim)))

    # hessian eval pmaps by default, so replicate params even for cpu tests.
    rep_params = flax.jax_utils.replicate(params)

    # True matrix factorization
    true_a = jnp.array(np.random.normal(size=(full_dim, low_rank_dim)))
    true_b = jnp.array(np.random.normal(size=(full_dim, low_rank_dim)))
    y = jnp.dot(true_a, true_b.T)

    # Set up the mse loss to match the hessian API
    def loss(params, unused_batch):
      y_pred = jnp.dot(params['AA']['inner'], params['AB']['inner'].T)
      return jnp.sum((y_pred - y) ** 2) / 2

    # Fake batches_gen to match the hessian_eval_api.
    def batches_gen():
      yield flax.jax_utils.replicate(jnp.array(1))  # Match expected API.

    # Set up curvature evaluator
    eval_config = hessian_eval.DEFAULT_EVAL_CONFIG.copy()
    eval_config['block_hessian'] = True
    eval_config['param_partition_fn'] = 'outer_key'
    evaluator = hessian_eval.CurvatureEvaluator(
        rep_params,
        eval_config,
        batches_gen=batches_gen,
        loss=loss)

    results, _, _ = evaluator.evaluate_spectrum(rep_params, step=0)
    a_max_eig = np.linalg.eigvalsh(
        np.dot(params['AB']['inner'], params['AB']['inner'].T)).max()
    b_max_eig = np.linalg.eigvalsh(
        np.dot(params['AA']['inner'], params['AA']['inner'].T)).max()

    self.assertAlmostEqual(
        a_max_eig, results['block_hessian']['AA']['max_eig_hess'], places=5)

    # True value is bigger than 1000, so need less places here.
    self.assertAlmostEqual(
        b_max_eig, results['block_hessian']['AB']['max_eig_hess'], places=2)


if __name__ == '__main__':
  absltest.main()
