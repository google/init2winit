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

"""Tests for checkpoint.py."""

import copy
import functools
import os.path
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from init2winit import checkpoint
from init2winit.model_lib import models
from init2winit.shared_test_utilities import pytree_equal
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import orbax.checkpoint as ocp
from tensorflow.io import gfile


FLAGS = flags.FLAGS

INPUT_SHAPE = [10, 28, 28, 1]
OUTPUT_SHAPE = (10,)


class CheckpointTest(parameterized.TestCase):
  """Tests for checkpoint.py mostly by doing round trips."""

  def setUp(self):
    super(CheckpointTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    model = models.get_model('fully_connected')
    model_hps = models.get_model_hparams('fully_connected')
    hps = copy.copy(model_hps)
    hps.update({'output_shape': OUTPUT_SHAPE})
    rng = jax.random.PRNGKey(0)
    model = model(hps, {}, loss_name, metrics_name)
    xs = jnp.array(np.random.normal(size=INPUT_SHAPE))
    rng, params_rng = jax.random.split(rng)
    model_init_fn = jax.jit(
        functools.partial(model.flax_module.init, train=False))
    init_dict = model_init_fn({'params': params_rng}, xs)
    self.params = init_dict['params']

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(CheckpointTest, self).tearDown()

  # TODO(gdahl): should we test that the accumulators get restored properly?
  # We could supply the params pytree as a fake gradient and do an update.
  def test_save_load_roundtrip(self):
    """Test that saving and loading produces the original state."""
    orbax_checkpoint_manager = ocp.CheckpointManager(
        self.test_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True),
    )
    state = dict(params=self.params, global_step=5, completed_epochs=4)
    checkpoint.save_checkpoint(
        0,
        state,
        orbax_checkpoint_manager=orbax_checkpoint_manager,
    )
    orbax_checkpoint_manager.wait_until_finished()
    latest = checkpoint.load_latest_checkpoint(
        target=state, orbax_checkpoint_manager=orbax_checkpoint_manager
    )

    assert pytree_equal(latest['params'], self.params)
    self.assertEqual(latest['global_step'], 5)
    self.assertEqual(latest['completed_epochs'], 4)

  def test_delete_old_checkpoints(self):
    """Test that old checkpoints are deleted."""
    orbax_checkpoint_manager = ocp.CheckpointManager(
        self.test_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True
        ),
    )
    state1 = dict(params=self.params,
                  global_step=5,
                  completed_epochs=4,)
    checkpoint.save_checkpoint(
        0,
        state1,
        orbax_checkpoint_manager=orbax_checkpoint_manager)

    state2 = dict(params=self.params,
                  global_step=10,
                  completed_epochs=8)
    checkpoint.save_checkpoint(
        1,
        state2,
        orbax_checkpoint_manager=orbax_checkpoint_manager)
    orbax_checkpoint_manager.wait_until_finished()
    dir_contents = gfile.glob(os.path.join(self.test_dir, '*'))

    self.assertLen(dir_contents, 1)

  def test_all_variables_restored(self):
    """Test that all variables are properly restored.

    This test checks that optimizer_state, params, batch_stats, and
    training_metrics_grabber are all properly restored after training
    is pre-empted.
    """

    fresh_train_dir = tempfile.mkdtemp()
    global_step = 100
    preemption_count = 8
    sum_train_cost = 0.9

    saved_optimizer_state = {'second_moments': 7}
    saved_params = {'kernel': 3}
    saved_batch_stats = {'mean': 2}
    saved_training_metrics = {'ema': 4}

    initial_optimizer_state = {'second_moments': 0}
    initial_params = {'kernel': 0}
    initial_batch_stats = {'mean': 0}
    initial_training_metrics = {'ema': 0}

    orbax_checkpoint_manager = ocp.CheckpointManager(
        fresh_train_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True
        ),
    )

    checkpoint.save_checkpoint(
        step=global_step,
        state=dict(global_step=global_step,
                   preemption_count=preemption_count,
                   sum_train_cost=sum_train_cost,
                   optimizer_state=saved_optimizer_state,
                   params=saved_params,
                   batch_stats=saved_batch_stats,
                   training_metrics_grabber=saved_training_metrics),
        orbax_checkpoint_manager=orbax_checkpoint_manager,)

    (
        ret_state,
        ret_params,
        ret_batch_stats,
        ret_training_metrics,
        ret_global_step,
        ret_sum_train_cost,
        ret_preemption_count,
        ret_is_restored,
    ) = checkpoint.maybe_restore_checkpoint(
        initial_optimizer_state,
        initial_params,
        initial_batch_stats,
        initial_training_metrics,
        orbax_checkpoint_manager=orbax_checkpoint_manager,
    )

    assert pytree_equal(
        ret_state, saved_optimizer_state
    )
    assert pytree_equal(
        ret_params, saved_params
    )
    assert pytree_equal(
        ret_batch_stats,
        saved_batch_stats,
    )
    assert pytree_equal(
        ret_training_metrics,
        saved_training_metrics,
    )
    self.assertEqual(ret_sum_train_cost, sum_train_cost)
    self.assertEqual(ret_preemption_count, preemption_count)
    self.assertEqual(ret_global_step, global_step)
    self.assertEqual(ret_is_restored, True)

    shutil.rmtree(fresh_train_dir)

  def test_maybe_restore_from_checkpoint_logic(self):
    """Test that the right checkpoint is returned.

      1.  If there is no latest checkpoint in the train_dir, then the function 
      should returnthe passed-in params, batch_stats, etc.
      2.  If there is a latest checkpoint in the train_dir, then the function
      should return the latest checkpoint.
      In the interest of conciseness, this test only checks the params,
      not the batch_stats, optimizer_state, or training_metics.  The below test
      test_all_variables_restored() covers the other three.
    """
    # mock parameters.
    initial_params = {'foo': 1.0}
    latest_params = {'foo': 3.0}

    checkpoint_dir = tempfile.mkdtemp()

    orbax_checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True
        ),
    )

    # two helper functions
    def save_checkpoint(
        orbax_checkpoint_manager,
        global_step,
        preemption_count,
        sum_train_cost,
        params,
    ):
      """Helper function to save a checkpoint."""

      checkpoint.save_checkpoint(
          step=global_step,
          state=dict(
              global_step=global_step,
              preemption_count=preemption_count,
              sum_train_cost=sum_train_cost,
              optimizer_state={},
              params=params,
              batch_stats={},
              training_metrics_grabber={},
          ),
          orbax_checkpoint_manager=orbax_checkpoint_manager,
      )

    def maybe_restore_checkpoint(orbax_checkpoint_manager, params):
      """Helper function to replicate_and_maybe_restore a checkpoint."""

      (
          _,
          ret_params,
          _,
          _,
          ret_global_step,
          ret_sum_train_cost,
          ret_preemption_count,
          ret_is_restored,
      ) = checkpoint.maybe_restore_checkpoint(
          {}, params, {}, {}, orbax_checkpoint_manager=orbax_checkpoint_manager
      )

      ret_params_unrep = ret_params

      return (
          ret_params_unrep,
          ret_global_step,
          ret_sum_train_cost,
          ret_preemption_count,
          ret_is_restored,
      )

    # If no latest checkpoint exists, the function should return the passed-in
    # params.

    (
        ret_params,
        ret_global_step,
        ret_sum_train_cost,
        ret_preemption_count,
        ret_is_restored,
    ) = maybe_restore_checkpoint(orbax_checkpoint_manager, initial_params)

    self.assertEqual(ret_preemption_count, 0)
    self.assertEqual(ret_global_step, 0)
    self.assertEqual(ret_sum_train_cost, 0.0)
    self.assertFalse(ret_is_restored)
    assert pytree_equal(ret_params, initial_params)

    # If no latest checkpoint exists, and an external checkpoint was provided,
    # the function should return the external checkpoint.

    # Save external checkpoint.
    save_checkpoint(
        orbax_checkpoint_manager,
        global_step=5,
        preemption_count=4,
        sum_train_cost=7.0,
        params=latest_params,
    )

    orbax_checkpoint_manager.wait_until_finished()

    (
        ret_params,
        ret_global_step,
        ret_sum_train_cost,
        ret_preemption_count,
        ret_is_restored,
    ) = maybe_restore_checkpoint(orbax_checkpoint_manager, latest_params)

    self.assertEqual(ret_preemption_count, 4)
    self.assertEqual(ret_global_step, 5)
    self.assertEqual(ret_sum_train_cost, 7.0)
    self.assertTrue(ret_is_restored)
    assert pytree_equal(ret_params, latest_params)

    shutil.rmtree(checkpoint_dir)


if __name__ == '__main__':
  absltest.main()
