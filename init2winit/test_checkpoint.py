# coding=utf-8
# Copyright 2023 The init2winit Authors.
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
from flax import jax_utils
from init2winit import checkpoint
from init2winit.model_lib import models
from init2winit.shared_test_utilities import pytree_equal
import jax.numpy as jnp
import jax.tree_util
import numpy as np
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
    baz = ['a', 'b', 'ccc']
    state = dict(params=self.params,
                 global_step=5,
                 completed_epochs=4,
                 baz=baz)
    checkpoint.save_checkpoint(
        self.test_dir,
        0,
        state)
    latest = checkpoint.load_latest_checkpoint(self.test_dir, target=state)

    self.assertEqual(latest['baz'], baz)
    assert pytree_equal(latest['params'], self.params)
    self.assertEqual(latest['global_step'], 5)
    self.assertEqual(latest['completed_epochs'], 4)

  def test_delete_old_checkpoints(self):
    """Test that old checkpoints are deleted."""
    state1 = dict(params=self.params,
                  global_step=5,
                  completed_epochs=4)
    checkpoint.save_checkpoint(
        self.test_dir,
        0,
        state1,
        max_to_keep=1)

    state2 = dict(params=self.params,
                  global_step=10,
                  completed_epochs=8)
    checkpoint.save_checkpoint(
        self.test_dir,
        1,
        state2,
        max_to_keep=1)
    dir_contents = gfile.glob(os.path.join(self.test_dir, '*'))
    self.assertLen(dir_contents, 1)

  def test_save_checkpoint_background_reraises_error(self):
    """Test than an error while saving a checkpoint is re-raised later."""
    # Checkpoint error is not raised when it actually happens, but when we next
    # write a checkpoint.
    baz = ['a', 'b', 'ccc']
    state = dict(params=self.params,
                 global_step=5, completed_epochs=4,
                 baz=baz)
    checkpoint.save_checkpoint_background(
        '/forbidden_directory/', 0, state)
    with self.assertRaisesRegex(BaseException, r'Permission\sdenied'):
      checkpoint.save_checkpoint_background(
          self.test_dir, 0, state)

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

    checkpoint.save_checkpoint(
        train_dir=fresh_train_dir,
        step=global_step,
        state=dict(global_step=global_step,
                   preemption_count=preemption_count,
                   sum_train_cost=sum_train_cost,
                   optimizer_state=saved_optimizer_state,
                   params=saved_params,
                   batch_stats=saved_batch_stats,
                   training_metrics_grabber=saved_training_metrics),
        max_to_keep=1)

    (ret_state, ret_params, ret_batch_stats, ret_training_metrics,
     ret_global_step, ret_sum_train_cost, ret_preemption_count, ret_is_restored,
     ) = checkpoint.replicate_and_maybe_restore_checkpoint(
         initial_optimizer_state, initial_params, initial_batch_stats,
         initial_training_metrics, fresh_train_dir)

    assert pytree_equal(
        jax.device_get(jax_utils.unreplicate(ret_state)),
        saved_optimizer_state)
    assert pytree_equal(
        jax.device_get(jax_utils.unreplicate(ret_params)),
        saved_params)
    assert pytree_equal(
        jax.device_get(jax_utils.unreplicate(ret_batch_stats)),
        saved_batch_stats)
    assert pytree_equal(
        jax.device_get(jax_utils.unreplicate(ret_training_metrics)),
        saved_training_metrics)
    self.assertEqual(ret_sum_train_cost, sum_train_cost)
    self.assertEqual(ret_preemption_count, preemption_count)
    self.assertEqual(ret_global_step, global_step)
    self.assertEqual(ret_is_restored, True)

    shutil.rmtree(fresh_train_dir)

  def test_replicate_and_maybe_restore_from_checkpoint_logic(self):
    """Test that the right checkpoint is returned.

      1.  If no external_checkpoint_path was passed, and if there is no
      latest checkpoint in the train_dir, then the function should return
      the passed-in params, batch_stats, etc.
      2.  If an external checkpoint was provided but no latest checkpoint
      exists in the train_dir, then the function should return the external
      checkpoint.
      3.  If a latest checkpoint exists in the train dir, then the function
      should return that checkpoint.

      In the interest of conciseness, this test only checks the params,
      not the batch_stats, optimizer_state, or training_metics.  The below test
      test_all_variables_restored() covers the other three.
    """
    # mock parameters.
    initial_params = {'foo': 1.0}
    latest_params = {'foo': 2.0}
    external_params = {'foo': 3.0}

    fresh_train_dir = tempfile.mkdtemp()
    external_dir = tempfile.mkdtemp()

    # two helper functions
    def save_checkpoint(train_dir, global_step, preemption_count,
                        sum_train_cost, params):
      """Helper function to save a checkpoint."""

      checkpoint.save_checkpoint(
          train_dir=train_dir,
          step=global_step,
          state=dict(global_step=global_step,
                     preemption_count=preemption_count,
                     sum_train_cost=sum_train_cost,
                     optimizer_state={},
                     params=params,
                     batch_stats={},
                     training_metrics_grabber={}),
          max_to_keep=1)

    def maybe_restore_checkpoint(params, train_dir, external_checkpoint_path):
      """Helper function to replicate_and_maybe_restore a checkpoint."""

      (_, ret_params, _, _,
       ret_global_step, ret_sum_train_cost, ret_preemption_count,
       ret_is_restored) = checkpoint.replicate_and_maybe_restore_checkpoint(
           {}, params, {}, {}, train_dir, external_checkpoint_path)

      ret_params_unrep = jax.device_get(jax_utils.unreplicate(ret_params))

      return (ret_params_unrep, ret_global_step, ret_sum_train_cost,
              ret_preemption_count, ret_is_restored)

    # Save external checkpoint.
    save_checkpoint(train_dir=external_dir,
                    global_step=5,
                    preemption_count=4,
                    sum_train_cost=7.0,
                    params=external_params)
    external_checkpoint_path = os.path.join(external_dir, 'ckpt_' + str(5))

    # If no latest checkpoint exists, and no external checkpoint was provided,
    # the function should return the passed-in params.

    (ret_params, ret_global_step, ret_sum_train_cost, ret_preemption_count,
     ret_is_restored) = maybe_restore_checkpoint(initial_params,
                                                 fresh_train_dir,
                                                 None)

    self.assertEqual(ret_preemption_count, 0)
    self.assertEqual(ret_global_step, 0)
    self.assertEqual(ret_sum_train_cost, 0.0)
    self.assertFalse(ret_is_restored)
    assert pytree_equal(ret_params, initial_params)

    # If no latest checkpoint exists, and an external checkpoint was provided,
    # the function should return the external checkpoint.

    (ret_params, ret_global_step, ret_sum_train_cost, ret_preemption_count,
     ret_is_restored) = maybe_restore_checkpoint(initial_params,
                                                 fresh_train_dir,
                                                 external_checkpoint_path)

    self.assertEqual(ret_preemption_count, 4)
    self.assertEqual(ret_global_step, 5)
    self.assertEqual(ret_sum_train_cost, 7.0)
    self.assertFalse(ret_is_restored)
    assert pytree_equal(ret_params, external_params)

    # Save latest checkpoint.
    save_checkpoint(train_dir=fresh_train_dir,
                    global_step=10,
                    preemption_count=2,
                    sum_train_cost=2.2,
                    params=latest_params)

    # If a latest checkpoint exists, then even if an external checkpoint was
    # provided, the function should return the latest checkpoint.

    (ret_params, ret_global_step, ret_sum_train_cost, ret_preemption_count,
     ret_is_restored) = maybe_restore_checkpoint(initial_params,
                                                 fresh_train_dir,
                                                 external_checkpoint_path)

    self.assertEqual(ret_preemption_count, 2)
    self.assertEqual(ret_global_step, 10)
    self.assertEqual(ret_sum_train_cost, 2.2)
    self.assertTrue(ret_is_restored)
    assert pytree_equal(ret_params, latest_params)

    shutil.rmtree(fresh_train_dir)
    shutil.rmtree(external_dir)


if __name__ == '__main__':
  absltest.main()
