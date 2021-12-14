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

"""Tests for checkpoint.py."""

import copy
import functools
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from init2winit import checkpoint
from init2winit.model_lib import models
import jax.numpy as jnp
import jax.tree_util
import numpy as np

FLAGS = flags.FLAGS


def pytree_equal(tree1, tree2):
  try:
    equal_tree = jax.tree_util.tree_multimap(np.array_equal, tree1, tree2)
    return jax.tree_util.tree_reduce(lambda x, y: x and y, equal_tree)
  # The tree_utils will raise TypeErrors if structures don't match.
  except TypeError:
    return False

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
  @parameterized.named_parameters(('deprecated', True), ('flax', False))
  def test_save_load_roundtrip(self, use_deprecated_checkpointing):
    """Test that saving and loading produces the original state."""
    baz = ['a', 'b', 'ccc']
    state = checkpoint.CheckpointState(self.params,
                                       global_step=5, completed_epochs=4,
                                       baz=baz)
    checkpoint.save_checkpoint(
        self.test_dir,
        'checkpoint',
        state,
        use_deprecated_checkpointing=use_deprecated_checkpointing)
    latest = checkpoint.load_latest_checkpoint(
        self.test_dir,
        target=state,
        use_deprecated_checkpointing=use_deprecated_checkpointing)

    self.assertEqual(latest.pystate['baz'], baz)
    assert pytree_equal(latest.pytree, self.params)
    self.assertEqual(latest.pystate['global_step'], 5)
    self.assertEqual(latest.pystate['completed_epochs'], 4)

  def test_save_checkpoint_background_reraises_error(self):
    """Test than an error while saving a checkpoint is re-raised later."""
    # Checkpoint error is not raised when it actually happens, but when we next
    # write a checkpoint.
    baz = ['a', 'b', 'ccc']
    state = checkpoint.CheckpointState(self.params,
                                       global_step=5, completed_epochs=4,
                                       baz=baz)
    checkpoint.save_checkpoint_background(
        '/forbidden_directory/', 'checkpoint', state)
    with self.assertRaisesRegex(BaseException, r'Permission\sdenied'):
      checkpoint.save_checkpoint_background(self.test_dir, 'checkpoint', state)


if __name__ == '__main__':
  absltest.main()
