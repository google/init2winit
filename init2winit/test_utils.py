# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Tests for utils.py.

"""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from init2winit import checkpoint
from init2winit import utils
import jax.numpy as jnp
import numpy as np


def _identity(i):
  return i


def _fn_that_always_fails(arg):
  del arg
  raise ValueError('I always fail.')


class UtilsTest(parameterized.TestCase):
  """Tests for utils.py."""

  def setUp(self):
    super(UtilsTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(UtilsTest, self).tearDown()

  @parameterized.named_parameters(
      dict(
          testcase_name='empty list of args',
          num_workers=1,
          input_list_dict=[],
          expected=[],
      ),
      dict(
          testcase_name='one worker, nonempty list',
          num_workers=1,
          input_list_dict=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
      dict(
          testcase_name='fewer workers than jobs, nonempty list',
          num_workers=3,
          input_list_dict=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
      dict(
          testcase_name='more workers than jobs, nonempty list',
          num_workers=20,
          input_list_dict=[dict(i=k) for k in range(1, 10)],
          expected=list(range(1, 10)),
      ),
  )
  def testRunInParallel(self, input_list_dict, num_workers, expected):
    """Test running successful fns in parallel, originally from mlbileschi."""
    actual = utils.run_in_parallel(_identity, input_list_dict, num_workers)
    self.assertEqual(actual, expected)

  def testRunInParallelOnFailingFn(self):
    """Test running failing fns in parallel, originally from mlbileschi."""
    with self.assertRaisesRegex(ValueError, 'I always fail.'):
      utils.run_in_parallel(_fn_that_always_fails, [dict(arg='hi')], 10)

  def testAppendPytree(self):
    """Test appending and loading pytrees."""
    pytrees = [{'a': i} for i in range(10)]
    pytree_path = os.path.join(self.test_dir, 'pytree.ckpt')
    logger = utils.MetricLogger(pytree_path=pytree_path)

    for pytree in pytrees:
      logger.append_pytree(pytree)

    latest = checkpoint.load_latest_checkpoint(pytree_path, prefix='')
    saved_pytrees = latest['pytree'] if latest else []
    self.assertEqual(
        pytrees, [saved_pytrees[str(i)] for i in range(len(saved_pytrees))]
    )

  def testArrayAppend(self):
    """Test appending to an array."""
    np.testing.assert_allclose(
        utils.array_append(jnp.array([1, 2, 3]), 4), jnp.array([1, 2, 3, 4])
    )
    np.testing.assert_allclose(
        utils.array_append(jnp.array([[1, 2], [3, 4]]), jnp.array([5, 6])),
        jnp.array([[1, 2], [3, 4], [5, 6]]),
    )

  def testTreeNormSqL2(self):
    """Test computing the squared L2 norm of a pytree."""
    pytree = {'foo': jnp.ones(10), 'baz': jnp.ones(20)}
    self.assertEqual(utils.tree_norm_sql2(pytree), {'foo': 10.0, 'baz': 20.0})
    self.assertEqual(utils.total_tree_norm_sql2(pytree), 30.0)

  def testTreeSum(self):
    """Test computing the sum of a pytree."""
    pytree = {'foo': 2 * jnp.ones(10), 'baz': jnp.ones(20)}
    self.assertEqual(utils.total_tree_sum(pytree), 40)


if __name__ == '__main__':
  absltest.main()
