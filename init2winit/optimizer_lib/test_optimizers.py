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

"""Tests for optimizers."""
# import os
import shutil
import tempfile

from absl.testing import absltest
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib import utils as optimizers_utils
from ml_collections import config_dict
# import pandas
# import tensorflow.compat.v1 as tf

# TODO(b/385225663): add test for nadamw.


class OptimizersTrainerTest(absltest.TestCase):
  """Tests for optimizers.py that require starting a trainer object."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    self.trainer.wait_until_orbax_checkpointer_finished()
    shutil.rmtree(self.test_dir)
    super().tearDown()



if __name__ == '__main__':
  absltest.main()
