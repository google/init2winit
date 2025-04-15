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

r"""Configuration for tpl schedule with random search for smoke test.

gxm third_party/py/init2winit/projects/optlrschedule/xm_launch.py \
--xm_resource_alloc=group:deepmind-dynamic/lather-dynamic \
--xm_resource_pool=deepmind-dynamic \
--config=third_party/py/init2winit/projects/optlrschedule/testdata/configs/run_search_decoupled_cifar10_smoketest.py \
--xm_skip_launch_confirmation
"""

import datetime
import ml_collections
import numpy as np


def get_config():
  """Test config for cosine schedule with grid search."""
  config = ml_collections.ConfigDict()
  config.binary_name = 'run_search_decoupled'

  # Type of TPU
  config.platform = 'df=4x4'
  config.num_workers = 2

  # General training configuration
  num_base_lr = 3
  config.total_steps = 26
  config.base_lr_list = np.logspace(
      np.log10(0.001), np.log10(0.1), num_base_lr
  ).tolist()

  # Schedule configuration
  config.schedule_family_config = ml_collections.ConfigDict()
  config.schedule_family_config.schedule_type = 'twopointsspline'
  config.schedule_family_config.y_min = 0.0
  config.schedule_family_config.y_max = 1.0
  config.schedule_family_config.total_steps = config.total_steps
  config.schedule_family_config.warmup_type = 'linear'
  config.schedule_family_config.is_monotonic_decay = True

  # Schedule parameter ranges
  config.schedule_param_range = ml_collections.ConfigDict()
  config.schedule_param_range.x0 = (0.01, 0.25)
  config.schedule_param_range.y1 = (0.1, 1.0)
  config.schedule_param_range.delta_x1 = (0.0, 1.0)
  config.schedule_param_range.delta_x2 = (0.0, 1.0)
  config.schedule_param_range.delta_y2 = (0.0, 1.0)

  # Search configuration
  config.search_config = ml_collections.ConfigDict()
  config.search_config.type = 'random'
  config.search_config.scoring_metric = 'best/train_errors'
  # Product of these numbers with num_base_lr is the total number of
  # schedules to run per worker. Currently: 42.
  config.search_config.num_generation = 1
  config.search_config.num_schedule_shapes_per_gen = 2
  config.search_config.num_param_rngs = 7
  # Maximum schedules to run in parallel. Currently corresponds to: 9 chunks
  config.search_config.num_parallel_schedules = 5
  config.search_config.seed = 0

  # Workload configuration
  config.workload_config = ml_collections.ConfigDict()
  config.workload_config.workload_name = 'cifar10_cnn'
  config.workload_config.optimizer = 'adam'  # choice: adam, sgd
  config.workload_config.rng_seed = 0
  config.workload_config.batch_size = 32  # Smaller batch size for smoke test
  config.workload_config.total_steps = config.total_steps
  config.workload_config.use_dummy_data = False
  config.workload_config.compute_option = 'jit(vmap)'

  # Evaluation configuration
  config.workload_config.eval_config = ml_collections.ConfigDict()
  config.workload_config.eval_config.eval_mode = 'step'
  config.workload_config.eval_config.eval_frequency = config.total_steps - 1

  # Experiment name
  config.xm_experiment_name = f'smoke_test_{config.workload_config.workload_name}_{config.platform}x{config.num_workers}'
  timestamp = datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
  config.xm_experiment_name += timestamp
  return config
