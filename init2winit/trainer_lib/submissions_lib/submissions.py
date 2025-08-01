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

"""Submissions library for the init2winit project."""

import collections
from init2winit.trainer_lib.submissions_lib import adamw_jax_paper_baseline

_Submission = collections.namedtuple(
    'Submission', (
        'map_i2w_hparams_to_algoperf_hparams',
        'init_optimizer_state'))


def get_submission_module(submission_name: str) -> _Submission:
  """Returns the submission module for the given submission name."""
  if submission_name == 'adamw_jax_paper_baseline':
    return _Submission(
        adamw_jax_paper_baseline.map_i2w_hparams_to_algoperf_hparams,
        adamw_jax_paper_baseline.init_optimizer_state,
    )
  else:
    raise ValueError(f'Unknown submission name: {submission_name}')
