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

"""Kitchen Sink: decomposing optimizers in JAX."""

from init2winit.optimizer_lib.kitchen_sink._src.alias import nadamw
from init2winit.optimizer_lib.kitchen_sink._src.core import kitchen_sink
from init2winit.optimizer_lib.kitchen_sink._src.transform import add_decayed_weights
from init2winit.optimizer_lib.kitchen_sink._src.transform import bias_correction
from init2winit.optimizer_lib.kitchen_sink._src.transform import BiasCorrectionState
from init2winit.optimizer_lib.kitchen_sink._src.transform import clip_updates
from init2winit.optimizer_lib.kitchen_sink._src.transform import first_moment_ema
from init2winit.optimizer_lib.kitchen_sink._src.transform import nesterov
from init2winit.optimizer_lib.kitchen_sink._src.transform import nesterovpp
from init2winit.optimizer_lib.kitchen_sink._src.transform import polyak_averaging
from init2winit.optimizer_lib.kitchen_sink._src.transform import Polyak_AveragingState
from init2winit.optimizer_lib.kitchen_sink._src.transform import polyak_hb
from init2winit.optimizer_lib.kitchen_sink._src.transform import precondition_by_amsgrad
from init2winit.optimizer_lib.kitchen_sink._src.transform import precondition_by_layered_adaptive_rms
from init2winit.optimizer_lib.kitchen_sink._src.transform import precondition_by_rms
from init2winit.optimizer_lib.kitchen_sink._src.transform import precondition_by_rss
from init2winit.optimizer_lib.kitchen_sink._src.transform import precondition_by_yogi
from init2winit.optimizer_lib.kitchen_sink._src.transform import PreconditionByLayeredAdaptiveRMSState
from init2winit.optimizer_lib.kitchen_sink._src.transform import PreconditionByRssState
from init2winit.optimizer_lib.kitchen_sink._src.transform import PreconditionBySecondMomentCoordinateWiseState
from init2winit.optimizer_lib.kitchen_sink._src.transform import sanitize_values
from init2winit.optimizer_lib.kitchen_sink._src.transform import scale_by_adam
from init2winit.optimizer_lib.kitchen_sink._src.transform import scale_by_amsgrad
from init2winit.optimizer_lib.kitchen_sink._src.transform import scale_by_learning_rate
from init2winit.optimizer_lib.kitchen_sink._src.transform import scale_by_nadam
from init2winit.optimizer_lib.kitchen_sink._src.transform import ScaleByAdamState
from init2winit.optimizer_lib.kitchen_sink._src.transform import ScaleByAMSGradState
from init2winit.optimizer_lib.kitchen_sink._src.utils import unfreeze_wrapper


__version__ = '0.0.1'

__all__ = (
    'nadamw',
    'kitchen_sink',
    'bias_correction',
    'BiasCorrectionState',
    'clip_updates',
    'first_moment_ema',
    'nesterov',
    'nesterovpp',
    'add_decayed_weights',
    'polyak_averaging',
    'Polyak_AveragingState',
    'polyak_hb',
    'precondition_by_amsgrad',
    'precondition_by_layered_adaptive_rms',
    'precondition_by_rms',
    'precondition_by_rss',
    'precondition_by_yogi',
    'PreconditionByLayeredAdaptiveRMSState',
    'PreconditionByRssState',
    'PreconditionBySecondMomentCoordinateWiseState',
    'sanitize_values',
    'scale_by_adam',
    'scale_by_amsgrad',
    'scale_by_learning_rate',
    'scale_by_nadam',
    'ScaleByAdamState',
    'ScaleByAMSGradState',
    'unfreeze_wrapper',
)
