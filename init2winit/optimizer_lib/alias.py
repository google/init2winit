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

"""Aliases for popular optimisers."""
from typing import Any, Callable, Optional, Union
from init2winit.optimizer_lib import transform
import optax


def nadamw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this)

  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    weight_decay: strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: a tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Nadam gradient transformations are applied to all parameters.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      transform.scale_by_nadam(b1, b2, eps, eps_root, debias),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate))
