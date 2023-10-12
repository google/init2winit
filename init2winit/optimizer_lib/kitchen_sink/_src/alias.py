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

"""Aliases for optimizers not found in optax."""
from typing import Any, Callable, Optional, Union
from init2winit.optimizer_lib.kitchen_sink._src import transform
import jax.numpy as jnp
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


def adapropw(
    learning_rate: optax.ScalarOrSchedule,
    alpha: float = 1.0,
    b1: float = 0.9,
    b3: float = 1.0,
    b4: float = 0.999,
    eps: float = 1e-8,
    use_nesterov: bool = False,
    quantized_dtype: str = 'float32',
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the AdaProp algorithm.

  Args:
    learning_rate: this is a fixed global scaling factor.
    alpha: upper bound on bet.
    b1: decay rate for the exponentially weighted average of grads.
    b3: decay rate for the exponentially weighted average of max grads.
    b4: decay rate for the exponentially weighted average of reward.
    eps: term added to the denominator to improve numerical stability.
    use_nesterov: Whether to use Nesterov-style update.
    quantized_dtype: type of the quantized input. Allowed options are
      'bfloat16' and 'float32'. If floating-point type is specified,
      accumulators are stored as such type, instead of quantized integers.
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
  if quantized_dtype == 'float32':
    q_dtype = jnp.float32
  else:
    q_dtype = jnp.bfloat16
  return optax.chain(
      transform.scale_by_adaprop(alpha=alpha, b1=b1, b3=b3, b4=b4,
                                 eps=eps, use_nesterov=use_nesterov,
                                 quantized_dtype=q_dtype),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      transform.scale_by_learning_rate(learning_rate, flip_sign=True),
  )
