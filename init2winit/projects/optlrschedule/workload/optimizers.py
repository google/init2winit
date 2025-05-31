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

"""Optimizers useful for multiple workloads."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, Union
import optax


def adamw(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
    *,
    nesterov: bool = False,
    disable_multiply_wd_by_base_lr: bool = False,
) -> optax.GradientTransformationExtraArgs:
  r"""Adam with weight decay regularization.
  
  Differs from optax.adamw in that the coupling of weight decay and learning
  rate can be disabled. When weight decay is decoupled from the
  learning rate, the weight decay is applied to the parameters as
  ``\lambda \theta`` instead of ``\alpha \lambda  \theta``.

  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam,
  see [Loshchilov et al, 2019].

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectively. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function. Let :math:`\lambda` be the weight decay and
  :math:`\theta_t` the parameter vector at time :math:`t`.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, v_0) = (0, 0)`, representing initial estimates for the
  first and second moments. In practice these values are stored as pytrees
  containing all zeros, with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t`, the optimizer state :math:`S_t`
  and the parameters :math:`\theta_t` and computes updates :math:`u_t` and
  new state :math:`S_{t+1}`. Thus, for :math:`t > 0`, we have,

  .. math::

    if disable_multiply_wd_by_base_lr is False (default behavior):    

    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \left( \hat{m}_t / \left({\sqrt{\hat{v}_t
      + \bar{\varepsilon}} + \varepsilon} \right)  +\lambda \theta_{t}\right)\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}
    
    if disable_multiply_wd_by_base_lr is True:
    
    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t \\
      v_t &\leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{v}_t &\leftarrow v_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow -\alpha_t \cdot \left( \hat{m}_t / \left({\sqrt{\hat{v}_t
      + \bar{\varepsilon}} + \varepsilon} \right)  \right)-\lambda \theta_{t}\\
      S_t &\leftarrow (m_t, v_t).
    \end{align*}

  This implementation can incorporate a momentum a la Nesterov introduced by
  [Dozat 2016]. The resulting optimizer is then often referred as NAdamW.
  With the keyword argument `nesterov=True`, the optimizer uses Nesterov
  momentum, replacing the above :math:`\hat{m}_t` with

  .. math::
      \hat{m}_t \leftarrow
        \beta_1 m_t / {(1-\beta_1^{t+1})} + (1 - \beta_1) g_t / {(1-\beta_1^t)}.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: A small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      instance when computing (meta-)gradients through Adam.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.
    nesterov: Whether to use Nesterov momentum.
    disable_multiply_wd_by_base_lr: Whether to disable the multiplication of
      the weight decay by the base learning rate, see equations in docstring
      for precise definition.

  Returns:
    The corresponding :class:`optax.GradientTransformationExtraArgs`.


  References:
    Loshchilov et al, `Decoupled Weight Decay
    Regularization <https://arxiv.org/abs/1711.05101>`_, 2019

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_, 2016

  .. seealso::
    See the related functions :func:`optax.adamw`, :func:`optax.nadamw`.
  """
  if disable_multiply_wd_by_base_lr:
    return optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        optax.scale_by_learning_rate(learning_rate, flip_sign=False),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(1.0, flip_sign=True),
    )
  else:
    return optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale_by_learning_rate(
            learning_rate,
        ),
    )


def get_optimizer_from_config(
    config: Mapping[Any, Any],
) -> optax.GradientTransformationExtraArgs:
  """Get optimizer from config.

  Given a config, returns the corresponding optimizer using
  optax.inject_hyperparams, so the optimizer takes in the learning rate
  schedule during training.

  Args:
    config: The overall config.

  Returns:
    The optimizer, with all hyperparameters except learning rate injected.
  """
  optimizer_name = config['optimizer']
  weight_decay = config['optimizer_config'].get('weight_decay', 0)
  if weight_decay > 0 and optimizer_name != 'adamw':
    raise ValueError(
        'Weight decay is only supported for AdamW optimizer. Set adamw as the'
        ' optimizer instead.'
    )
  if optimizer_name.lower() == 'adam':
    beta_1 = config['optimizer_config']['beta1']
    beta_2 = config['optimizer_config']['beta2']
    return optax.inject_hyperparams(optax.adam)(
        learning_rate=0.0, b1=beta_1, b2=beta_2
    )
  elif optimizer_name.lower() == 'adamw':
    beta_1 = config['optimizer_config']['beta1']
    beta_2 = config['optimizer_config']['beta2']
    disable_multiply_wd_by_base_lr = config['optimizer_config'][
        'disable_multiply_wd_by_base_lr'
    ]
    return optax.inject_hyperparams(adamw)(
        learning_rate=0.0,
        b1=beta_1,
        b2=beta_2,
        weight_decay=weight_decay,
        disable_multiply_wd_by_base_lr=disable_multiply_wd_by_base_lr,
    )
  elif optimizer_name.lower() == 'sgd':
    return optax.inject_hyperparams(optax.sgd)(learning_rate=0.0)
  elif optimizer_name.lower() == 'momentumsgd':
    return optax.inject_hyperparams(optax.sgd)(learning_rate=0.0, momentum=0.9)
  else:
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')
