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

"""Transforms."""

import functools
from typing import Any, List, NamedTuple, Optional

import chex
from init2winit.optimizer_lib.kitchen_sink._src import utils
import jax
import jax.numpy as jnp
import optax


# pylint:disable=invalid-name
# pylint:disable=no-value-for-parameter


def _safe_int32_increment(count: chex.Numeric) -> chex.Numeric:
  """Increments int32 counter by one.

  Normally `max_int + 1` would overflow to `min_int`. This functions ensures
  that when `max_int` is reached the counter stays at `max_int`.

  Args:
    count: a counter to be incremented.

  Returns:
    A counter incremented by 1, or max_int if the maximum precision is reached.
  """
  chex.assert_type(count, jnp.int32)
  max_int32_value = jnp.iinfo(jnp.int32).max
  one = jnp.array(1, dtype=jnp.int32)
  return jnp.where(count < max_int32_value, count + one, max_int32_value)


def _canonicalize_dtype(dtype):
  """Canonicalise a dtype, skip if None."""
  if dtype is not None:
    return jax.dtypes.canonicalize_dtype(dtype)
  return dtype


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
  )


def _update_first_moment_variance_preserved(updates, moments, decay):
  """Applies variance preserved EMA.

  Multiplies incoming gradient by sqrt{1-beta^2} as opposed to 1-beta.
  Introduces bias.

  Args:
    updates: updates.
    moments: moments,
    decay: the decay parameter.

  Returns:
    Variance Preserved EMA.
  """
  return jax.tree_map(
      lambda g, t: ((1 - decay**2) ** 0.5) * g + decay * t,
      updates,
      moments,
  )


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


# TODO(namanagarwal): make this count generic
def _variance_correction(moment, decay, count):
  """Performs variance correction to make the first moment have unit variance.

  Specifically, the variance of the first moment at iteration t is
  ((1-decay)/(1+decay)*(1 - decay^{2*t})) times the variance of the input
  gradients (assuming all of them are iid)

  Args:
    moment: which gradient moment to use; typically this is 2.
    decay: the raw decay.
    count: current step, 0-based.

  Returns:
    Variance-normalized decay.
  """
  beta = (((1 - decay) / (1 + decay)) * (1 - decay ** (2 * count))) ** 0.5
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def _bias_corrected_decay(decay, count):
  """Incorporates bias correction into decay.

  Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
  derivation of the formulas below. With bias-corrected decay, we can simply
  do

  m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
  v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

  without further bias correction.

  Args:
    decay: the raw decay. As t -> infinity, bias corrected decay converges to
      this value.
    count: current step, 0-based.

  Returns:
    Bias corrected decay.
  """
  t = count.astype(jnp.float32) + 1.0
  return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def nesterov(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  return optax.trace(
      decay=decay, nesterov=True, accumulator_dtype=accumulator_dtype
  )


def polyak_hb(
    decay: float = 0.9,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  return optax.trace(
      decay=decay, nesterov=False, accumulator_dtype=accumulator_dtype
  )


def compute_params_ema_for_eval(
    decay: float, warmup: bool = False
) -> optax.GradientTransformation:
  """Applies exponential moving average on weights.

  Note, this implementation averages the weight before optimization because
  trainable and non-trainable variables are handled separately. In such case
  the updates on non-trainable variables like bn stats are not available in
  updates.

  This differs from optax.ema which applies ema on gradients so it changes
  training process.

  ema = ema * decay + new_weight * (1.0 - decay)

  Args:
    decay: A float number represents the weight on the moving average.
    warmup: bool controlling if we ignore initial training steps for EMA.

  Returns:
    A GradientTransformation applying ema.
  """

  def init_fn(params):
    return optax.EmaState(
        count=jnp.array(0, dtype=jnp.int32), ema=jax.tree_map(jnp.copy, params))

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError('Params required for the EMA')

    if warmup:
      # https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L469
      ema_decay = jnp.minimum(decay, (1. + state.count) / (10. + state.count))
    else:
      ema_decay = decay

    def update_func(old_v, new_v):
      if old_v.dtype == jnp.bool_ or jnp.issubdtype(old_v, jnp.integer):
        # If it is integer, we directly return the new variable
        # This is mainly supported for non_trainable
        return new_v
      else:
        return old_v - (1.0 - ema_decay) * (old_v - new_v)

    new_ema = jax.tree_map(update_func, state.ema, params)
    count_inc = state.count + jnp.array(1, jnp.int32)

    return updates, optax.EmaState(count=count_inc, ema=new_ema)

  return optax.GradientTransformation(init_fn, update_fn)


def first_moment_ema(
    decay: float = 0.9,
    debias: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  return optax.ema(
      decay=decay, debias=debias, accumulator_dtype=accumulator_dtype
  )


def normalized_first_moment_ema(
    decay: float = 0.9,
    debias: bool = False,
) -> optax.GradientTransformation:
  """Implements a scaled version of first moment ema.

  Args:
    decay: the decay rate used for the moment accumulation.
    debias: whether to use bias correction or not.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return optax.EmaState(
        count=jnp.zeros([], jnp.int32), ema=jax.tree_map(jnp.zeros_like, params)
    )

  def update_fn(updates, state, params=None):
    del params
    new_ema = _update_moment(updates, state.ema, decay, 1)

    count = state.count + jnp.array(1, dtype=jnp.int32)
    new_ema = new_ema if not debias else _bias_correction(new_ema, decay, count)

    scale_factor = (1 + decay) / (1 - decay) ** 0.5
    updates = jax.tree_map(lambda x: x * scale_factor, new_ema)

    return updates, optax.EmaState(count=count, ema=new_ema)

  return optax.GradientTransformation(init_fn, update_fn)


def nesterovpp(
    moment_decay: float,
    update_decay: float,
) -> optax.GradientTransformation:
  """Decouples the betas of the two Nesterov steps.

  Args:
    moment_decay: the decay rate used for the first moment.
    update_decay: the decay rate used for the update step.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return optax.TraceState(trace=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    f_moment = lambda g, t: g + moment_decay * t
    f_update = lambda g, t: g + update_decay * t
    new_trace = jax.tree_map(f_moment, updates, state.trace)
    updates = jax.tree_map(f_update, updates, new_trace)
    return updates, optax.TraceState(trace=new_trace)

  return optax.GradientTransformation(init_fn, update_fn)


def ema_nesterov(
    moment_decay: float,
    update_decay: float = None,
) -> optax.GradientTransformation:
  """Decouples the betas of the two Nesterov steps.

  Args:
    moment_decay: the decay rate used for the first moment and update step.
    update_decay: the decay rate used for the update step. If none, this is set
      equal to moment_decay

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return optax.TraceState(trace=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    f_moment = lambda g, t: (1 - moment_decay) * g + moment_decay * t
    if update_decay is not None:
      f_update = lambda g, t: (1 - update_decay) * g + update_decay * t
    else:
      f_update = lambda g, t: (1 - moment_decay) * g + moment_decay * t
    new_trace = jax.tree_map(f_moment, updates, state.trace)
    updates = jax.tree_map(f_update, updates, new_trace)
    return updates, optax.TraceState(trace=new_trace)

  return optax.GradientTransformation(init_fn, update_fn)


class PreconditionBySecondMomentCoordinateWiseState(NamedTuple):
  """State for the Adam preconditioner."""

  count: chex.Array
  nu: optax.Updates


def precondition_by_rms(
    decay: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = False,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Preconditions updates according to the RMS Preconditioner from Adam.

  References:
    [Kingma, Ba 2015] https://arxiv.org/pdf/1412.6980.pdf

  Args:
    decay: decay rate for exponentially weighted average of moments of grads.
    eps: Term added to the denominator to improve numerical stability. The
      default is kept to 1e-8 to match optax Adam implementation.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction or not
    power: the power to which the second moment is raised to

  Gotcha: Note that the usage of epsilon and defaults are different from optax's
    scale_by_rms. This matches optax's adam template.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    return PreconditionBySecondMomentCoordinateWiseState(
        count=jnp.zeros([], jnp.int32), nu=jax.tree_map(jnp.zeros_like, params)
    )

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, decay, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    nu_hat = nu if not debias else _bias_correction(nu, decay, count)
    updates = jax.tree_map(
        lambda u, v: u / (raise_power(v + eps_root) + eps), updates, nu_hat
    )
    return updates, PreconditionBySecondMomentCoordinateWiseState(
        count=count, nu=nu
    )

  return optax.GradientTransformation(init_fn, update_fn)


def precondition_by_yogi(
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6,
    debias: bool = True,
) -> optax.GradientTransformation:
  """Preconditions updates according to the Yogi Preconditioner.

  References:
    [Zaheer et al,
    2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)
    #pylint:disable=line-too-long

  Args:
    b2: decay rate for the exponentially weighted average of moments of grads.
    eps: Term added to the denominator to improve numerical stability. The
      default is changed to 1e-8. Optax Yogi's default is 1e-3.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    initial_accumulator_value: The starting value for accumulators. Only
      positive values are allowed.
    debias: whether to use bias correction or not

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    value_like = lambda p: jnp.full_like(p, initial_accumulator_value)
    nu = jax.tree_map(value_like, params)  # Second Central moment
    return PreconditionBySecondMomentCoordinateWiseState(
        count=jnp.zeros([], jnp.int32), nu=nu
    )

  def update_fn(updates, state, params=None):
    del params
    nu = jax.tree_map(
        lambda g, v: v - (1 - b2) * jnp.sign(v - g**2) * (g**2),
        updates,
        state.nu,
    )
    count = state.count + jnp.array(1, dtype=jnp.int32)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda u, v: u / (jnp.sqrt(v + eps_root) + eps), updates, nu_hat
    )
    return updates, PreconditionBySecondMomentCoordinateWiseState(
        count=count, nu=nu
    )

  return optax.GradientTransformation(init_fn, update_fn)


# TODO(namanagarwal): Testing needed
def precondition_by_amsgrad(
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> optax.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  References:
    [Reddi et al, 2018](https://arxiv.org/abs/1904.09237v1)

  Args:
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return PreconditionBySecondMomentCoordinateWiseState(
        count=jnp.zeros([], jnp.int32), nu=jax.tree_map(jnp.zeros_like, params)
    )

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    nu_hat = jax.tree_map(jnp.maximum, nu, state.nu)
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), updates, nu_hat
    )
    return updates, PreconditionBySecondMomentCoordinateWiseState(
        count=count, nu=nu
    )

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAMSGradState(NamedTuple):
  """State for the AMSGrad algorithm."""

  mu: optax.Updates
  nu: optax.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  References:
    [Reddi et al, 2018](https://arxiv.org/abs/1904.09237v1)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
    )
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAMSGradState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    nu_hat = jax.tree_map(jnp.maximum, nu, state.nu)
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu, nu_hat
    )
    return updates, ScaleByAMSGradState(mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


# TODO(namanagarwal) : Remove the following
class BiasCorrectionState(NamedTuple):
  """Holds an exponential moving average of past updates."""

  count: chex.Array  # shape=(), dtype=jnp.int32.


def bias_correction(
    decay: float = 0.9, accumulator_dtype: Optional[Any] = None
) -> optax.GradientTransformation:
  """Compute the Adam style bias correction.

  Args:
    decay: the decay rate for the exponential moving average.
    accumulator_dtype: optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  accumulator_dtype = _canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    del params
    return BiasCorrectionState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    count_inc = _safe_int32_increment(state.count)
    new_vals = _bias_correction(updates, decay, count_inc)
    return new_vals, BiasCorrectionState(count=count_inc)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleBy_Adaptive_GD_State(NamedTuple):
  """State for the adaptive GD algorithm."""

  r_squared: Any
  lambda_prev: Any
  lambda_sum: Any
  init_params: optax.Updates
  prev_update: optax.Updates


def scale_by_adaptive_gd(
    init_r_squared: float = 1.0,
) -> optax.GradientTransformation:
  """Rescale updates according to adaptive GD.

  Args:
    init_r_squared: initial guess for r^2.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    prev_update = jax.tree_map(
        jnp.zeros_like, params
    )  # previous update with step-size/lr included
    return ScaleBy_Adaptive_GD_State(
        r_squared=init_r_squared*jnp.ones([], jnp.float64),
        lambda_prev=jnp.zeros([], jnp.float64),
        lambda_sum=jnp.zeros([], jnp.float64),
        init_params=init_params,
        prev_update=prev_update,
    )

  def update_fn(updates, state, params):
    # we can use layer-wise distances later for a layer-wise variant
    layer_wise_curr_distance_squared = jax.tree_map(
        lambda x_t, x_0: jnp.sum((x_t - x_0) ** 2), params, state.init_params
    )
    curr_distance_norm_squared = utils.total_tree_sum(
        layer_wise_curr_distance_squared
    )
    # curr_r_squared plays the role of r_t^2 here
    curr_r_squared = jnp.maximum(state.r_squared, curr_distance_norm_squared)
    new_updates = jax.tree_map(
        lambda g, g_prev: g - state.lambda_prev * g_prev,
        updates,
        state.prev_update,
    )
    new_update_norm_squared = utils.total_tree_norm_sql2(new_updates)
    lambda_new = 0.5 * (
        jnp.sqrt(
            state.lambda_sum**2
            + jnp.divide(new_update_norm_squared, curr_r_squared)
        )
        - state.lambda_sum
    )
    lambda_sum_new = state.lambda_sum + lambda_new
    new_updates_with_lr = jax.tree_map(
        lambda u: u / lambda_sum_new, new_updates
    )
    negative_new_updates_with_lr = jax.tree_map(
        lambda u: -u, new_updates_with_lr
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_State(
        r_squared=curr_r_squared,
        lambda_prev=lambda_new,
        lambda_sum=lambda_sum_new,
        init_params=state.init_params,
        prev_update=negative_new_updates_with_lr,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_layerwise_adaptive_gd(
    init_r_squared: float = 1.0,
) -> optax.GradientTransformation:
  """Rescale updates according to LAYER-WISE Adaptive GD.

  Args:
    init_r_squared: initial guess for r^2.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    prev_update = jax.tree_map(
        jnp.zeros_like, params
    )  # previous update with step-size/lr included
    return ScaleBy_Adaptive_GD_State(
        r_squared=jax.tree_map(
            lambda x: init_r_squared * jnp.ones([], jnp.float64), params
        ),
        lambda_prev=jax.tree_map(lambda x: jnp.zeros([], jnp.float64), params),
        lambda_sum=jax.tree_map(lambda x: jnp.zeros([], jnp.float64), params),
        init_params=init_params,
        prev_update=prev_update,
    )

  def update_fn(updates, state, params):
    layer_wise_curr_distance_squared = jax.tree_map(
        lambda x_t, x_0: jnp.sum((x_t - x_0) ** 2), params, state.init_params
    )
    curr_distance_norm_squared = layer_wise_curr_distance_squared
    # curr_r_squared plays the role of r_t^2 here
    curr_r_squared = jax.tree_map(
        jnp.maximum,
        state.r_squared,
        curr_distance_norm_squared,
    )
    new_updates = jax.tree_map(
        lambda g, g_prev, l_prev: g - l_prev * g_prev,
        updates,
        state.prev_update,
        state.lambda_prev,
    )
    new_update_norm_squared = jax.tree_map(
        lambda u: jnp.sum(u ** 2), new_updates
    )
    lambda_new = jax.tree_map(
        lambda l, g, r: 0.5 * (jnp.sqrt(l**2 + jnp.divide(g, r)) - l),
        state.lambda_sum,
        new_update_norm_squared,
        curr_r_squared,
    )
    lambda_sum_new = jax.tree_map(
        lambda l1, l2: l1 + l2, state.lambda_sum, lambda_new
    )
    new_updates_with_lr = jax.tree_map(
        lambda u, l: u / l, new_updates, lambda_sum_new
    )
    negative_new_updates_with_lr = jax.tree_map(
        lambda u: -u, new_updates_with_lr
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_State(
        r_squared=curr_r_squared,
        lambda_prev=lambda_new,
        lambda_sum=lambda_sum_new,
        init_params=state.init_params,
        prev_update=negative_new_updates_with_lr,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_coordinate_wise_adaptive_gd(
    init_r_squared: float = 1.0,
) -> optax.GradientTransformation:
  """Rescale updates according to COORDINATE-WISE Adaptive GD.

  Args:
    init_r_squared: Initial guess for r^2.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    prev_update = jax.tree_map(
        jnp.zeros_like, params
    )  # previous update with step-size/lr included
    return ScaleBy_Adaptive_GD_State(
        r_squared=jax.tree_map(
            lambda x: init_r_squared * jnp.ones_like(x) / jnp.size(x),
            params,
        ),
        lambda_prev=jax.tree_map(jnp.zeros_like, params),
        lambda_sum=jax.tree_map(jnp.zeros_like, params),
        init_params=init_params,
        prev_update=prev_update,
    )

  def update_fn(updates, state, params):
    curr_distance_norm_squared = jax.tree_map(
        lambda x_t, x_0: jnp.square(x_t - x_0), params, state.init_params
    )
    curr_r_squared = jax.tree_map(
        jnp.maximum,
        state.r_squared,
        curr_distance_norm_squared,
    )
    new_updates = jax.tree_map(
        lambda g, g_prev, l_prev: g - jnp.multiply(l_prev, g_prev),
        updates,
        state.prev_update,
        state.lambda_prev,
    )
    new_update_norm_squared = jax.tree_map(
        jnp.square, new_updates
    )
    lambda_new = jax.tree_map(
        lambda l, g, r: 0.5 * (jnp.sqrt(jnp.square(l) + jnp.divide(g, r)) - l),
        state.lambda_sum,
        new_update_norm_squared,
        curr_r_squared,
    )
    lambda_sum_new = jax.tree_map(
        lambda l1, l2: l1 + l2, state.lambda_sum, lambda_new
    )
    new_updates_with_lr = jax.tree_map(
        jnp.divide, new_updates, lambda_sum_new
    )
    negative_new_updates_with_lr = jax.tree_map(
        lambda u: -u, new_updates_with_lr
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_State(
        r_squared=curr_r_squared,
        lambda_prev=lambda_new,
        lambda_sum=lambda_sum_new,
        init_params=state.init_params,
        prev_update=negative_new_updates_with_lr,
    )

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleBy_Adaptive_GD_Simple_State(NamedTuple):
  """State for the simpler adaptive GD algorithm."""

  r_squared: Any
  mu_sum: Any
  init_params: optax.Updates


def scale_by_adaptive_gd_simple(
    init_r_squared: float = 1.0,
) -> optax.GradientTransformation:
  """Rescale updates according to simpler adaptive GD.

  Args:
    init_r_squared: Initial guess for r^2.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    return ScaleBy_Adaptive_GD_Simple_State(
        r_squared=init_r_squared * jnp.ones([], jnp.float64),
        mu_sum=jnp.zeros([], jnp.float64),
        init_params=init_params,
    )

  def update_fn(updates, state, params):
    # we can use layer-wise distances later for a layer-wise variant
    layer_wise_curr_distance_squared = jax.tree_map(
        lambda x_t, x_0: jnp.sum((x_t - x_0) ** 2), params, state.init_params
    )
    curr_distance_norm_squared = utils.total_tree_sum(
        layer_wise_curr_distance_squared
    )
    curr_r_squared = jnp.maximum(state.r_squared, curr_distance_norm_squared)
    update_norm_squared = utils.total_tree_norm_sql2(updates)
    mu_sum_new = 0.5 * (
        jnp.sqrt(
            state.mu_sum**2
            + jnp.divide((4*update_norm_squared), curr_r_squared)
        )
        + state.mu_sum
    )
    new_updates_with_lr = jax.tree_map(
        lambda u: u / mu_sum_new, updates
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_Simple_State(
        r_squared=curr_r_squared,
        mu_sum=mu_sum_new,
        init_params=state.init_params,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_layerwise_adaptive_gd_simple(
    init_r_squared: float = 1.0,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
  """Rescale updates according to simpler LAYER-WISE Adaptive GD.

  Args:
    init_r_squared: Initial guess for r^2.
    eps: Initial value of mu_sum.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    return ScaleBy_Adaptive_GD_Simple_State(
        r_squared=jax.tree_map(
            lambda x: init_r_squared * jnp.ones([], jnp.float64), params
        ),
        mu_sum=jax.tree_map(lambda x: eps * jnp.ones([], jnp.float64), params),
        init_params=init_params,
    )

  def update_fn(updates, state, params):
    curr_distance_norm_squared = jax.tree_map(
        lambda x_t, x_0: jnp.sum((x_t - x_0) ** 2), params, state.init_params
    )
    curr_r_squared = jax.tree_map(
        jnp.maximum,
        state.r_squared,
        curr_distance_norm_squared,
    )
    update_norm_squared = jax.tree_map(
        lambda u: jnp.sum(u ** 2), updates
    )
    mu_sum_new = jax.tree_map(
        lambda l, g, r: 0.5 * (jnp.sqrt(l**2 + 4 * jnp.divide(g, r)) + l),
        state.mu_sum,
        update_norm_squared,
        curr_r_squared,
    )
    new_updates_with_lr = jax.tree_map(
        lambda u, l: u / l, updates, mu_sum_new
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_Simple_State(
        r_squared=curr_r_squared,
        mu_sum=mu_sum_new,
        init_params=state.init_params,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_coordinate_wise_adaptive_gd_simple(
    init_r_squared: float = 1.0,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
  """Rescale updates according to simpler COORDINATE-WISE Adaptive GD.

  Args:
    init_r_squared: Initial guess for r^2.
    eps: Initial value for mu_sum.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    init_params = jax.tree_map(jnp.copy, params)  # x0
    return ScaleBy_Adaptive_GD_Simple_State(
        r_squared=jax.tree_map(
            lambda x: init_r_squared*jnp.ones_like(x), params
        ),
        mu_sum=jax.tree_map(
            lambda x: eps*jnp.ones_like(x), params
        ),
        init_params=init_params,
    )

  def update_fn(updates, state, params):
    curr_distance_norm_squared = jax.tree_map(
        lambda x_t, x_0: jnp.square(x_t - x_0), params, state.init_params
    )
    curr_r_squared = jax.tree_map(
        jnp.maximum,
        state.r_squared,
        curr_distance_norm_squared,
    )
    update_norm_squared = jax.tree_map(
        jnp.square, updates
    )
    mu_sum_new = jax.tree_map(
        lambda l, g, r: 0.5*(jnp.sqrt(jnp.square(l) + 4*jnp.divide(g, r)) + l),
        state.mu_sum,
        update_norm_squared,
        curr_r_squared,
    )
    new_updates_with_lr = jax.tree_map(
        jnp.divide, updates, mu_sum_new
    )
    return new_updates_with_lr, ScaleBy_Adaptive_GD_Simple_State(
        r_squared=curr_r_squared,
        mu_sum=mu_sum_new,
        init_params=state.init_params,
    )

  return optax.GradientTransformation(init_fn, update_fn)


# TODO(namanagarwal): Add a test for Nadam
class ScaleByAdamState(NamedTuple):
  """State for the NAdam algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use moment bias correction.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_adam_plus(
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    rescale: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Convex Combination of preconditioned and raw un-preconditioned quantities.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    b3: convex comb. coeff. for mixing preconditioned grads with ema of grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    rescale: whether to do re-scaling of un-preconditioned update or not.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates_rms = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )

    mu_hat_2 = mu_hat
    # rescale un-preconditioned update if rescale is True
    if rescale:
      mu_hat_2 = jax.tree_map(lambda m: m / (1.0 - b1), mu_hat)
    updates = _update_moment(mu_hat_2, updates_rms, b3, 1)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_normalized_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm with Var. Normalization.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use moment bias correction.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)

    scale_factor = (1 + b1) / (1 - b1) ** 0.5

    updates = jax.tree_map(
        lambda m, v: scale_factor * (m / (raise_power(v + eps_root) + eps)),
        mu_hat,
        nu_hat,
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_normalized_adam_plus(
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm with Var. Normalization.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    b3: convex combination coefficient.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use moment bias correction.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)

    scale_factor = (1 + b1) / (1 - b1) ** 0.5

    updates_rms = jax.tree_map(
        lambda m, v: scale_factor * (m / (raise_power(v + eps_root) + eps)),
        mu_hat,
        nu_hat,
    )
    updates = _update_moment(mu_hat, updates_rms, b3, 1)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


# TODO(namanagarwal): merge this with the constant normalized adam version.
def scale_by_iteration_dependent_norm_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _variance_correction(mu, b1, count)
    nu_hat = _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_adam_var_preserved(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Edit first moments to preserve the variance across updates.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use moment bias correction. Note inspite of
            implementation Adam style bias correction might not make sense here.
            So it should not be used.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = _update_first_moment_variance_preserved(updates, state.mu, b1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    # see above comment about the debias.
    mu_hat = mu if not debias else _bias_correction(mu, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)

    updates = jax.tree_map(
        lambda m, v: (m / (raise_power(v + eps_root) + eps)),
        mu_hat,
        nu_hat,
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAdapropState(NamedTuple):
  """State for the AdaProp algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  pp: optax.Updates
  mu: optax.Updates
  nu: optax.Updates
  gain: optax.Updates


def scale_by_adaprop(
    alpha: float = 1.0,
    b1: float = 0.9,
    b3: float = 1.0,
    b4: float = 0.9,
    eps: float = 1e-8,
    use_nesterov: bool = False,
    quantized_dtype: jnp.dtype = jnp.float32,
) -> optax.GradientTransformation:
  """Rescale updates according to the AdaProp algorithm.

  Args:
    alpha: upper bound on bet.
    b1: decay rate for the exponentially weighted average of grads.
    # b2: decay rate for the exponentially weighted average of absolute grads
    #     is omitted because it is calculated from alpha and b1.
    b3: decay rate for the exponentially weighted average of max grads.
    b4: decay rate for the exponentially weighted average of reward.
    eps: term added to the denominator to improve numerical stability.
    use_nesterov: Whether to use Nesterov-style update.
    quantized_dtype: type of the quantized input. Allowed options are
      jnp.bfloat16 and jnp.float32. If floating-point type is specified,
      accumulators are stored as such type, instead of quantized integers.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    prev_params = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=quantized_dtype), params
    )
    mu = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=quantized_dtype), params
    )
    nu = jax.tree_map(
        lambda p: jnp.zeros_like(p, dtype=quantized_dtype), params
    )
    gain = jax.tree_map(
        lambda p: jnp.ones_like(p, dtype=quantized_dtype), params
    )

    return ScaleByAdapropState(
        count=jnp.zeros([], jnp.int32),
        pp=prev_params,
        mu=mu,
        nu=nu,
        gain=gain,
    )

  def update_fn(updates, state, params):
    new_count = optax.safe_int32_increment(state.count)
    b2 = 1.0 - (1.0 - b1)/alpha
    mu = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, state.mu)
    if use_nesterov:
      mu2 = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, mu)
      mu_hat = _bias_correction(mu2, b1, new_count)
    else:
      mu_hat = _bias_correction(mu, b1, new_count)
    nu = jax.tree_map(lambda g, t: (1-b2)*jnp.abs(g) + b2*t, updates, state.nu)
    nu_hat = _bias_correction(nu, b2, new_count)
    pp = jax.tree_map(lambda p, t: (1-b4)*p + b4*t, params, state.pp)
    pp_hat = _bias_correction(pp, b4, new_count)
    param_change = jax.tree_map(lambda p, i: p - i, params, pp_hat)
    g_max = jax.tree_map(lambda g, n: jnp.maximum(jnp.abs(g), n),
                         updates, nu_hat)
    gain = jax.tree_map(
        lambda r, p, g, x: jnp.maximum(b3*r - p*g/(x + eps), 0.0),
        state.gain, param_change, updates, g_max)
    wealth = jax.tree_map(lambda g: 1.0 + g, gain)

    bet_factor = jax.tree_map(
        lambda m, n: m / (n + eps),
        mu_hat,
        nu_hat,
    )
    new_updates = jax.tree_map(lambda b, w: b * w,
                               bet_factor, wealth)
    return new_updates, ScaleByAdapropState(
        count=new_count,
        pp=pp,
        mu=mu,
        nu=nu,
        gain=gain,
    )

  return optax.GradientTransformation(init_fn, update_fn)


class PreconditionByRssState(NamedTuple):
  """State holding the sum of gradient squares to date."""

  sum_of_squares: optax.Updates


def precondition_by_rss(
    initial_accumulator_value: float = 0.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Rescale updates by the powers of the sum of all squared gradients to date.

  Args:
    initial_accumulator_value: starting value for accumulators, must be >= 0.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the root to improve numerical
      stability when backpropagating gradients through the rescaling.
    power: the power to use when scaling (default is 0.5 to scale by root sum of
      squares).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    sum_of_squares = jax.tree_map(
        lambda t: jnp.full_like(t, initial_accumulator_value), params
    )
    return PreconditionByRssState(sum_of_squares=sum_of_squares)

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jax.tree_map(
        lambda g, t: jnp.square(g) + t, updates, state.sum_of_squares
    )
    updates = jax.tree_map(
        lambda scale, g: g / (raise_power(scale + eps_root) + eps),
        sum_of_squares,
        updates,
    )
    return updates, PreconditionByRssState(sum_of_squares=sum_of_squares)

  return optax.GradientTransformation(init_fn, update_fn)


def _sanitize_values(array, replacement=0.0):
  """Sanitizes NaN and Infinity values."""
  return jnp.nan_to_num(
      array, nan=replacement, posinf=replacement, neginf=replacement
  )


def sanitize_values(replacement=0.0):
  """Sanitizes updates by replacing NaNs and Infinity values with zeros.

  Args:
    replacement: value to replace NaNs and Infinity.

  Returns:
    An (init_fn, update_fn) tuple.0
  """

  def init_fn(params):
    del params
    return

  def update_fn(updates, state, params=None):
    del params

    updates = jax.tree_map(lambda x: _sanitize_values(x, replacement), updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def _reduce_mean(array):
  num_elements = array.size
  if num_elements > 1e8:
    # When x is too large, simple jnp.mean() can result in nan or inf values.
    array_sum = jnp.sum(array, axis=-1)
    array_sum = jnp.sum(array_sum)
    return array_sum / jnp.array(num_elements, dtype=array_sum.dtype)
  else:
    return jnp.mean(array)


def _reduce_rms(array):
  """Computes the RMS of `array` (in a numerically stable way).

  Args:
    array: Input array.

  Returns:
    The root mean square of the input array as a scalar array.
  """
  sq = jnp.square(array)
  sq_mean = _reduce_mean(sq)
  return jnp.sqrt(sq_mean)


def _clip_update(update, clip_threshold):
  mean_update = _sanitize_values(_reduce_rms(update), 1.0)
  clip_threshold = jnp.array(clip_threshold, dtype=update.dtype)
  denom = jnp.maximum(1.0, mean_update / clip_threshold)

  return update / denom


def clip_updates(clip_threshold=1.0):
  """Implemented update capping as described in adafactor paper.

  http://proceedings.mlr.press/v80/shazeer18a/shazeer18a.pdf

  Args:
    clip_threshold: upper threshold beyond which we scale updates.

  Returns:
    An (init_fn, update_fn) tuple.0
  """

  def init_fn(params):
    del params
    return

  def update_fn(updates, state, params=None):
    del params

    updates = jax.tree_map(lambda x: _clip_update(x, clip_threshold), updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_nadam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 0.5,
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
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_nadam_plus(
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    rescale: bool = True,
    power: float = 0.5,
) -> optax.GradientTransformation:
  """Nadam with convex comb. of preconditioned and un-preconditioned updates.

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    b3: convex comb. coeff. for mixing preconditioned grads with ema of grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    rescale: whether to do rescaling of un-preconditioned updates.
    power: the power to use in the preconditioner (0.5 in default adam).

  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates_rms = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )

    mu_hat_2 = mu_hat
    if rescale:
      # Note: The rescaling factor is different here than Adam because Nadam
      # uses double momentum.
      mu_hat_2 = jax.tree_map(lambda m: m / (1.0 - (b1**2)), mu_hat)
    updates = _update_moment(mu_hat_2, updates_rms, b3, 1)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class PreconditionByLayeredAdaptiveRMSState(NamedTuple):
  """State for the Layered Adaptive RMS Preconditioner algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  nu: optax.Updates
  beta_array: Any
  scale_array: Any


def precondition_by_layered_adaptive_rms(
    decays: List[float],
    scales: List[float],
    decay_distribution: List[float],
    modality: Any = 'output',
    eps: float = 1e-8,
    eps_root: float = 0.0,
    # debias: bool = False,
) -> optax.GradientTransformation:
  """Code implementing beta-factored rms preconditioner.

  Args:
    decays: List of beta values
    scales: List of scales accompanying the beta values. Must of the same length
      as decays.
    decay_distribution: distribution according to which tensors are divided.
      Must be of the same length as decays.
    modality: one of two values 'input' vs 'output' indicating whether factoring
      is perfomed on the input vs the output axis. For conv-nets convention fits
      the Height-Width-InpChannels-OutChannels convention of parameters in flax
      linen Conv
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    count = jnp.zeros([], jnp.int32)
    nu = jax.tree_map(jnp.zeros_like, params)
    beta_array = jax.tree_map(
        lambda x: _generate_per_parameter_array(decays, x), params
    )
    scale_array = jax.tree_map(
        lambda x: _generate_per_parameter_array(scales, x), params
    )

    return PreconditionByLayeredAdaptiveRMSState(
        count=count, nu=nu, beta_array=beta_array, scale_array=scale_array
    )

  def _generate_partition(decays, decay_distribution, length):
    # Generates length-sized array split according to decay_distribution.
    decays = jnp.array(decays)
    decay_distribution = jnp.array(decay_distribution)
    multiples = jnp.int32(jnp.floor(decay_distribution * length))
    multiples = multiples.at[-1].set(
        multiples[-1] + length - jnp.sum(multiples)
    )
    return jnp.repeat(decays, multiples)

  def _generate_per_parameter_array(arr, params):
    # For 1D Tensor - input_Index = output_index = 1
    # For 2D Tensor - input_Index = 0, output_index = 1
    # For > 2D Tensor - input_Index = second last dim , output_index = last dim
    input_index = max(len(params.shape) - 2, 0)
    output_index = len(params.shape) - 1
    array_len = (
        params.shape[output_index]
        if modality == 'output'
        else params.shape[input_index]
    )
    full_array = _generate_partition(arr, decay_distribution, array_len)
    # enlarge beta array to have same ndims as params but ones in shape
    # everywhere but the target_index for easy broadcasting
    target_index = output_index if modality == 'output' else input_index
    new_shape = [1] * jnp.ndim(params)
    new_shape[target_index] = array_len
    return jnp.reshape(full_array, new_shape)

  def _update_moment_general(updates, moments, beta, order):
    # input beta can be an array here
    # the following code handles the adagrad case
    one_minus_beta = 1.0 - beta
    one_minus_beta = jnp.where(
        one_minus_beta <= 0.0, jnp.ones_like(one_minus_beta), one_minus_beta
    )
    return one_minus_beta * (updates**order) + beta * moments

  def update_fn(updates, state, params=None):
    del params
    nu = jax.tree_map(
        lambda g, t, b: _update_moment_general(g, t, b, 2),
        updates,
        state.nu,
        state.beta_array,
    )
    count = state.count + jnp.array(1, dtype=jnp.int32)
    # Decide what to do with bias correction
    # nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v, s: s * (m / (jnp.sqrt(v + eps_root) + eps)),
        updates,
        nu,
        state.scale_array,
    )
    return updates, PreconditionByLayeredAdaptiveRMSState(
        count=count,
        nu=nu,
        beta_array=state.beta_array,
        scale_array=state.scale_array,
    )

  return optax.GradientTransformation(init_fn, update_fn)


class Polyak_AveragingState(NamedTuple):
  """State for Polyak Averaging."""

  ema: optax.Updates


def polyak_averaging(decay: float = 0.999) -> optax.GradientTransformation:
  """Preconditions updates according to the RMS Preconditioner from Adam.

  References:
    [Polyak, Juditsky 1992] -
    https://epubs.siam.org/doi/abs/10.1137/0330046?journalCode=sjcodc

  Args:
    decay: decay rate for exponentially weighted average of moments of params.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return Polyak_AveragingState(ema=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params):
    new_ema = _update_moment(params, state.ema, decay, 1)
    return updates, Polyak_AveragingState(ema=new_ema)

  return optax.GradientTransformation(init_fn, update_fn)


AddDecayedWeightsState = optax.EmptyState


def add_decayed_weights(
    weight_decay: float = 0.0,
    learning_rate: float = 1.0,
    flip_sign: float = False,
) -> optax.GradientTransformation:
  """Add parameter scaled by `weight_decay`.

  Args:
    weight_decay: A scalar weight decay rate.
    learning_rate: An optional learning rate parameter to multiplied with the
      weight decay
    flip_sign: flips the sign of the weight decay operation. Default is False
      under which weight decay is added mirroring the default behavior in optax
      add_decayed_weights. True is to be used when there is no eventual
      scale_by_learning_rate in the chain which flips the sign.

  Returns:
    A `GradientTransformation` object.
  """
  m = -1 if flip_sign else 1

  def init_fn(params):
    del params
    return AddDecayedWeightsState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(optax.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda g, p: g + m * learning_rate * weight_decay * p, updates, params
    )
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


class PytreeScalarState(NamedTuple):
  """State for Scaling Updates Parameter-wise."""

  pytree_scales: Any


def scale_by_pytree(pytree_scales: Any) -> optax.GradientTransformation:
  """Scales the updates parameter-wise.

  Args:
    pytree_scales: Scales for Parameters. Should be a pytree of the same
      structure as params.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return PytreeScalarState(pytree_scales=pytree_scales)

  def update_fn(updates, state, params):
    del params
    return jax.tree_map(lambda x, y: x * y, updates, state.pytree_scales), state

  return optax.GradientTransformation(init_fn, update_fn)


def no_op() -> optax.GradientTransformation:
  """Implements a no_op transformation."""

  def init_fn(params):
    del params
    return optax.EmptyState()

  def update_fn(updates, state, params):
    del params
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


# scale_by_rms exists only for backward compatability
_composites = {
    'scale_by_adaptive_gd': scale_by_adaptive_gd,
    'scale_by_adaptive_gd_simple': scale_by_adaptive_gd_simple,
    'scale_by_layerwise_adaptive_gd': scale_by_layerwise_adaptive_gd,
    'scale_by_layerwise_adaptive_gd_simple': (
        scale_by_layerwise_adaptive_gd_simple
    ),
    'scale_by_coordinate_wise_adaptive_gd': (
        scale_by_coordinate_wise_adaptive_gd
    ),
    'scale_by_coordinate_wise_adaptive_gd_simple': (
        scale_by_coordinate_wise_adaptive_gd_simple
    ),
    'scale_by_adam': scale_by_adam,
    'scale_by_adam_plus': scale_by_adam_plus,
    'scale_by_yogi': optax.scale_by_yogi,
    'scale_by_amsgrad': scale_by_amsgrad,
    'scale_by_nadam': scale_by_nadam,
    'scale_by_nadam_plus': scale_by_nadam_plus,
    'scale_by_rms': precondition_by_rms,
    'scale_by_lamb': functools.partial(optax.lamb, learning_rate=1.0),
    'scale_by_normalized_adam': scale_by_normalized_adam,
    'scale_by_normalized_adam_plus': scale_by_normalized_adam_plus,
    'scale_by_iteration_dependent_norm_adam': (
        scale_by_iteration_dependent_norm_adam
    ),
    'scale_by_adam_var_preserved': scale_by_adam_var_preserved,
    'sgd': optax.sgd,
}

_first_moment_accumulators = {
    'nesterov': nesterov,
    'ema_nesterov': ema_nesterov,
    'polyak_hb': polyak_hb,
    'first_moment_ema': first_moment_ema,
    'compute_params_ema_for_eval': compute_params_ema_for_eval,
    'normalized_first_moment_ema': normalized_first_moment_ema,
    'nesterovpp': nesterovpp,
}

_preconditioners = {
    'precondition_by_rms': precondition_by_rms,
    'precondition_by_yogi': precondition_by_yogi,
    'precondition_by_rss': precondition_by_rss,
    'precondition_by_amsgrad': precondition_by_amsgrad,
    'precondition_by_layered_adaptive_rms': (
        precondition_by_layered_adaptive_rms
    ),
}

_miscellaneous = {
    'scale_by_learning_rate': scale_by_learning_rate,
    'add_decayed_weights': add_decayed_weights,
    'polyak_averaging': polyak_averaging,
    'clip_updates': clip_updates,
    'sanitize_values': sanitize_values,
    'no_op': no_op,
    'scale_by_pytree': scale_by_pytree,
}

transformation_registry = {}
transformation_registry.update(_composites)
transformation_registry.update(_preconditioners)
transformation_registry.update(_first_moment_accumulators)
transformation_registry.update(_miscellaneous)
