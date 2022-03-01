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

"""Hessian-free optimization algorithm."""

from functools import partial  # pylint: disable=g-importing-member
import math
from typing import NamedTuple
import jax
from jax import jit
from jax import jvp
from jax import lax
from jax import vjp
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from optax import apply_updates
from optax._src import base


@jit
def residual_norm_test(step, rs_norm, obj_val, obj_arr, tol):
  """Residual norm test, terminates CG if sqrt(rs_norm) < tol.

  Args:
    step: An integer value of the iteration step counter.
    rs_norm: A residual norm.
    obj_val: A current objective value.
    obj_arr: A jax.numpy array of objective values in recent steps.
    tol: The convergence tolerance.

  Returns:
    A bool value indicating if the test is satisfied.
  """
  del step, obj_val, obj_arr
  return jnp.less(jnp.sqrt(rs_norm), tol)


@jit
def relative_per_iteration_progress_test(step, rs_norm, obj_val, obj_arr, tol):
  """Relative per-iteration progress test proposed by Martens (2010).

  Terminate CG if:
    step > k, f_value(step) < 0, and
    (f_value(step) - f_value(step-k)) / f_value(step) < k * eps.
  For more inforamtion, see Section 4.4 of
  https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf.

  Args:
    step: An integer value of the iteration step counter.
    rs_norm: A residual norm.
    obj_val: A current objective value.
    obj_arr: A jax.numpy array of objective values in recent steps.
    tol: The convergence tolerance.

  Returns:
    A bool value indicating if the test is satisfied.
  """
  del rs_norm

  # k = max(10, ceil(0.1 * step))
  k = lax.max(10, jnp.int32(lax.ceil(0.1 * step)))
  arr_len = len(obj_arr)

  step_condition = jnp.less(k, step)
  negativity_condition = jnp.less(obj_val, 0.)
  progress_condition = jnp.less(
      k * obj_val * tol,
      obj_val - obj_arr[(step + arr_len - k) % arr_len])

  return step_condition & negativity_condition & progress_condition


_TERMINATION_CRITERIA = {
    'residual_norm_test': (False, residual_norm_test),
    'relative_per_iteration_progress_test':
        (True, relative_per_iteration_progress_test),
}


def require_obj_arr(criterion_name):
  """Indicates if the criterion function requires an objective array.

  Args:
    criterion_name: (str) e.g. residual_norm_test.

  Returns:
    A bool indicating if the criterion functions requires an objective array.
  Raises:
    ValueError if criterion_name is unrecognized.
  """
  try:
    return _TERMINATION_CRITERIA[criterion_name][0]
  except KeyError as termination_criterion_not_found_error:
    raise ValueError('Unrecognized criterion name: {}'.format(
        criterion_name)) from termination_criterion_not_found_error


def get_termination_criterion_fn(criterion_name):
  """Get the termination criterion function based on the criterion_name.

  Args:
    criterion_name: (str) e.g. residual_norm_test.

  Returns:
    The termination criterion function.
  Raises:
    ValueError if criterion_name is unrecognized.
  """
  try:
    return _TERMINATION_CRITERIA[criterion_name][1]
  except KeyError as termination_criterion_not_found_error:
    raise ValueError('Unrecognized criterion name: {}'.format(
        criterion_name)) from termination_criterion_not_found_error


@partial(jit, static_argnums=(0, 3, 6, 7))
def mf_conjgrad_solver(matmul_fn,
                       b,
                       x0,
                       max_iter,
                       tol=1e-6,
                       residual_refresh_frequency=10,
                       precond_fn=None,
                       termination_criterion='residual_norm_test'):
  """Solves Ax = b using 'matrix-free' preconditioned conjugate gradient method.

  This implements the preconditioned conjugate gradient algorithm in page 32 of
  http://www.cs.toronto.edu/~jmartens/docs/thesis_phd_martens.pdf in a 'matrix-
  free' manner. 'Matrix-free' means that this method does not require explicit
  knowledge of matrix A and preconditioner P. Instead, it iteratively calls
  linear operators matmul_fn and precond_fn, which return Ax and the solution of
  Py=r for any given x and r, respectively. A termination criterion function
  name can be passed as an argument.

  Args:
    matmul_fn: A linear operator that returns Ax given x.
    b: A numpy vector of shape (n,).
    x0: An initial guess numpy vector of shape (n,).
    max_iter: The number of iterations to run.
    tol: The convergence tolerance.
    residual_refresh_frequency: A frequency to refresh the residual.
    precond_fn: A linear operator that returns the solution of Py=r for any r.
    termination_criterion: A termination criterion function name.

  Returns:
    An approximate solution to the linear system Ax=b.
  """
  if precond_fn is None:
    precond_fn = lambda x: x

  x = x0
  r = matmul_fn(x) - b
  y = precond_fn(r)
  p = -y

  alpha = 0
  beta = 1
  rs_norm = jnp.dot(r, y)

  def cg_objective(x, r):
    """Returns the CG objective function value."""
    return jnp.dot(x, r - b) / 2

  use_obj_arr = require_obj_arr(termination_criterion)
  termination_criterion_fn = get_termination_criterion_fn(termination_criterion)

  obj_val = cg_objective(x, r)
  obj_arr = jnp.array([])

  if use_obj_arr:
    obj_arr = jnp.zeros(max(10, math.ceil(0.1 * max_iter)))
  arr_len = len(obj_arr)

  # define an array to save iterates for CG backtracking
  # an iterate at every ceil(initial_save_step * gamma^j)-th step where j >= 0,
  # and the last iterate will be saved.
  # if the last iteration is ceil(initial_save_step * gamma^j*) for some j*,
  # only one copy will be saved.
  # the max number of copies is ceil(log(max_iter / initial_save_step, gamma))
  # this amounts to 10/13/16/19/28 copies for 50/100/200/500/5000 max_iter
  # when gamma = 1.3 and initial_save_step = 5
  gamma = 1.3
  initial_save_step = 5.0
  x_arr = jnp.zeros(
      (math.ceil(math.log(max_iter / initial_save_step, gamma)) + 1, len(x0)))
  # index to track the last saved element in the array
  x_arr_idx = -1
  next_save_step = initial_save_step

  def termination_condition(state):
    *_, step, rs_norm, obj_val, obj_arr = state
    return jnp.logical_and(
        jnp.less(step, max_iter),
        jnp.equal(
            termination_criterion_fn(
                rs_norm=rs_norm, tol=tol, step=step,
                obj_val=obj_val, obj_arr=obj_arr), False))

  def update_obj_arr(step, obj_val, obj_arr):
    if use_obj_arr:
      return obj_arr.at[step % arr_len].set(obj_val)
    return obj_arr

  def update_x_arr(x, x_arr, x_arr_idx):
    return x_arr.at[x_arr_idx, :].set(x)

  @jit
  def one_step_conjgrad(state):
    """One step of conjugate gradient iteration."""
    x, x_arr, x_arr_idx, save_step, next_save_step, r, y, p, alpha, beta, step, rs_norm, obj_val, obj_arr = state
    obj_arr = update_obj_arr(step, obj_val, obj_arr)

    step += 1

    # Compute Ap
    matmul_product = matmul_fn(p)

    # Update x
    alpha = rs_norm / jnp.dot(p, matmul_product)
    x += alpha * p

    # Update r, y and the square of residual norm
    refresh_residual = jnp.equal(
        jnp.remainder(step, residual_refresh_frequency), 0)
    r = jnp.where(refresh_residual,
                  matmul_fn(x) - b,
                  r + alpha * matmul_product)
    y = precond_fn(r)
    rs_norm_new = jnp.dot(r, y)

    # Compute the objective value
    obj_val = cg_objective(x, r)

    # Update p
    beta = rs_norm_new / rs_norm
    p = beta * p - y

    # Save iterates for CG backtracking
    save_step = jnp.equal(step, jnp.int32(lax.ceil(next_save_step)))
    x_arr_idx = jnp.where(save_step, x_arr_idx + 1, x_arr_idx)
    x_arr = jnp.where(save_step, update_x_arr(x, x_arr, x_arr_idx), x_arr)
    next_save_step *= jnp.where(save_step, gamma, 1)

    return (x, x_arr, x_arr_idx, save_step, next_save_step, r, y, p, alpha,
            beta, step, rs_norm_new, obj_val, obj_arr)

  init_state = x, x_arr, x_arr_idx, False, next_save_step, r, y, p, alpha, beta, 0, rs_norm, obj_val, obj_arr
  x, x_arr, x_arr_idx, save_step, *_ = lax.while_loop(termination_condition,
                                                      one_step_conjgrad,
                                                      init_state)

  # Save the last iterate if not saved yet.
  x_arr_idx = jnp.where(save_step, x_arr_idx, x_arr_idx + 1)
  x_arr = jnp.where(save_step, x_arr, update_x_arr(x, x_arr, x_arr_idx))

  return x_arr, x_arr_idx


def hvp(f, x, v):
  """Returns the product of Hessian matrix and a vector.

  Args:
    f: A callable function that takes a numpy vector of shape (n,).
    x: A numpy vector of shape (n,) where the Hessian is evaluated.
    v: A numpy vector of shape (n,).

  Returns:
    The product of Hessian matrix and a vector
  """
  return jax.jvp(jax.grad(f), [x], [v])[1]


def gvp(variables, outputs, damping, forward_fn, loss_fn, v):
  """Returns the product of generalized Gauss-Newton matrix and a vector.

  Args:
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    outputs: A numpy vector of network outputs computed by forward_fn(params).
    damping: A damping parameter.
    forward_fn: A function that maps params to outputs.
    loss_fn: A loss function.
    v: A numpy vector of shape (n,).

  Returns:
    The product of Generalized Gauss-Newton matrix and a vector
  """
  _, unravel_fn = ravel_pytree(variables)
  jv = jvp(forward_fn, [variables], [unravel_fn(v)])[1]
  hjv = hvp(loss_fn, outputs, jv)
  gvp_fn = vjp(forward_fn, variables)[1]
  return ravel_pytree(gvp_fn(hjv)[0])[0] + damping * v


def cg_backtracking(p_arr, p_arr_idx, obj_fn, variables, unravel_fn):
  """Backtracks CG iterates (Section 4.6, Martens (2010)).

  This function iteratively compares the function values of two consecutive
  iterates. If the function value of the iterate at idx is smaller than the
  function value of the iterate at idx - 1, then the iterate at idx is returned
  as a search direction. Otherwise, we decrease idx by 1 and repeat the
  comparison. If no iterate satisfies the condition, the first element in p_arr
  will be returned.

  Args:
    p_arr: An array of CG iterates of shape (m, n).
    p_arr_idx: The index of the last element in p_arr.
    obj_fn: A function that maps params to a loss value.
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    unravel_fn: A function that maps a numpy vector of shape (n,) to a pytree.

  Returns:
    The backtracked iterate as a pytree and a vector with its function value.
  """
  # Initialize the search direction and compute the objective value along it.
  flattened_p = p_arr[p_arr_idx]

  # We need to make a new dict in order to avoid possible unintended
  # side-effects in the calling function that could happen if we reassigned the
  # keys of "variables", but to avoid copying all the (possibly large) values in
  # the original "variables" we reassign them in the new dict instead of using
  # copy.deepcopy. We should only be calling with train=False in the forward_fn
  # so there should not be any updates to possible "batch_stats" in variables.
  updated_variables = {
      'params': apply_updates(variables['params'], unravel_fn(flattened_p))
  }
  for k, v in variables.items():
    if k != 'params':
      updated_variables[k] = v

  obj_val = obj_fn(updated_variables)

  def termination_condition_cg_backtracking(state):
    *_, idx, keep_backtracking = state
    return jnp.logical_and(keep_backtracking, jnp.greater_equal(idx, 0))

  def one_step_cg_backtracking(state):
    """One step of cg backtracking iteration."""
    flattened_p, obj_val, idx, keep_backtracking = state

    # Compute the objective value for the iterate to be compared with.
    flattened_p_prev = p_arr[idx]
    updated_variables = {
        'params': apply_updates(
            variables['params'], unravel_fn(flattened_p_prev))
    }
    for k, v in variables.items():
      if k != 'params':
        updated_variables[k] = v
    obj_val_prev = obj_fn(updated_variables)

    # Compare the objective values.
    keep_backtracking = jnp.greater_equal(obj_val, obj_val_prev)

    # Update flattened_p and obj_val if obj_val >= obj_val_prev.
    flattened_p = jnp.where(keep_backtracking, flattened_p_prev, flattened_p)
    obj_val = jnp.where(keep_backtracking, obj_val_prev, obj_val)

    return flattened_p, obj_val, idx - 1, keep_backtracking

  init_state = flattened_p, obj_val, p_arr_idx - 1, True
  flattened_p, obj_val, *_ = lax.while_loop(
      termination_condition_cg_backtracking, one_step_cg_backtracking,
      init_state)
  return flattened_p, obj_val


class HessianFreeState(NamedTuple):
  """State for Hessian-free updates.

  p0: An intial guess to the search direction generated by Hessian-free updates.
  damping: A damping parameter.
  """
  p0: jnp.DeviceArray
  damping: float


def hessian_free(flax_module,
                 loss_fn,
                 learning_rate=1.0,
                 max_iter=100,
                 tol=0.0005,
                 residual_refresh_frequency=10,
                 termination_criterion='relative_per_iteration_progress_test'):
  """Hessian-free optimizer.

  Args:
    flax_module: A flax linen.nn.module.
    loss_fn: A loss function.
    learning_rate: A learning rate.
    max_iter: The number of CG iterations.
    tol: The convergence tolerance.
    residual_refresh_frequency: A frequency to refresh the residual.
    termination_criterion: A function chekcing a termination criterion.

  Returns:
    A base.GradientTransformation object of (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    """Initializes the HessianFreeState object for Hessian-free updates."""
    return HessianFreeState(
        p0=ravel_pytree(params)[0],
        damping=1)

  @jit
  def update_fn(grads, state, variables_batch_tuple):
    """Transforms the grads and updates the HessianFreeState object.

    Args:
      grads: pytree of model parameter gradients.
      state: optimizer state (damping and p0 are the used attributes).
      variables_batch_tuple: a tuple of (Dict[str, Any], batch) where the dict
        of variables is passed directly into flax_module.apply, and batch is the
        current minibatch. It is required to have a key 'params'. We need to put
        these into a tuple here so that we can be compatible with the optax API.

    Returns:
      A tuple of (pytree of the model updates, new HessianFreeState).
    """
    variables, batch = variables_batch_tuple

    def partial_forward_fn(variables):
      return flax_module.apply(variables, batch['inputs'], train=False)
    def partial_loss_fn(logits):
      return loss_fn(logits, batch['targets'])

    outputs = partial_forward_fn(variables)
    flattened_grads, unravel_fn = ravel_pytree(grads)

    def matmul_fn(v):
      return lax.pmean(
          gvp(variables, outputs, state.damping, partial_forward_fn,
              partial_loss_fn, v),
          axis_name='batch')

    def obj_fn(variables):
      return lax.pmean(
          partial_loss_fn(partial_forward_fn(variables)), axis_name='batch')

    p_arr, p_arr_idx = mf_conjgrad_solver(matmul_fn, -flattened_grads, state.p0,
                                          max_iter, tol,
                                          residual_refresh_frequency, None,
                                          termination_criterion)
    ## CG backtracking
    # CG solution to be used to initialize the next CG run.
    p_sol = p_arr[p_arr_idx]
    # CG backtracking uses a logarithmic amount of memory to save CG iterates.
    # If this causes OOM, we can consider computing the objective value at
    # each save step in the CG loop and keeping the best one.
    flattened_p, obj_val = cg_backtracking(p_arr,
                                           p_arr_idx,
                                           obj_fn,
                                           variables,
                                           unravel_fn)

    # update the damping parameter
    reduction_f = obj_val - lax.pmean(partial_loss_fn(outputs),
                                      axis_name='batch')
    reduction_q = jnp.dot(flattened_p,
                          flattened_grads + 0.5 * matmul_fn(flattened_p))

    damping_new = state.damping * jnp.where(reduction_f / reduction_q < 0.25,
                                            3.0 / 2.0, 2.0 / 3.0)

    return unravel_fn(flattened_p * learning_rate), HessianFreeState(
        p_sol, damping_new)

  return base.GradientTransformation(init_fn, update_fn)
