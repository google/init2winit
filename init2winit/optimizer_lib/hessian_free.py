# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

from jax import jit
from jax import lax
import jax.numpy as jnp
from jax.ops import index
from jax.ops import index_update


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

  k = jnp.where(jnp.less(10, step // 10), step // 10, 10)
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
  except KeyError:
    raise ValueError('Unrecognized criterion name: {}'.format(criterion_name))


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
  except KeyError:
    raise ValueError('Unrecognized criterion name: {}'.format(criterion_name))


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
    obj_arr = jnp.zeros(max(10, max_iter // 10))
  arr_len = len(obj_arr)

  def termination_condition(state):
    *_, step, rs_norm, obj_val, obj_arr = state
    return jnp.logical_and(
        jnp.less(step, max_iter),
        jnp.equal(
            termination_criterion_fn(
                rs_norm=rs_norm, tol=tol, step=step-1,
                obj_val=obj_val, obj_arr=obj_arr), False))

  def update_obj_arr(step, obj_val, obj_arr):
    if use_obj_arr:
      return index_update(obj_arr, index[step % arr_len], obj_val)
    return obj_arr

  @jit
  def one_step_conjgrad(state):
    """One step of conjugate gradient iteration."""
    x, r, y, p, alpha, beta, step, rs_norm, obj_val, obj_arr = state

    obj_arr = update_obj_arr(step, obj_val, obj_arr)

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

    return (x, r, y, p, alpha, beta, step + 1, rs_norm_new, obj_val, obj_arr)

  init_state = x, r, y, p, alpha, beta, 0, rs_norm, obj_val, obj_arr
  x, *_ = lax.while_loop(termination_condition, one_step_conjgrad, init_state)

  return x
