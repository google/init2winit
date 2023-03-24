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

"""Hessian-free optimization algorithm."""

import enum
from functools import partial  # pylint: disable=g-importing-member
import math
from typing import NamedTuple

import chex
import jax
from jax import jit
from jax import jvp
from jax import lax
from jax import vjp
import jax.numpy as jnp
from optax import apply_updates
from optax._src import base
import tree_math as tm
import tree_math.numpy as tnp


@jit
def residual_norm_test(step, rss, obj_val, obj_arr, tol):
  """Residual norm test, terminates CG if residual_norm < tol.

  Args:
    step: An integer value of the iteration step counter.
    rss: A residual sum of squares.
    obj_val: A current objective value.
    obj_arr: A jax.numpy array of objective values in recent steps.
    tol: The convergence tolerance.

  Returns:
    A bool value indicating if the test is satisfied.
  """
  del step, obj_val, obj_arr
  return jnp.less(jnp.sqrt(rss), tol)


@jit
def relative_per_iteration_progress_test(step, rss, obj_val, obj_arr, tol):
  """Relative per-iteration progress test proposed by Martens (2010).

  Terminate CG if:
    step > k, obj_val(step) < 0, and
    (obj_val(step) - obj_val(step-k)) / obj_val(step) < k * eps.
  For more inforamtion, see Section 4.4 of
  https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf.

  Args:
    step: An integer value of the iteration step counter.
    rss: A residual sum of squares.
    obj_val: A current objective value.
    obj_arr: A jax.numpy array of objective values in recent steps.
    tol: The convergence tolerance.

  Returns:
    A bool value indicating if the test is satisfied.
  """
  del rss

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


@jit
@partial(tm.unwrap, vector_argnames=['x'])
def tree_slice(x, idx):
  """Slices the pytree using the given index."""
  return jax.tree_map(lambda x: x[idx], x)


def generate_updated_variables(variables, params):
  """Generates a new dict of variables using the params argument.

  We need to make a new dict in order to avoid possible unintended
  side-effects in the calling function that could happen if we reassigned
  some of the keys of "variables", but to avoid copying all the (possibly large)
  values in the original "variables" we reassign them in the new dict
  instead of using copy.deepcopy. We should only be calling with
  train=False in the forward_fn so there should not be any updates to
  possible "batch_stats" in variables.

  Args:
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    params: A pytree of model parameters.

  Returns:
    A dict of variables with the params field replaced with the params argument.

  """
  updated_variables = {'params': params}
  for k, v in variables.items():
    if k != 'params':
      updated_variables[k] = v
  return updated_variables


class CGIterationTrackingMethod(enum.Enum):
  """Methods to track iterates in the conjugate gradient solver.

  LAST_TRACKING means only the last iterate will be tracked.

  BEST_TRACKING means that an objective value will be computed at each tracking
  step and the iterate with the best objective value will be tracked.

  BACK_TRACKING means iterates at tracking steps will be saved and backtracked
  later to find an iterate that has a better objective value than the one saved
  right before it.
  """
  LAST_TRACKING = 'last_tracking'
  BEST_TRACKING = 'best_tracking'
  BACK_TRACKING = 'back_tracking'


# pylint: disable=invalid-name
@partial(jit, static_argnums=(0, 3, 6, 7, 8, 11))
def mf_conjgrad_solver(
    A_fn,
    b,
    x0,
    max_iter,
    tol=1e-6,
    residual_refresh_frequency=10,
    precond_fn=None,
    termination_criterion='residual_norm_test',
    iter_tracking_method=CGIterationTrackingMethod.LAST_TRACKING,
    initial_tracking_step=5.0,
    next_tracking_step_multiplier=1.3,
    obj_fn=None,
    variables=None):
  """Solves Ax = b using 'matrix-free' preconditioned conjugate gradient method.

  This implements the preconditioned conjugate gradient algorithm in page 32 of
  http://www.cs.toronto.edu/~jmartens/docs/thesis_phd_martens.pdf in a 'matrix-
  free' manner. 'Matrix-free' means that this method does not require explicit
  knowledge of matrix A and preconditioner P. Instead, it iteratively calls
  linear operators A_fn and precond_fn, which return Ax and the solution of
  Py=r for any given x and r, respectively. A termination criterion function
  name can be passed as an argument.

  Args:
    A_fn: A linear operator that returns Ax given a tm.Vector x.
    b: A right-hand side tm.Vector.
    x0: An initial guess tm.Vector.
    max_iter: The maximum number of iterations to run.
    tol: The convergence tolerance.
    residual_refresh_frequency: A frequency to refresh the residual.
    precond_fn: A linear operator that returns the solution of Py=r for any r.
    termination_criterion: A termination criterion function name.
    iter_tracking_method: A CGIterationTrackingMethod.
    initial_tracking_step: The first step to track an iterate.
    next_tracking_step_multiplier: A constant used to determine next track step.
    obj_fn: A function that maps variables to a loss value.
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.

  Returns:
    An approximate solution to the linear system Ax=b.
  """
  if precond_fn is None:
    precond_fn = lambda x: x
  x = x0
  r = A_fn(x) - b
  y = precond_fn(r)
  p = -y

  alpha = 0
  beta = 1
  rss = r @ y

  def cg_objective(x, r):
    """Returns the CG objective function value."""
    return x @ (0.5 * (r - b))

  use_obj_arr = require_obj_arr(termination_criterion)
  termination_criterion_fn = get_termination_criterion_fn(termination_criterion)

  obj_val = cg_objective(x, r)
  obj_arr = jnp.array([])

  if use_obj_arr:
    obj_arr = jnp.zeros(max(10, math.ceil(0.1 * max_iter)))
  arr_len = len(obj_arr)

  ## CG iteration best-tracking
  # an iterate with the best objective is tracked with its objective value
  x_best = jax.tree_map(lambda x: jnp.array([]), x)
  x_best_obj = 0.0
  if iter_tracking_method == CGIterationTrackingMethod.BEST_TRACKING:
    x_best = x0
    x_best_obj = get_obj_val(obj_fn, variables, x_best)

  ## CG iteration backtracking
  # a tm.Vector of an array of iterates for backtracking
  # iterates at every ceil(initial_tracking_step * next_tracking_step_multiplier
  # ^j)-th step (j >= 0), and the last iterate (if not saved in the loop) will
  # be saved. ceil(log(max_iter / initial_track_step, gamma)) is the max number
  # of copies. this amounts to 10/13/16/19/28 for 50/100/200/500/5000 max_iter
  # when initial_track_step = 5 and next_tracking_step_multiplier = 1.3.
  x_arr = jax.tree_map(lambda x: jnp.array([]), x)
  x_arr_idx = -1  # index to track the last saved element in x_arr
  if iter_tracking_method == CGIterationTrackingMethod.BACK_TRACKING:
    max_save_size = math.ceil(
        math.log(max_iter / initial_tracking_step,
                 next_tracking_step_multiplier)) + 1
    # define a pytree to save iterates for backtracking
    x_arr = jax.tree_map(lambda x: jnp.zeros((max_save_size, *x.shape)), x)

  next_tracking_step = initial_tracking_step

  def conditional_iteration_tracking_update(x, x_best, x_best_obj, x_arr,
                                            x_arr_idx, condition):
    if iter_tracking_method == CGIterationTrackingMethod.BEST_TRACKING:
      x_obj = jnp.where(condition, get_obj_val(obj_fn, variables, x),
                        x_best_obj)
      update_x_best = jnp.less(x_obj, x_best_obj)
      x_best_obj = jnp.where(update_x_best, x_obj, x_best_obj)
      x_best = tnp.where(update_x_best, x, x_best)

    if iter_tracking_method == CGIterationTrackingMethod.BACK_TRACKING:
      x_arr_idx = jnp.where(condition, x_arr_idx + 1, x_arr_idx)
      x_arr = conditional_tree_index_update(x_arr, x, x_arr_idx, condition)

    return x_best, x_best_obj, x_arr, x_arr_idx

  def termination_condition(state):
    *_, obj_val, obj_arr, step, rss = state
    return jnp.logical_and(
        jnp.less(step, max_iter),
        jnp.equal(
            termination_criterion_fn(
                rss=rss, tol=tol, step=step,
                obj_val=obj_val, obj_arr=obj_arr), False))

  def index_update(arr, idx, val):
    return arr.at[idx].set(val)

  @partial(tm.unwrap, vector_argnames=['orig', 'new'])
  def conditional_tree_index_update(orig, new, idx, condition):
    return jax.tree_map(lambda x, y: jnp.where(condition, x.at[idx].set(y), x),
                        orig, new)

  def _one_step_conjgrad(x, x_best, x_best_obj, x_arr, x_arr_idx,
                         should_track_step, next_tracking_step, r, y, p, alpha,
                         beta, obj_val, obj_arr, step, rss):

    if use_obj_arr:
      obj_arr = index_update(obj_arr, step % arr_len, obj_val)

    step += 1

    # Compute Ap
    Ap = A_fn(p)

    # Update x
    alpha = rss / (p @ Ap)
    x = x + p * alpha

    # Update r, y and the square of residual norm
    refresh_residual = jnp.equal(
        jnp.remainder(step, residual_refresh_frequency), 0)
    r = tnp.where(refresh_residual, A_fn(x) - b, r + alpha * Ap)
    y = precond_fn(r)
    rss_new = r @ y

    # Compute the objective value
    obj_val = cg_objective(x, r)

    # Update p
    beta = rss_new / rss
    p = p * beta - y

    # Update for CG iteration tracking
    should_track_step = jnp.equal(step, jnp.int32(lax.ceil(next_tracking_step)))
    x_best, x_best_obj, x_arr, x_arr_idx = conditional_iteration_tracking_update(
        x, x_best, x_best_obj, x_arr, x_arr_idx, should_track_step)
    next_tracking_step *= jnp.where(should_track_step,
                                    next_tracking_step_multiplier, 1)
    return (x, x_best, x_best_obj, x_arr, x_arr_idx, should_track_step,
            next_tracking_step, r, y, p, alpha, beta, obj_val, obj_arr,
            step, rss_new)

  @jit
  def one_step_conjgrad(state):
    """One step of conjugate gradient iteration."""
    return _one_step_conjgrad(*state)

  init_state = (x, x_best, x_best_obj, x_arr, x_arr_idx, False,
                next_tracking_step, r, y, p, alpha, beta, obj_val, obj_arr, 0,
                rss)
  x, x_best, x_best_obj, x_arr, x_arr_idx, track_step, *_, step, rss = lax.while_loop(
      termination_condition, one_step_conjgrad, init_state)

  # Track the step if not tracked yet.
  x_best, x_best_obj, x_arr, x_arr_idx = conditional_iteration_tracking_update(
      x, x_best, x_best_obj, x_arr, x_arr_idx, jnp.logical_not(track_step))

  # Whatever the tracked solution is, the last iterate will be used to
  # initialize the next CG run.

  if iter_tracking_method == CGIterationTrackingMethod.BEST_TRACKING:
    return x, x_best, x_best_obj, step, rss

  # CG iteration backtracking uses a logarithmic amount of memory.
  # If this causes OOM, switch to CGIterationTrackingMethod.BEST_TRACKING.
  if iter_tracking_method == CGIterationTrackingMethod.BACK_TRACKING:
    return x, *cg_backtracking(x_arr, x_arr_idx, obj_fn, variables), step, rss

  x_obj = None
  if obj_fn:
    x_obj = get_obj_val(obj_fn, variables, x)

  return x, x, x_obj, step, rss


# pylint: enable=invalid-name
def hvp(f, x, v):
  """Returns the product of the Hessian matrix and a pytree.

  Args:
    f: A callable function that takes a pyree.
    x: A pytree where the Hessian is evaluated.
    v: A pytree to be multiplied by the Hessian matrix.

  Returns:
    A pytree of the product of the Hessian matrix and a pytree.
  """
  return jax.jvp(jax.grad(f), [x], [v])[1]


def gvp(variables, outputs, damping, forward_fn, loss_fn, v):
  """Returns the product of the Gauss-Newton matrix and a pytree.

  Args:
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    outputs: A jnp.array of network outputs computed by forward_fn(params).
    damping: A damping parameter.
    forward_fn: A function that maps variables to outputs.
    loss_fn: A loss function.
    v: A pytree to be multiplied by the Gauss-Newton matrix.

  Returns:
    A pytree of the product of the Gauss-Newton matrix and a pytree.
  """
  jv = jvp(forward_fn, [variables],
           [generate_updated_variables(variables, v)])[1]
  hjv = hvp(loss_fn, outputs, jv)
  gvp_fn = vjp(forward_fn, variables)[1]
  return jax.tree_map(lambda x, y: x + damping * y, gvp_fn(hjv)[0]['params'], v)


@partial(tm.unwrap, vector_argnames=['updates'], out_vectors=False)
def get_obj_val(obj_fn, variables, updates):
  """Computes the function value after applying updates to params in variables.

  This function constructs a new set of model parameters by adding updates
  to the model parameters in variables and computes the objective vale at the
  updated model parameters.

  Args:
    obj_fn: A function that maps variables to a loss value.
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    updates: A tm.Vector of updates.

  Returns:
    The objective value at the new params.
  """
  new_params = apply_updates(variables['params'], updates)
  updated_variables = generate_updated_variables(variables, new_params)
  return obj_fn(updated_variables)


def cg_backtracking(p_arr, p_arr_idx, obj_fn, variables):
  """Backtracks CG iterates (Section 4.6, Martens (2010)).

  This function iteratively compares the function values of two consecutive
  iterates. If the function value of the iterate at idx is smaller than the
  function value of the iterate at idx - 1, then the iterate at idx is returned
  as a search direction. Otherwise, we decrease idx by 1 and repeat the
  comparison. If no iterate satisfies the condition, the first element in p_arr
  will be returned.

  Args:
    p_arr: A pytree of an array of CG iterates of the shape (n, *p.shape).
    p_arr_idx: The index of the last stored element in p_arr.
    obj_fn: A function that maps variables to a loss value.
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.

  Returns:
    The backtracked iterate as a tm.Vector and the objective value at it.
  """

  def termination_condition_cg_backtracking(state):
    *_, idx, keep_backtracking = state
    return jnp.logical_and(keep_backtracking, jnp.greater_equal(idx, 0))

  def one_step_cg_backtracking(state):
    """One step of cg backtracking iteration."""
    p, obj_val, idx, keep_backtracking = state

    # Compute the objective value for the iterate to be compared with.
    p_prev = tree_slice(p_arr, idx)
    obj_val_prev = get_obj_val(obj_fn, variables, p_prev)

    # Compare the objective values.
    keep_backtracking = jnp.greater_equal(obj_val, obj_val_prev)

    # Update p and obj_val if obj_val >= obj_val_prev.
    p = jax.tree_map(lambda x, y: jnp.where(keep_backtracking, x, y), p_prev, p)
    obj_val = jnp.where(keep_backtracking, obj_val_prev, obj_val)

    return p, obj_val, idx - 1, keep_backtracking

  # Initialize the search direction and compute the objective value along it.
  p = tree_slice(p_arr, p_arr_idx)
  obj_val = get_obj_val(obj_fn, variables, p)

  init_state = p, obj_val, p_arr_idx - 1, True
  p, obj_val, *_ = lax.while_loop(
      termination_condition_cg_backtracking, one_step_cg_backtracking,
      init_state)
  return p, obj_val


def line_search(initial_lr,
                initial_obj_val,
                obj_fn,
                variables,
                grads,
                p,
                sufficient_decrease_constant=10**-2,
                shrinkage_factor=0.8,
                max_line_search_step=60):
  """Determines the learning rate using backtracking line search.

  Incrementing step from 0 to max_line_search_step, this method finds
    lr(step) = initial_lr * shrinkage_factor ** step
  that satisfies the Armijo-Goldstein inequality:
    get_obj_val(obj_fn, variables, lr(step) * p) <=
    obj_fn(variables) + sufficient_decrease_constant * lr(step) * dot(p, grads).
  If this is not met until max_line_search_step, returns 0.0 (no update).

  Args:
    initial_lr: A learning rate to start line search with.
    initial_obj_val: The objective value with the initial step size.
    obj_fn: A function that maps variables to a loss value.
    variables: A dict of variables is passed directly into flax_module.apply,
      required to have a key 'params' that is a pytree of model parameters.
    grads: A tm.Vector of model parameter gradients.
    p: A tm.Vector of search direction.
    sufficient_decrease_constant: A constant in the Armijo-Goldstein inequality.

    shrinkage_factor: A constant used to shrink the learning rate.
    max_line_search_step: The max number of line search steps.

  Returns:
    A step size on [0.0, initial_lr].
  """
  obj_val_ref = obj_fn(variables)
  p_dot_grads = p @ grads
  def line_search_should_continue(state):
    step, lr, obj_val = state
    return jnp.logical_and(
        step <= max_line_search_step,
        jnp.greater(
            obj_val,
            obj_val_ref + sufficient_decrease_constant * lr * p_dot_grads))

  def one_step_line_search(state):
    """One step of line search iteration."""
    step, lr, _ = state
    lr *= shrinkage_factor
    obj_val = get_obj_val(obj_fn, variables, lr * p)

    return step + 1, lr, obj_val

  init_state = 0, initial_lr, initial_obj_val
  step, lr, _ = lax.while_loop(
      line_search_should_continue, one_step_line_search, init_state)

  return jnp.where(step == max_line_search_step, 0.0, lr)


def update_damping(damping, rho, damping_ub, damping_lb):
  """Updates the damping parameter."""
  damping_new = damping * jnp.where(rho < 0.25, 1.5,
                                    jnp.where(rho > 0.75, 2.0 / 3.0, 1.0))
  damping_new = jnp.where(damping_new > damping_ub, damping_ub, damping_new)
  damping_new = jnp.where(damping_new < damping_lb, damping_lb, damping_new)
  return damping_new


class HessianFreeState(NamedTuple):
  """State for Hessian-free updates.

  p0: An intial guess to the search direction generated by Hessian-free updates.
  damping: A damping parameter.
  total_cg_steps
  """
  p0: base.Params
  damping: chex.Array
  total_cg_steps: chex.Array
  final_lr: chex.Array


def hessian_free(
    flax_module,
    training_objective_fn,
    learning_rate=1.0,
    cg_max_iter=100,
    cg_iter_tracking_method=CGIterationTrackingMethod.BACK_TRACKING,
    tol=0.0005,
    residual_refresh_frequency=10,
    termination_criterion='relative_per_iteration_progress_test',
    use_line_search=True,
    line_search_sufficient_increase_constant=10**(-2),
    line_search_shrinkage_factor=0.8,
    max_line_search_step=60,
    warmstart_refresh_rss_threshold=1.0,
    init_damping=50.0,
    damping_ub=10**2,
    damping_lb=10**-6):
  """Hessian-free optimizer.

  In this implementation, every dot product is computed by tree_math.numpy.dot.
  Note that this might have a different default precision than jax.numpy.dot.
  For more information, see
  https://github.com/google/tree-math/blob/main/tree_math/_src/vector.py#L104.

  Args:
    flax_module: A flax linen.nn.module.
    training_objective_fn: A training objective function.
    learning_rate: A learning rate.
    cg_max_iter: The max number of CG iterations.
    cg_iter_tracking_method: A CGIterationTrackingMethod.
    tol: The convergence tolerance.
    residual_refresh_frequency: A frequency to refresh the residual.
    termination_criterion: A function chekcing a termination criterion.
    use_line_search: A bool indicating whether to use line search.
    line_search_sufficient_increase_constant: A constant for backtracking line
      search.
    line_search_shrinkage_factor: A constant used to shrink the learning rate.
    max_line_search_step: The max number of line search steps.
    warmstart_refresh_rss_threshold: A rss threshold used to refresh warmstart.
    init_damping: An initial damping value.
    damping_ub: The upper bound of the damping parameter.
    damping_lb: The lower bound of the damping parameter.

  Returns:
    A base.GradientTransformation object of (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    """Initializes the HessianFreeState object for Hessian-free updates."""
    return HessianFreeState(
        p0=jax.tree_map(jnp.zeros_like, params),
        final_lr=jnp.zeros([]),
        damping=jnp.array(init_damping),
        total_cg_steps=jnp.zeros([], jnp.int32))

  @partial(tm.wrap, vector_argnames=['grads', 'p0'])
  def _update_fn(grads, p0, damping, total_cg_steps, variables, batch):
    def forward_fn(variables):
      return flax_module.apply(variables, batch['inputs'], train=False)

    def loss_fn(logits):
      return training_objective_fn(
          variables['params'], logits, batch['targets'], batch.get('weights'))

    outputs = forward_fn(variables)

    @tm.unwrap
    def matmul_fn(v):
      """Computes the product of the Gauss-Newton matrix and a tm.Vector.

      Args:
        v: a tm.Vector to be multiplied by a Gauss-Newton matrix. This
          is converted to a pytree by the wrapper.

      Returns:
        A tm.Vector of the product of the Gauss-Newton matrix and a tm.Vector.
      """
      return lax.pmean(
          gvp(variables, outputs, damping, forward_fn, loss_fn, v),
          axis_name='batch')

    def obj_fn(variables):
      return lax.pmean(loss_fn(forward_fn(variables)), axis_name='batch')

    p_warmstart, p, obj_val, cg_steps, rss = mf_conjgrad_solver(
        A_fn=matmul_fn,
        b=-grads,
        x0=p0,
        max_iter=cg_max_iter,
        tol=tol,
        residual_refresh_frequency=residual_refresh_frequency,
        precond_fn=None,
        termination_criterion=termination_criterion,
        iter_tracking_method=cg_iter_tracking_method,
        obj_fn=obj_fn,
        variables=variables)
    total_cg_steps += cg_steps

    # Update the damping parameter.
    reduction_f = obj_val - lax.pmean(loss_fn(outputs), axis_name='batch')
    reduction_q = p @ (grads + 0.5 * matmul_fn(p))
    rho = reduction_f / reduction_q
    damping_new = update_damping(damping, rho, damping_ub, damping_lb)

    # Line search
    final_lr = learning_rate
    if use_line_search:
      initial_lr = learning_rate
      initial_obj_val = get_obj_val(obj_fn, variables, initial_lr * p)
      final_lr = line_search(initial_lr, initial_obj_val, obj_fn, variables,
                             grads, p, line_search_sufficient_increase_constant,
                             line_search_shrinkage_factor, max_line_search_step)

    refresh_warmstart = jnp.logical_or(final_lr == 0.0,
                                       rss > warmstart_refresh_rss_threshold)
    p_warmstart = (1 - refresh_warmstart) * p_warmstart

    return final_lr * p, HessianFreeState(
        p0=p_warmstart,
        final_lr=final_lr,
        damping=damping_new,
        total_cg_steps=total_cg_steps)

  @jit
  def update_fn(grads, state, variables_batch_tuple):
    """Transforms the grads and updates the HessianFreeState object.

    Args:
      grads: pytree of model parameter gradients. This is converted to a
        tm.Vector by the wrapper.
      state: optimizer state (damping and p0 are the used attributes).
      variables_batch_tuple: a tuple of (Dict[str, Any], batch) where the dict
        of variables is passed directly into flax_module.apply, and batch is the
        current minibatch. It is required to have a key 'params'. We need to put
        these into a tuple here so that we can be compatible with the optax API.

    Returns:
      A tuple of (pytree of the model updates, new HessianFreeState).
    """
    variables, batch = variables_batch_tuple
    return _update_fn(grads, state.p0, state.damping, state.total_cg_steps,
                      variables, batch)

  return base.GradientTransformation(init_fn, update_fn)

