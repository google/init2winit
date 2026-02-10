# coding=utf-8
# Copyright 2026 The init2winit Authors.
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

"""Methods for optimizing and manipulating schedules."""

import copy
from typing import Any, Dict
from init2winit.projects.optlrschedule.scheduler import base_schedule_family
import numpy as np


def optimize_base_lr(
    target_sched: np.ndarray, cur_sched: np.ndarray, base_lr: float
) -> float:
  """Optimize base learning rate to minimize L2 loss between schedules.

  Args:
    target_sched: Target schedule to optimize for in np.ndarray format.
    cur_sched: Current schedule, absolue, in np.ndarray format.
    base_lr: Base learning rate for current schedule, which is passed to the
      schedule family.

  Returns:
    Optimized base learning rate for cur_sched
  """
  sched_fac = np.dot(target_sched, cur_sched) / np.dot(cur_sched, cur_sched)
  return sched_fac * base_lr


def l2_sched_loss(sched_1: np.ndarray, sched_2: np.ndarray):
  """L2 loss between two vectorized schedules."""
  return 0.5 * np.mean(np.square(sched_1 - sched_2))


def sched_grad_finite_diff(
    sched_fam: base_schedule_family.BaseScheduleFamily,
    sched_params: Dict[str, Any],
    base_lr: float,
    target_sched: np.ndarray,
    param_constraints=None,
    eps=1e-4,
):
  """Finite differences estimate of gradient of L2 loss between two schedules.

  Args:
    sched_fam: Schedule family object.
    sched_params: Schedule parameters as a dictionary.
    base_lr: Base learning rate for current schedule, which is passed to the
      schedule family.
    target_sched: Target schedule to optimize for in np.ndarray format.
    param_constraints: Optional string indicating parameter constraints.
    eps: Epsilon value for finite differences approximation.

  Returns:
    Dictionary with same keys as schedule parameters, values are gradients.
  """
  cur_sched = sched_fam.get_schedule(sched_params, base_lr)
  # For L2 loss derivative, need difference in schedules * schedule gradient
  base_sched_diff = cur_sched - target_sched
  pert_params = copy.deepcopy(sched_params)
  grad = {}  # Return dictionary with same keys as schedule parameters
  # Perturb each parameter and compute gradient
  for param_name in sched_params.keys():
    neg_pert = False
    if param_constraints == '0_1':  # Constraints handled manually for now
      # Use negative perturbation if values are close to boundary
      if sched_params[param_name] >= 1.0 - 2 * eps:
        neg_pert = True
        pert_params[param_name] -= eps
      else:
        pert_params[param_name] += eps
    else:
      pert_params[param_name] += eps
    pert_sched = sched_fam.get_schedule(pert_params, base_lr)
    pert_sched_diff = pert_sched - cur_sched
    grad[param_name] = np.mean(pert_sched_diff * base_sched_diff) / eps
    if neg_pert:
      grad[param_name] = -grad[param_name]
    pert_params[param_name] = sched_params[param_name]  # Reset
  return grad


def sched_gd_update(
    gd_lr: float,
    sched_fam: base_schedule_family.BaseScheduleFamily,
    sched_params: Dict[str, Any],
    base_lr: float,
    target_sched: np.ndarray,
    param_constraints=None,
    eps=1e-4,
):
  """Compute finite differences gradient approximation for schedule L2 loss.

  Uses finite differences to compute gradient of L2 loss with respect to
  schedule parameters, where L2 loss is the distance between schedule family
  member and a target schedule given as an array of learning rates.
  Then performs gradient descent update on schedule parameters.

  Args:
    gd_lr: Learning rate for gradient descent update.
    sched_fam: Schedule family object.
    sched_params: Schedule parameters as a dictionary.
    base_lr: Base learning rate for current schedule, which is passed to the
      schedule family.
    target_sched: Target schedule to optimize for in np.ndarray format.
    param_constraints: Optional string indicating parameter constraints.
    eps: Epsilon value for finite differences approximation.

  Returns:
    Updated schedule parameters and new base learning rate for schedule family.
  """
  grad_est = sched_grad_finite_diff(
      sched_fam, sched_params, base_lr, target_sched, param_constraints, eps
  )
  # Gradient step
  for param_name in sched_params.keys():
    if param_constraints == '0_1':  # Constraints handled manually for now
      proposed_delta = gd_lr * grad_est[param_name]
      if proposed_delta >= 0.5:  # Manually shrink large updates
        proposed_delta = 0.1 * np.sign(proposed_delta)
      new_val = sched_params[param_name] - proposed_delta
      if new_val <= 0:
        sched_params[param_name] = 1e-5  # Tolerance issues, need to debug
      elif new_val >= 1:
        sched_params[param_name] = 1.0 - 1e-5
      else:
        sched_params[param_name] = new_val
    else:
      sched_params[param_name] -= gd_lr * grad_est[param_name]
  # Re-optimize base lr
  new_sched = sched_fam.get_schedule(sched_params, base_lr)
  new_base_lr = optimize_base_lr(target_sched, new_sched, base_lr)
  return sched_params, new_base_lr


def get_filtered_loss_fn(loss_fn, accel_pow=0.0):
  """Reshape loss function by raising to (1-accel_pow).
  
  Args:
    loss_fn: loss function to be filtered
    accel_pow: power to which to raise loss function. 0 is no change, values
      approaching 1 from below limit to log.
  Returns:
    Filtered loss function g(x) = f(x)**(1-accel_pow)/(1-accel_pow).
  """
  def filtered_loss_fn(*args, **kwargs):
    return (loss_fn(*args, **kwargs)**(1.-accel_pow))/(1.-accel_pow)
  return filtered_loss_fn


def base_lr_grid_search(lrs,
                        loss_from_lrs,
                        grid_steps=5,
                        lr_fac=4.0,
                        par=False):
  """Grid search over base learning rate for given schedule shape.
  
  Args:
    lrs: array of learning rates
    loss_from_lrs: function that takes array of learning rates as input, and
      returns final loss
    grid_steps: number of steps in one direction of grid
    lr_fac: multiplicative factor for max of search
    par: if True, loss_from_lrs is alread parallelized over schedules
  Returns:
    Base learning rate that minimizes final loss.
  """
  log_fac = np.log(lr_fac)
  base_lrs = np.exp(np.linspace(-log_fac, log_fac, grid_steps*2+1))
  final_losses = np.zeros(grid_steps)
  if par:
    lr_mat = np.outer(base_lrs, lrs)
    final_losses = loss_from_lrs(lr_mat)
    min_idx = np.nanargmin(final_losses)
    return lr_mat[min_idx]
  else:
    for i, base_lr in enumerate(base_lrs):
      final_losses[i] = loss_from_lrs(base_lr * lrs)
    min_idx = np.nanargmin(final_losses)
    return base_lrs[min_idx] * lrs


def early_base_lr_search(
    lrs,
    loss_from_lrs,
    grid_step_list=(10, 10, 10),
    lr_facs=(100.0, 2.0, 1.4),
    par=False,
):
  """Base lr grid search procedure for initial schedule search."""
  for grid_steps, lr_fac in zip(grid_step_list, lr_facs):
    lrs = base_lr_grid_search(lrs, loss_from_lrs, grid_steps, lr_fac, par)
  return lrs


def schedule_descent_step(
    sched_lr, lrs, prev_loss, sched_deriv_fn, loss_thresh=1e1, nan_fac=0.3
):
  """Single step of schedule descent.

  Args:
    sched_lr: schedule GD learning rate
    lrs: current schedule
    prev_loss: previous loss
    sched_deriv_fn: function that takes schedule as input and returns derivative
      of loss function with respect to schedule
    loss_thresh: upper threshold for loss to detect too large learning rate
    nan_fac: base learning rate reduction factor for large or negative learning
      rates

  Returns:
    Next schedule.
  """
  if np.isnan(prev_loss) or prev_loss > loss_thresh:
    return (1.0 - nan_fac) * lrs
  else:
    g_lrs = sched_deriv_fn(lrs)
    lrs_next = lrs - sched_lr * g_lrs
    if min(lrs_next) < 0:
      return (1.0 - nan_fac) * lrs
    else:
      return lrs_next
