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
