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

"""Defines different learning_rate schedules."""

import jax.numpy as jnp


def _check_schedule_hparams(schedule_hparams, expected_keys):
  if set(schedule_hparams.keys()) != set(expected_keys):
    raise ValueError(
        'Provided schedule_hparams keys are invalid. Recieved: {}, Expected: {}'
        .format(sorted(schedule_hparams.keys()), sorted(expected_keys)))


def constant_schedule(schedule_hparams, max_training_steps):
  del max_training_steps
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr'])
  return lambda t: schedule_hparams['base_lr']


def cosine_schedule(schedule_hparams, max_training_steps):
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr'])

  def lr_fn(t):
    decay_factor = (1 + jnp.cos(t / max_training_steps * jnp.pi)) * 0.5
    return schedule_hparams['base_lr'] * decay_factor
  return lr_fn


def polynomial_schedule(schedule_hparams, max_training_steps):
  """Same behavior as tf.train.polynomial_decay.

  Supports either decay_steps or decay_steps_factor, but providing both is an
  error.

  Args:
    schedule_hparams: Relevant hparams are schedule,
      base_lr, end_factor, power, and one of decay_steps or
      decay_steps_factor.
    max_training_steps: Only used when decay_steps_factor is provided.

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  expected_keys = ['schedule', 'base_lr', 'end_factor', 'power']
  if 'decay_steps' in schedule_hparams:
    expected_keys.append('decay_steps')
    decay_steps = schedule_hparams.decay_steps
  else:
    expected_keys.append('decay_steps_factor')
    decay_steps = int(
        max_training_steps * schedule_hparams['decay_steps_factor'])
  _check_schedule_hparams(schedule_hparams, expected_keys)

  end_learning_rate = schedule_hparams['base_lr'] * schedule_hparams[
      'end_factor']

  def lr_fn(t):
    step = min(decay_steps, t)
    decayed_learning_rate = (schedule_hparams['base_lr'] -
                             end_learning_rate) * (1 - step / decay_steps)**(
                                 schedule_hparams['power']) + end_learning_rate
    return decayed_learning_rate
  return lr_fn


def piecewise_constant_schedule(schedule_hparams, max_training_steps):
  """Computes a piecewise constant decay schedule.

  Note that each element of decay_factors is absolute (not relative). For
  example, to decay the learning rate to 0.5 of its initial value after
  100 steps, followed by 0.1 of its *initial value* after 200 steps,
  with a plateau of 0.1 of its initial value thereafter, use
  decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Args:
    schedule_hparams: Relevant hparams are base_lr, decay_events
      decay_factors.
    max_training_steps: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_steps
  _check_schedule_hparams(schedule_hparams, [
      'schedule', 'base_lr', 'decay_events',
      'decay_factors'
  ])
  boundaries = jnp.array([0] + schedule_hparams['decay_events'])
  factors = [1.0] + schedule_hparams['decay_factors']
  def lr_fn(t):
    index = jnp.sum(boundaries[1:] < t)
    return factors[index] * schedule_hparams['base_lr']

  return lr_fn


def piecewise_linear_schedule(schedule_hparams, max_training_steps):
  """Computes a piecewise linear decay schedule.

  Note that each element of decay_factors is absolute (not relative). For
  example, to decay the learning rate linearly to 0.5 of its initial value after
  100 steps, followed by linearly to 0.1 of its *initial value* after 200 steps,
  with a plateau of 0.1 of its initial value thereafter, use
  decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Args:
    schedule_hparams: Relevant hparams are base_lr, decay_events
      decay_factors.
    max_training_steps: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_steps
  _check_schedule_hparams(schedule_hparams, [
      'schedule', 'base_lr', 'decay_events',
      'decay_factors'
  ])
  boundaries = jnp.array([0] + schedule_hparams['decay_events'])
  factors = [1.0] + schedule_hparams['decay_factors']
  def lr_fn(t):
    index = jnp.sum(boundaries[1:] < t)
    if index+1 == len(factors):
      return factors[index] * schedule_hparams['base_lr']
    m = (factors[index + 1] - factors[index]) / (
        boundaries[index + 1] - boundaries[index])
    interpolated_factor = m * (t - boundaries[index]) + factors[index]
    return schedule_hparams['base_lr'] * interpolated_factor

  return lr_fn


# TODO(gilmer) Change the code path before open source.
def mlperf_polynomial_schedule(schedule_hparams, max_training_steps):
  """Polynomial learning rate schedule for LARS optimizer.

  This function is copied from
  third_party/tensorflow_models/mlperf/models/rough/resnet_jax/train.py and
  modified to fit the init2winit API.

  Args:
    schedule_hparams: Relevant hparams are base_lr, warmup_steps.
    max_training_steps: Used to calculate the number of decay steps.

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  _check_schedule_hparams(
      schedule_hparams,
      ['schedule', 'base_lr', 'warmup_steps', 'power', 'start_lr', 'end_lr',
       'decay_end', 'warmup_power'])
  decay_steps = max_training_steps - schedule_hparams.warmup_steps + 1
  end_lr = schedule_hparams['end_lr']
  def step_fn(step):
    decay_end = schedule_hparams['decay_end']
    if decay_end > 0 and step >= decay_end:
      step = decay_end
    r = (step / schedule_hparams.warmup_steps) ** schedule_hparams.warmup_power
    warmup_lr = (
        schedule_hparams.base_lr * r + (1 - r) * schedule_hparams.start_lr)
    decay_step = jnp.minimum(step - schedule_hparams.warmup_steps, decay_steps)
    poly_lr = (
        end_lr + (schedule_hparams.base_lr - end_lr) *
        (1 - decay_step / decay_steps) ** schedule_hparams.power)
    return jnp.where(step <= schedule_hparams.warmup_steps, warmup_lr, poly_lr)
  return step_fn


def compound_schedule(schedule_hparams, max_training_steps):
  """Creates a learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)

  Args:
    schedule_hparams: Relevant schedule_hparams are --- factors - a string with
      factors separated by '*' that defines the schedule. base_learning_rate -
      float, the starting constant for the lr schedule. warmup_steps - how many
      steps to warm up for in the warmup schedule. decay_factor - The amount to
      decay the learning rate by. steps_per_decay - How often to decay the
      learning rate. steps_per_cycle - Steps per cycle when using cosine decay.
    max_training_steps: This is ignored (needed to match API of other lr
      functions).

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  del max_training_steps
  factors = [n.strip() for n in schedule_hparams['factors'].split('*')]
  expected_keys = ['schedule', 'factors']
  if 'constant' in factors:
    expected_keys.append('base_lr')
  if 'linear_warmup' in factors or 'rsqrt_decay' in factors:
    expected_keys.append('warmup_steps')
  _check_schedule_hparams(schedule_hparams, expected_keys)

  def lr_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= schedule_hparams['base_lr']
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / schedule_hparams['warmup_steps'])
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, schedule_hparams['warmup_steps']))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return lr_fn


def prepend_linear_warmup(schedule_hparams, max_training_steps,
                          base_lr_schedule):
  """Models the base_lr_schedule to include a warmup phase.

  The returned schedule will have the following form:

  if step < hps.warmup_steps:
     lr = (step / warmup_steps) ** warmup_power * base_lr

  otherwise:
    lr = base_lr_schedule(step - hps.warmup_steps)
    where the max train steps input to base_lr_schedule is
    max_train_steps - hps.warmup_steps.

  Effectively, what this does is the first warmup_steps will be linear warmup
  (if power =1), followed by what the base_lr_schedule would be if called with
  max_train_steps - warmup_steps.

  Args:
    schedule_hparams: Must include all required hparams needed in
      base_lr_schedule. Additionally we require warmup_steps, warmup_power to
      be added.
    max_training_steps: Full number of steps to be used in training.
    base_lr_schedule: One of the schedule functions defined in this module.
      Must satisfy the API of -
      base_lr_schedule(schedule_hparams, max_training_steps) -> returns lr_fn.

  Returns:
    A function mapping global_step to learning rate.
  """

  # grab warmup hparams
  schedule_hparams = dict(schedule_hparams)  # convert to dict so we can pop
  warmup_steps = schedule_hparams.pop('warmup_steps')
  warmup_power = schedule_hparams.pop('warmup_power')
  base_lr = schedule_hparams['base_lr']

  base_lr_fn = base_lr_schedule(schedule_hparams,
                                max_training_steps - warmup_steps)

  def lr_fn(t):
    if t < warmup_steps:
      return ((t / warmup_steps) ** warmup_power) * base_lr
    step = t - warmup_steps
    return base_lr_fn(step)

  return lr_fn


def warmup_then_piecewise_constant_schedule(schedule_hparams,
                                            max_training_steps):
  return prepend_linear_warmup(schedule_hparams, max_training_steps,
                               piecewise_constant_schedule)


lr_fn_dict = {
    'constant': constant_schedule,
    'cosine': cosine_schedule,
    'piecewise_linear': piecewise_linear_schedule,
    'piecewise_constant': piecewise_constant_schedule,
    'polynomial': polynomial_schedule,
    'mlperf_polynomial': mlperf_polynomial_schedule,
    'compound': compound_schedule,
    'warmup_then_piecewise_constant': warmup_then_piecewise_constant_schedule
}


def schedule_stretcher(schedule_fn, stretch_factor):
  """Stretch a schedule and return the stretched schedule fn."""
  if stretch_factor == 1 or stretch_factor is None:
    return schedule_fn

  def _schedule(global_step):
    return schedule_fn(global_step // stretch_factor)

  return _schedule


def get_schedule_fn(schedule_hparams, max_training_steps, stretch_factor=1):
  """Retrieve a schedule function."""
  warmup_suffix = '_warmup'
  if schedule_hparams['schedule'][-len(warmup_suffix):] == warmup_suffix:
    base_name = schedule_hparams['schedule'][:-len(warmup_suffix)]
    base_lr_schedule = lr_fn_dict[base_name]
    schedule_fn = prepend_linear_warmup(
        schedule_hparams, max_training_steps, base_lr_schedule)
  else:
    schedule_fn = lr_fn_dict[schedule_hparams['schedule']](
        schedule_hparams, max_training_steps)
  return schedule_stretcher(schedule_fn, stretch_factor)
