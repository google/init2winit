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

"""Defines different learning_rate schedules."""

import numpy as np


def _check_schedule_hparams(schedule_hparams, expected_keys):
  if set(schedule_hparams.keys()) != set(expected_keys):
    raise ValueError(
        'Provided schedule_hparams keys are invalid. Recieved: {}, Expected: {}'
        .format(sorted(schedule_hparams.keys()), sorted(expected_keys)))


def constant_schedule(schedule_hparams, max_training_updates):
  del max_training_updates
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr'])
  return lambda t: schedule_hparams['base_lr']


def cosine_schedule(schedule_hparams, max_training_updates):
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr'])

  def lr_fn(t):
    decay_factor = (1 + np.cos(t / max_training_updates * np.pi)) * 0.5
    return schedule_hparams['base_lr'] * decay_factor
  return lr_fn


# code based on
# https://github.com/tensorflow/lingvo/blob/master/lingvo/core/schedule.py#L305
def transformer_schedule(schedule_hparams, max_training_updates):
  """Computes a reverse sqrt style decay schedule scaled by sqrt of model's encoder dimension.

  lr = base_lr * min((step + 1) / sqrt(warmup_steps**3) , 1/sqrt(step + 1)) *
  (1/sqrt(enocder_dim))
  Args:
    schedule_hparams: Relevant hparams are base_lr, encoder_dim, warmup_steps.
    max_training_updates: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_updates
  _check_schedule_hparams(
      schedule_hparams, ['schedule', 'warmup_steps', 'base_lr', 'encoder_dim'])

  def lr_fn(t):
    warmup_steps = schedule_hparams['warmup_steps']
    model_dim = schedule_hparams['encoder_dim']
    decay_factor = model_dim**-0.5 * np.minimum((t + 1) * warmup_steps**-1.5,
                                                (t + 1)**-0.5)

    return schedule_hparams['base_lr'] * decay_factor

  return lr_fn


def rsqrt_normalized_decay(schedule_hparams, max_training_updates):
  """Computes a "squashed" reverse sqrt decay schedule.

  lr = base_lr * sqrt(squash_steps) / sqrt(step + squash_steps)

  Args:
    schedule_hparams: Relevant hparams are base_lr, squash_steps.
    max_training_updates: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_updates
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr', 'squash_steps'])

  def lr_fn(t):
    squash_steps = schedule_hparams['squash_steps']
    return (schedule_hparams['base_lr'] *
            np.sqrt(squash_steps)) / np.sqrt(t + squash_steps)

  return lr_fn


def t2t_rsqrt_normalized_decay(schedule_hparams, max_training_updates):
  """Computes rsqrt_normalized_decay according to T2T implementation.

  It's an implemetation of T2T learning_rate_schedule function
  with 'constant*rsqrt_normalized_decay' schedule.

  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/learning_rate.py

  lr = base_lr * (sqrt(defer_steps) / sqrt(max(step, defer_steps))

  Used in training of a local attention transformer on PG19 dataset.

  Args:
    schedule_hparams: Relevant hparams are base_lr, defer_steps.
    max_training_updates: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_updates
  _check_schedule_hparams(schedule_hparams,
                          ['schedule', 'base_lr', 'defer_steps'])

  def lr_fn(step):
    scale = np.sqrt(schedule_hparams['defer_steps'])
    scale /= np.sqrt(np.maximum(step, schedule_hparams['defer_steps'])).astype(
        np.float32)
    return schedule_hparams['base_lr'] * scale

  return lr_fn


def polynomial_schedule(schedule_hparams, max_training_updates):
  """Same behavior as tf.train.polynomial_decay.

  Supports either decay_steps or decay_steps_factor, but providing both is an
  error.

  Args:
    schedule_hparams: Relevant hparams are schedule,
      base_lr, end_factor, power, and one of decay_steps or
      decay_steps_factor.
    max_training_updates: Only used when decay_steps_factor is provided.

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
        max_training_updates * schedule_hparams['decay_steps_factor'])
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


def piecewise_constant_schedule(schedule_hparams, max_training_updates):
  """Computes a piecewise constant decay schedule.

  Note that each element of decay_factors is absolute (not relative). For
  example, to decay the learning rate to 0.5 of its initial value after
  100 steps, followed by 0.1 of its *initial value* after 200 steps,
  with a plateau of 0.1 of its initial value thereafter, use
  decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Args:
    schedule_hparams: Relevant hparams are base_lr, decay_events
      decay_factors.
    max_training_updates: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_updates
  _check_schedule_hparams(schedule_hparams, [
      'schedule', 'base_lr', 'decay_events',
      'decay_factors'
  ])
  boundaries = np.array([0] + schedule_hparams['decay_events'])
  factors = [1.0] + schedule_hparams['decay_factors']
  def lr_fn(t):
    index = np.sum(boundaries[1:] < t)
    return factors[index] * schedule_hparams['base_lr']

  return lr_fn


def piecewise_linear_schedule(schedule_hparams, max_training_updates):
  """Computes a piecewise linear decay schedule.

  Note that each element of decay_factors is absolute (not relative). For
  example, to decay the learning rate linearly to 0.5 of its initial value after
  100 steps, followed by linearly to 0.1 of its *initial value* after 200 steps,
  with a plateau of 0.1 of its initial value thereafter, use
  decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Args:
    schedule_hparams: Relevant hparams are base_lr, decay_events
      decay_factors.
    max_training_updates: This is ignored (needed to match API of other lr
      functions).

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  del max_training_updates
  _check_schedule_hparams(schedule_hparams, [
      'schedule', 'base_lr', 'decay_events',
      'decay_factors'
  ])
  boundaries = np.array([0] + schedule_hparams['decay_events'])
  factors = [1.0] + schedule_hparams['decay_factors']
  def lr_fn(t):
    index = np.sum(boundaries[1:] < t)
    if index+1 == len(factors):
      return factors[index] * schedule_hparams['base_lr']
    m = (factors[index + 1] - factors[index]) / (
        boundaries[index + 1] - boundaries[index])
    interpolated_factor = m * (t - boundaries[index]) + factors[index]
    return schedule_hparams['base_lr'] * interpolated_factor

  return lr_fn


# TODO(gilmer) Change the code path before open source.
def mlperf_polynomial_schedule(schedule_hparams, max_training_updates):
  """Polynomial learning rate schedule for LARS optimizer.

  This function is copied from
  third_party/tensorflow_models/mlperf/models/rough/resnet_jax/train.py and
  modified to fit the init2winit API.

  Args:
    schedule_hparams: Relevant hparams are base_lr, warmup_steps.
    max_training_updates: Used to calculate the number of decay steps.

  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  _check_schedule_hparams(
      schedule_hparams,
      ['schedule', 'base_lr', 'warmup_steps', 'power', 'start_lr', 'end_lr',
       'decay_end', 'warmup_power'])
  decay_steps = max_training_updates - schedule_hparams.warmup_steps + 1
  end_lr = schedule_hparams['end_lr']
  def step_fn(step):
    decay_end = schedule_hparams['decay_end']
    if decay_end > 0 and step >= decay_end:
      step = decay_end
    r = (step / schedule_hparams.warmup_steps) ** schedule_hparams.warmup_power
    warmup_lr = (
        schedule_hparams.base_lr * r + (1 - r) * schedule_hparams.start_lr)
    decay_step = np.minimum(step - schedule_hparams.warmup_steps, decay_steps)
    poly_lr = (
        end_lr + (schedule_hparams.base_lr - end_lr) *
        (1 - decay_step / decay_steps) ** schedule_hparams.power)
    return np.where(step <= schedule_hparams.warmup_steps, warmup_lr, poly_lr)
  return step_fn


def compound_schedule(schedule_hparams, max_training_updates):
  """Creates a learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps),
  * cosine: multiply by the cosine schedule.

  Args:
    schedule_hparams: Relevant schedule_hparams are --- factors - a string with
      factors separated by '*' that defines the schedule. base_learning_rate -
      float, the starting constant for the lr schedule. warmup_steps - how many
      steps to warm up for in the warmup schedule. decay_factor - The amount to
      decay the learning rate by. steps_per_decay - How often to decay the
      learning rate. steps_per_cycle - Steps per cycle when using cosine decay.
    max_training_updates: Full number of model updates to be used in training.
      Only used when 'cosine' factor is requested.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
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
        ret *= np.minimum(1.0, step / schedule_hparams['warmup_steps'])
      elif name == 'rsqrt_decay':
        ret /= np.sqrt(np.maximum(step, schedule_hparams['warmup_steps']))
      elif name == 'cosine':
        warmup_steps = schedule_hparams.get('warmup_steps', 0)
        shift = np.maximum(0, step - warmup_steps)
        cosine_steps = max_training_updates - warmup_steps - 1
        ret *= (1 + np.cos(shift / cosine_steps * np.pi)) * 0.5
      else:
        raise ValueError('Unknown factor %s.' % name)
    return np.asarray(ret, dtype=np.float32)

  return lr_fn


def prepend_polynomial_warmup(schedule_hparams, max_training_updates,
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
  max_train_steps - warmup_steps. The default value for warmup_power is 1
  meaning linear warmup

  Args:
    schedule_hparams: Must include all required hparams needed in
      base_lr_schedule. Additionally we require warmup_steps, warmup_power to
      be added.
    max_training_updates: Full number of model updates to be used in training.
    base_lr_schedule: One of the schedule functions defined in this module.
      Must satisfy the API of -
      base_lr_schedule(schedule_hparams, max_training_updates) -> returns lr_fn.

  Returns:
    A function mapping global_step to learning rate.
  """

  # grab warmup hparams
  schedule_hparams = dict(schedule_hparams)  # convert to dict so we can pop
  warmup_steps = schedule_hparams.pop('warmup_steps')
  warmup_power = schedule_hparams.pop('warmup_power', 1)
  base_lr = schedule_hparams['base_lr']

  base_lr_fn = base_lr_schedule(schedule_hparams,
                                max_training_updates - warmup_steps)

  def lr_fn(t):
    if t < warmup_steps:
      return ((t / warmup_steps) ** warmup_power) * base_lr
    step = t - warmup_steps
    return base_lr_fn(step)

  return lr_fn


def warmup_then_piecewise_constant_schedule(schedule_hparams,
                                            max_training_updates):
  return prepend_polynomial_warmup(schedule_hparams, max_training_updates,
                                   piecewise_constant_schedule)


lr_fn_dict = {
    'constant': constant_schedule,
    'cosine': cosine_schedule,
    'piecewise_linear': piecewise_linear_schedule,
    'piecewise_constant': piecewise_constant_schedule,
    'polynomial': polynomial_schedule,
    'mlperf_polynomial': mlperf_polynomial_schedule,
    'compound': compound_schedule,
    'warmup_then_piecewise_constant': warmup_then_piecewise_constant_schedule,
    'rsqrt_normalized_decay': rsqrt_normalized_decay,
    't2t_rsqrt_normalized_decay': t2t_rsqrt_normalized_decay,
    'transformer_schedule': transformer_schedule,
}


def schedule_stretcher(schedule_fn, stretch_factor):
  """Stretch a schedule and return the stretched schedule fn."""
  if stretch_factor == 1 or stretch_factor is None:
    return schedule_fn

  def _schedule(global_step):
    return schedule_fn(global_step // stretch_factor)

  return _schedule


def add_decay_lr_on_plateau(base_schedule_fn, decay_lr_on_plateau_hparams):
  """LR schedule for automatically reducing learning rate as necessary."""

  class DecayLROnPlateau:
    """LR schedule for automatically reducing learning rate as necessary."""

    def __init__(self, base_schedule_fn, decay_lr_on_plateau_hparams):
      self._base_schedule_fn = base_schedule_fn
      self._decay_lr_on_plateau_hparams = dict(decay_lr_on_plateau_hparams)

      # Grab decay params.
      self._patience = self._decay_lr_on_plateau_hparams['plateau_patience']
      self._plateau_decay_factor = self._decay_lr_on_plateau_hparams[
          'plateau_decay_factor']
      self._metric_name = self._decay_lr_on_plateau_hparams[
          'plateau_metric_name']
      self._mode = self._decay_lr_on_plateau_hparams['plateau_mode']

      self._lr_multiplier = 1.0
      self._counter = 0
      self._best_metric_val = None

    def __call__(self, step):
      return self._lr_multiplier * self._base_schedule_fn(step)

    def decay(self, metrics):
      """Decay learning rate if validation metric has plateaued."""
      metric_val = metrics[self._metric_name]

      if not self._best_metric_val:
        self._best_metric_val = metric_val
        self._counter = 0
        return

      if (self._mode == 'lower' and metric_val < self._best_metric_val) or (
          self._mode == 'higher' and metric_val > self._best_metric_val):
        self._best_metric_val = metric_val
        self._counter = 0
      else:
        self._counter += 1

      if self._counter == self._patience:
        self._lr_multiplier /= self._plateau_decay_factor
        self._counter = 0

      return

  return DecayLROnPlateau(base_schedule_fn, decay_lr_on_plateau_hparams)


# Note that everything inside schedules.py should be in terms of number of model
# updates, not gradient calculations.
def get_schedule_fn(schedule_hparams, max_training_updates, stretch_factor=1):
  """Retrieve a schedule function."""

  plateau_suffix = '_decay_lr_on_plateau'
  decay_lr_on_plateau_hparams = None
  if schedule_hparams['schedule'][-len(plateau_suffix):] == plateau_suffix:
    schedule_hparams['schedule'] = schedule_hparams[
        'schedule'][:-len(plateau_suffix)]

    # Each schedule expects schedule_hparams to have a specific set of keys. Pop
    # decay_lr_on_plateau-specific hparams and pass them directly to the decay
    # lr schedule.
    decay_lr_on_plateau_hparams = {}
    schedule_hparams = dict(schedule_hparams)
    for key in list(schedule_hparams.keys()):  # pylint: disable=g-builtin-op
      if 'plateau_' in key:
        decay_lr_on_plateau_hparams[key] = schedule_hparams.pop(key)

  warmup_suffix = '_warmup'
  if schedule_hparams['schedule'][-len(warmup_suffix):] == warmup_suffix:
    base_name = schedule_hparams['schedule'][:-len(warmup_suffix)]
    base_lr_schedule = lr_fn_dict[base_name]
    schedule_fn = prepend_polynomial_warmup(
        schedule_hparams, max_training_updates, base_lr_schedule)
  else:
    schedule_fn = lr_fn_dict[schedule_hparams['schedule']](
        schedule_hparams, max_training_updates)

  schedule_fn = schedule_stretcher(schedule_fn, stretch_factor)

  if decay_lr_on_plateau_hparams:
    return add_decay_lr_on_plateau(schedule_fn, decay_lr_on_plateau_hparams)
  else:
    return schedule_fn
