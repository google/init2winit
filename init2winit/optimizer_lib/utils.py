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

"""Optimizer utilities."""
import functools
import inspect
from typing import Callable
from typing import Iterable
from typing import Union

import optax


def static_inject_hyperparams(
    inner_factory: Callable[..., optax.GradientTransformation],
    injectable_args: Union[str, Iterable[str]] = ('learning_rate',)
) -> Callable[..., optax.GradientTransformation]:
  """Wrapper for `optax.inject_hyperparams` making all args static by default.

  This wrapper resolves two issues:

  1. If anyone adds an optional argument to an `optax` optimizer, code
     will break because `optax.inject_hyperparams` will pass 0.0.
  2. Optimizers like `adafactor` have arguments that are not boolean, but are
     used in boolean statements, which leads to ConcretizationTypeErrors.

  Args:
    inner_factory: a function that returns the inner
      ``optax.GradientTransformation`` given the hyperparameters.
    injectable_args: a string or iterable of strings specifying which callable
      parameters **are** schedules.

  Returns:
    A callable that returns a ``optax.GradientTransformation``. This callable
    accepts the same arguments as ``inner_factory`` and you may provide
    schedules for the args listed in `injectable_args`.
  """

  injectable_args = ({injectable_args} if isinstance(injectable_args, str) else
                     set(injectable_args))
  inner_signature = inspect.signature(inner_factory)

  @functools.wraps(inner_factory)
  def wrapped_transform(*args, **kwargs) -> optax.GradientTransformation:
    bound_arguments = inner_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    static_args = set(bound_arguments.arguments.keys()) - injectable_args

    return optax.inject_hyperparams(inner_factory, static_args)(*args, **kwargs)

  return wrapped_transform


def extract_field(state, field_name):
  """Extract a field from a nested tuple (especially an optax optimizer state).

  Suppose that we'd like to extract Adam's "nu" pytree from an optax optimizer
  state that consists of a ScaleByAdam namedtuple wrapped inside of a
  combine.chain() tuple wrapped inside of an InjectHyperparamsState namedtuple
  wrapped inside of a GradientAccumulatorState namedtuple.  This function
  can do that.

  Args:
    state: An optax optimizer state.  This should be a nested tuple, meaning a
      tuple (potentially a namedtuple) that contains other tuples (or
      potentially namedtuples) in its slots.  Note that the state can contain
      non-tuple values as well (e.g. one of the slots in InjectHyperparamsState
      is a dict), but these will be ignored.
    field_name: (str) The name of the field we'd like to extract from the nested
      tuple.  For example, "nu" to extract Adam's second-moment accumulator.

  Returns:
    The value of a field with the given field name.  If there is more than
      one field with this name in the nested tuple, the behavior of this
      function is undefined.  Returns None if there is no field with the
      given name in "state".
  """
  assert isinstance(state, tuple)

  # If "state" is a namedtuple containing a field with the right name, return
  # the value in that field.
  if hasattr(state, '_fields') and field_name in state._fields:
    return getattr(state, field_name)

  # Else, recursively call this function on the slots of the tuple "state".
  for element in state:
    if isinstance(element, tuple):
      field = extract_field(element, field_name)
      if field is not None:
        return field

  # If we didn't find anything, return None.
  return None


def no_cross_device_gradient_aggregation(
    update_fn: optax.TransformUpdateFn,
) -> optax.TransformUpdateFn:
  """Decorator for signaling that the optimizer requires access to the device-level mean gradients.

  Standard Optax optimizers assume that the `update` argument passed to the
  `update_fn` function has already been aggregated across devices, however, in
  some cases the per-device gradient is useful (e.g., in order to calculate
  batch statistics or to avoid the synchronization point). This decorator is
  used to signal the trainer not to perform any gradient aggregation before
  calling the optimizer's update function. It will then be the responsibility of
  the optimizer to aggregate the updates before returning the updated values.

  Note that to get the true per-example gradients you would need to use the
  unnormalized loss functions (e.g., weighted_unnormalized_cross_entropy) and
  then instead of jax.value_and_grad you'd need to use one of the JAX jacobian
  functions

  Args:
    update_fn: An optax update transformation function.

  Returns:
    The same callable, with an additional attribute that signals that no
    aggregation should be performed.
  """
  setattr(update_fn, 'init2winit_requires_gradient_aggregation', False)
  return update_fn


def requires_gradient_aggregation(
    update_fn: optax.TransformUpdateFn,
) -> bool:
  """Returns whether the given update_fn requires the gradient to be aggregated.

  See no_cross_device_gradient_aggregation() above for additional details.

  Args:
    update_fn: An Optax update transformation function.

  Returns:
    True if the gradient should be aggregated across devices before invoking the
    update function, False otherwise.
  """
  return getattr(update_fn, 'init2winit_requires_gradient_aggregation', True)
