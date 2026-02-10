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

"""Optimizer utilities."""

import functools
import inspect
from typing import Callable, Iterable, Union

import jax.numpy as jnp
import optax


def static_inject_hyperparams(
    inner_factory: Callable[..., optax.GradientTransformation],
    injectable_args: Union[str, Iterable[str]] = ('learning_rate',),
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

  injectable_args = (
      {injectable_args}
      if isinstance(injectable_args, str)
      else set(injectable_args)
  )
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


def overwrite_hparam_names(
    base_opt: optax.GradientTransformationExtraArgs,
    **hparam_names_to_aliases: dict[str, str]
) -> optax.GradientTransformationExtraArgs:
  """Create aliases for hyperparams of an optimizer defined by inject_hyperparams.

  Enables access to some hyperparams through an alias. In particular, if an
  optimizer called its learning rate say 'lr' we can use this utility to create
  an alias 'learning_rate' for 'lr' such that injecting learning rate can be
  done through the 'learning_rate' key of the state (so all optimizers, even
  the ones defined outside init2winit can comply with the manual injection
  of learning rates).

  Example:
    >>> import optax
    >>> opt = optax.inject_hyperparams(optax.sgd)(learning_rate=0.)
    >>> opt = overwrite_hparam_names(opt, learning_rate='lr')
    >>> params = jnp.array([1., 2., 3.])
    >>> state = opt.init(params)
    >>> # We set the learning rate via lr
    >>> state = optax.tree_utils.tree_set(state, lr=0.5)
    >>> updates, state = opt.update(params, state)
    >>> # The resulting update used the learning rate set via lr
    >>> print(updates)
    [-0.5 -1.  -1.5]

  Args:
    base_opt: base ``optax.GradientTransformationExtraArgs`` to transform. Its
      state must be 'InjectHyperparamsState'-like such that a hyperparams
      attribute is present in its state.
    **hparam_names_to_aliases: keyword-value argument detailing how to replace
      hyperparameter names by new aliases.

  Returns:
    new_opt: new ``optax.GradientTransformationExtraArgs`` with updated
      hyperparams attribute in its state that can be fecthed and modified
      through the given aliases.

  .. warning::
    Changing manually the hyperparam with some of its old names won't have any
    effect anymore.
  """
  init_fn, update_fn = base_opt

  def new_init_fn(params):
    state = init_fn(params)
    for hparam_name, alias in hparam_names_to_aliases.items():
      state.hyperparams[alias] = state.hyperparams[hparam_name]
    return state

  def new_update_fn(updates, state, params=None, **extra_args):
    for hparam_name, alias in hparam_names_to_aliases.items():
      state.hyperparams[hparam_name] = state.hyperparams[alias]
      del state.hyperparams[alias]
    updates, state = update_fn(updates, state, params, **extra_args)
    for hparam_name, alias in hparam_names_to_aliases.items():
      state.hyperparams[alias] = state.hyperparams[hparam_name]
    return updates, state

  return optax.GradientTransformationExtraArgs(new_init_fn, new_update_fn)


def append_hparam_name(
    base_opt: optax.GradientTransformationExtraArgs,
    hparam_name: str,
    default_value: float = 0.,
) -> optax.GradientTransformationExtraArgs:
  """Create artificicial hparam name to comply with pipeline.

  Some optimizers may not have a ``learning_rate`` entry in their input
  arguments (this happens naturally for some learning-rate free optimizers).
  The init2winit pipeline requires the optimizer to have a ``learning_rate``
  entry. This utility adds such an entry in the hyperparams of an optimizer
  that won't affect the optimizer while making it fit the pipeline.

  Examples:
    >>> import optax
    >>> opt = optax.inject_hyperparams(optax.sgd)(learning_rate=0.5)
    >>> new_opt = utils.append_hparam_name(opt, 'foo')
    >>> optax.tree_utils.tree_set(state, foo=2.)
    >>> foo = optax.tree_utils.tree_get(state, 'foo')
    >>> print(foo)
    2.0

  Args:
    base_opt: base ``optax.GradientTransformationExtraArgs`` to transform. Its
      state must be 'InjectHyperparamsState'-like such that a hyperparams
      attribute is present in its state.
    hparam_name: hyperparameter name to add.
    default_value: default value for the new hyperparameter
      (never used, simply there to fill the entry)

  Returns:
    new_opt: new ``optax.GradientTransformationExtraArgs`` with new
      ``hparam_name`` in the keys of the ``hyperparams`` entry of its state.
  """
  init_fn, update_fn = base_opt

  def new_init_fn(params):
    state = init_fn(params)
    # Mimics the behavior of inject_hyperparams initialization
    # such that the parameter is converted to an array.
    state.hyperparams[hparam_name] = jnp.asarray(default_value)
    return state

  def new_update_fn(updates, state, params=None, **extra_args):
    del state.hyperparams[hparam_name]
    updates, state = update_fn(updates, state, params, **extra_args)
    state.hyperparams[hparam_name] = jnp.asarray(default_value)
    return updates, state

  return optax.GradientTransformationExtraArgs(new_init_fn, new_update_fn)
