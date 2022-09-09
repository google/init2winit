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

"""Combine utilities."""
import functools
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import optax

# TODO(dsuo): Add back grafting combinator.


def join(by: Union[str, Callable[[optax.GradientTransformation, ...],
                                 optax.Updates]], *args,
         **kwargs) -> Callable[..., optax.GradientTransformation]:
  """Join multiple chains."""

  if by is None or by == 'chain':
    return lambda *args, **kwargs: optax.chain(*(args + tuple(kwargs.values())))

  if isinstance(by, str):
    if by not in combinator_registry:
      raise ValueError(f'Unrecognized `by` function {by}.')
    by = combinator_registry[by](*args, **kwargs)

  # TODO(dsuo): match docs/autocomplete with combinator args.
  def transform(*args, **kwargs):

    if by is None:
      return optax.chain(*(args + tuple(kwargs.values())))

    def init(params: optax.Params) -> optax.OptState:
      args_state = tuple(chain.init(params) for chain in args)
      kwargs_state = {
          name: chain.init(params) for name, chain in kwargs.items()
      }

      return args_state, kwargs_state

    def update(
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None
    ) -> Tuple[optax.Updates, optax.OptState]:
      args_state, kwargs_state = state

      args_results = [
          chain.update(updates, state, params)
          for chain, state in zip(args, args_state)
      ]
      args_updates = tuple(result[0] for result in args_results)
      args_state = tuple(result[1] for result in args_results)

      kwargs_results = {
          name: chain.update(updates, kwargs_state[name], params)
          for name, chain in kwargs.items()
      }
      kwargs_updates = {
          name: result[0] for name, result in kwargs_results.items()
      }
      kwargs_state = {
          name: result[1] for name, result in kwargs_results.items()
      }

      return by(*args_updates, **kwargs_updates), (args_state, kwargs_state)

    return optax.GradientTransformation(init, update)

  return transform


def _grafting_helper(chain, use_global_norm=False):
  norm = jax.tree_map(jnp.linalg.norm, chain)
  if use_global_norm:
    global_norm = jax.tree_util.tree_reduce(lambda x, y: jnp.sqrt(x**2 + y**2),
                                            norm)
    norm = jax.tree_map(lambda x: global_norm, norm)
  return norm


def combine_by_grafting(eps: float = 1e-6, use_global_norm: bool = False):
  """Grafting combinator.

  Args:
    eps (float, optional): term added to D normalization denominator for
      numerical stability (default: 1e-16)
    use_global_norm (bool, optional): graft global l2 norms rather than
      per-layer (default: False)

  Returns:
    updates in the shape of params.
  """

  def combinator(mag_chain, dir_chain):
    mag_norm = _grafting_helper(mag_chain, use_global_norm=use_global_norm)
    dir_norm = _grafting_helper(dir_chain, use_global_norm=use_global_norm)

    norm = jax.tree_map(lambda dir, dirn, magn: dir / (dirn + eps) * magn,
                        dir_chain, dir_norm, mag_norm)

    return norm

  return combinator


def combine_by_sum(*args, **kwargs):
  del args, kwargs

  def combinator(*args, **kwargs):
    args = args + tuple(kwargs.values())
    return functools.reduce(
        lambda x, y: jax.tree_multimap(lambda i, j: i + j, x, y), args)

  return combinator


combinator_registry = {
    'grafting': combine_by_grafting,
    'sum': combine_by_sum,
}
