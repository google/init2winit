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

"""Meta opt trainer for the init2winit project."""
import functools
import jax
import jax.numpy as jnp


def _roll(xs, shift=1):
  return xs[-shift:] + xs[:-shift]


# @functools.partial(jax.jit, static_argnames='history_length')
def init_pytree_history(pytree, history_length):
  # NOTE(dsuo): we can have different initialization functions besides replicate
  # (e.g., zeros).
  return jax.tree_map(
      lambda x: jnp.tile(
          (ex := jnp.expand_dims(x, axis=0)),
          (history_length,) + (1,) * (ex.ndim - 1),
      ),
      pytree,
  )


# @functools.partial(jax.jit, static_argnames='history_length')
def init_pytree_history_zeros(pytree, history_length):
  # NOTE(dsuo): jnp.zeros((history_length,) + x.shape)
  return jax.tree_map(
      lambda x: jnp.zeros(
          (history_length,) + jnp.expand_dims(x, axis=0).shape[1:],
          dtype=x.dtype,
      ),
      pytree,
  )


@functools.partial(jax.jit, static_argnames='index')
def update_pytree_history(pytree_history, pytree, index=0):
  return jax.tree_map(lambda x, y: x.at[index].set(y), pytree_history, pytree)


@jax.jit
def roll_and_update_leaf_history(leaf_history, leaf):
  leaf_history = jnp.roll(leaf_history, shift=-1, axis=0)
  return leaf_history.at[-1].set(leaf)


def roll_and_update_pytree_history(pytree_history, pytree):
  return jax.tree_map(roll_and_update_leaf_history, pytree_history, pytree)


# @functools.partial(jax.jit, static_argnames='shift')
def roll_pytree_history(pytree_history, shift=1):
  return jax.tree_map(
      lambda x: jnp.roll(x, shift=shift, axis=0), pytree_history
  )


# @functools.partial(jax.jit, static_argnames='index')
def get_pytree_history_index(pytree_history, index=0):
  return jax.tree_map(lambda x: x.at[index].get(), pytree_history)


# @functools.partial(jax.jit, static_argnames=('start_index', 'length'))
def get_pytree_history_window(pytree_history, start_index=0, length=1):
  return jax.tree_map(
      lambda x: jax.lax.dynamic_slice_in_dim(x, start_index, length, axis=0),
      pytree_history,
  )
