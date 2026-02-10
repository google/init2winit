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

"""Noise schedules for Masked Diffusion Language Models (MDLM).

Each schedule defines alpha_t (noise level) as a function of t in [0, 1],
where alpha_0 ~ 1 (clean) and alpha_1 ~ 0 (fully masked).
"""

import jax.numpy as jnp


def log_linear_alpha(t):
  return 1.0 - t


def log_linear_alpha_derivative(t):
  del t
  return -1.0


def cosine_alpha(t):
  return jnp.cos(jnp.pi * t / 2.0)


def cosine_alpha_derivative(t):
  return -jnp.pi / 2.0 * jnp.sin(jnp.pi * t / 2.0)


def geometric_alpha(t):
  return (1.0 - t) ** 2


def geometric_alpha_derivative(t):
  return -2.0 * (1.0 - t)


_ALL_SCHEDULES = {
    'log_linear': (log_linear_alpha, log_linear_alpha_derivative),
    'cosine': (cosine_alpha, cosine_alpha_derivative),
    'geometric': (geometric_alpha, geometric_alpha_derivative),
}


def get_schedule(name):
  try:
    return _ALL_SCHEDULES[name]
  except KeyError:
    raise ValueError(f'Unknown noise schedule: {name}') from None
