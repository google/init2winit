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

"""Paterson-Stockmeyer method for polynomial evaluation."""
from typing import Any, Callable, List, Sequence, TypeVar

import numpy as np

T = TypeVar('T')


def _powers(x: T, n: int, product: Callable[[T, T], T]) -> List[T]:
  """Returns the list [x, x², ..., xⁿ]."""
  xp = [None] * (n + 1)
  xp[1] = x
  for j in range(2, n + 1):
    # To reduce round-off, compute xʲ as the result of O(log j) mutliplies
    xp[j] = product(xp[j // 2], xp[(j + 1) // 2])
  return xp[1:]


def polynomial_no_constant(a: Sequence[Any], x: T, product: Callable[[T, T],
                                                                     T]) -> T:
  """Paterson-Stockmeyer evaluation of a[0] x + a[1] x² + ... + a[n-1] xⁿ.

  A variant of the Paterson-Stockmeyer method for polynomial evaluation
  presented in [2], which avoids using the multiplicative identity (x⁰). The
  algorithm uses only ⌈2√n⌉ - 2 multiplications instead of n - 1, making it
  especially suitable when multiplications are expensive, e.g., for matrices.
  The reduced number of multiplications is accomplished by grouping the terms as

          (a[0]  x +    a[1] x² + ... + a[ s-1] xˢ) +
     xˢ   (a[s]  x +  a[s+1] x² + ... + a[2s-1] xˢ) +
    (xˢ)² (a[2s] x + a[2s+1] x² + ... + a[3s-1] xˢ) +
            ...

  with s = ⌈√n⌉. The powers up to xˢ are precomputed with s - 1 multiplications,
  allowing all the (at most) degree s polynomials in parentheses above to be
  evaluated. These are then combined using Horner's rule with ⌈n/s⌉ - 1
  subsequent multiplications.

  [1] Michael S. Paterson and Larry J. Stockmeyer, "On the number of nonscalar
    multiplications necessary to evaluate polynomials," SIAM J. Comput., 2
    (1973), pp. 60–66.

  [2] M. Fasi, "Optimality of the Paterson-Stockmeyer method for evaluating
    matrix polynomials and rational matrix functions," Linear Algebra Appl.,
    574 (2019), pp. 182–200.

  Args:
    a: Polynomial coefficients. a[j] is the coefficient of xʲ⁺¹.
    x: Argument to evaluate the polynomial at.
    product: Multiplication function.

  Returns:
    The polynomial a[0] x + a[1] x² + ... + a[n-1] xⁿ .

  Raises:
    ValueError if `a` is empty.
  """
  n = len(a)
  if n == 0:
    raise ValueError('polynomial_no_constant: coefficients empty.')
  s = int(np.ceil(np.sqrt(n)))
  xp = _powers(x, s, product)
  inner = lambda alpha: sum([cj * xj for (cj, xj) in zip(alpha, xp)])
  inner_poly = lambda i: inner(a[s * i:min(n, s * (i + 1))])
  i = (n + s - 1) // s - 1
  y = inner_poly(i)
  for i in reversed(range(i)):
    y = inner_poly(i) + product(xp[s - 1], y)
  return y
