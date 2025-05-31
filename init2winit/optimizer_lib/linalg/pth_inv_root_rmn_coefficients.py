# coding=utf-8
# Copyright 2025 The init2winit Authors.
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

"""Coefficients for pth inverse root iteration.

  The faster schedule allows for a faster convergence of the coupled
  iteration. The faster schedule is a R_{1,1} approximation to x^{1/p}. For
  matrices with condition number < 1e+7, this converges in 3 Cholesky steps.
  The slower schedule is a mixed R_{2,2} andR_{3,3} approximation to x^{1/p}.
  It takes a total of 5 serial Cholesky steps or 2 parallel Cholesky steps.
"""

from jax import numpy as jnp


def r11_schedule(p: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns the coefficients of the R_{m,n} minimax rational approximation.

  The coefficients in this function are hand computed through a minimax
  optimization using Mathematica. I have precomputed the coefficients of the
  rational polynomials for p = 1, 2, 3, 4, 6, and 8 such that it is accurate in
  the interval [1e-5, 1] (within the precision of bfloat16).

  The fixed point iteration with this approximation converges in 3 steps.
  However, I have to perform 2 Cholesky and solves in each step. The
  Cholesky computation is completely independent, so they are parallelizable,
  but it appears we are unable to take advantage of this on the current
  hardware.

  Note: If the parallelization is possible, a R_{3,3} approximation might
  be better because it converges in 2 steps (with 3 parallelizable choleskys
  in each step).

  Args:
    p: Exponent

  Returns:
    Coefficients of the R_{2,2} minimax rational approximation to x^{1/p}.
  """
  pth_inv_root_coeffs = {
      "1": {
          "a": jnp.array([
              jnp.array([
                  0.00728429,
              ]),
              jnp.array([
                  0.562921,
              ]),
              jnp.array([
                  2.47914,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000013169,
              ]),
              jnp.array([
                  0.0469985,
              ]),
              jnp.array([
                  0.309659,
              ]),
          ]),
          "c": jnp.array([
              0.9927684811086588,
              0.650342340923137,
              0.34566570659013346,
          ]),
          "alpha": jnp.array([
              0.00031,
          ]),
      },
      "2": {
          "a": jnp.array([
              jnp.array([
                  0.00728429,
              ]),
              jnp.array([
                  0.562921,
              ]),
              jnp.array([
                  2.47914,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000013169,
              ]),
              jnp.array([
                  0.0469985,
              ]),
              jnp.array([
                  0.309659,
              ]),
          ]),
          "c": jnp.array([
              0.9927684811086588,
              0.650342340923137,
              0.34566570659013346,
          ]),
          "alpha": jnp.array([
              0.00031,
          ]),
      },
      "3": {
          "a": jnp.array([
              jnp.array([
                  0.00223321,
              ]),
              jnp.array([
                  0.235877,
              ]),
              jnp.array([
                  1.31091,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.0000422318,
              ]),
              jnp.array([
                  0.0515935,
              ]),
              jnp.array([
                  0.436073,
              ]),
          ]),
          "c": jnp.array([
              0.9977720423285437,
              0.816790180627646,
              0.5227820022651496,
          ]),
          "alpha": jnp.array([
              0.004580446299383034,
          ]),
      },
      "4": {
          "a": jnp.array([
              jnp.array([
                  0.00127078,
              ]),
              jnp.array([
                  0.1504,
              ]),
              jnp.array([
                  0.906412,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.0000715897,
              ]),
              jnp.array([
                  0.0567652,
              ]),
              jnp.array([
                  0.508511,
              ]),
          ]),
          "c": jnp.array([
              0.9987309217849512,
              0.8754103928382763,
              0.6246621843282536,
          ]),
          "alpha": jnp.array([
              0.01760681686165901,
          ])
      },
      "6": {
          "a": jnp.array([
              jnp.array([
                  0.000693758,
              ]),
              jnp.array([
                  0.0892148,
              ]),
              jnp.array([
                  0.569847,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000118336,
              ]),
              jnp.array([
                  0.0646857,
              ]),
              jnp.array([
                  0.591824,
              ]),
          ]),
          "c": jnp.array([
              0.9993068094793026,
              0.9226841705328986,
              0.7363859445928534,
          ]),
          "alpha": jnp.array([
              0.06767899452107008,
          ])
      },
      "8": {
          "a": jnp.array([
              jnp.array([
                  0.000484871,
              ]),
              jnp.array([
                  0.0642155,
              ]),
              jnp.array([
                  0.418634,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000151106,
              ]),
              jnp.array([
                  0.0699718,
              ]),
              jnp.array([
                  0.639096,
              ]),
          ]),
          "c": jnp.array([
              0.9995154369091881,
              0.9433819538731253,
              0.7965555193155961,
          ]),
          "alpha": jnp.array([
              0.13269068114098673,
          ])
      },
  }

  assert p in [1, 2, 3, 4, 6, 8], f"Unsupported exponent: {p}"

  indexstr = str(p)
  return (
      pth_inv_root_coeffs[indexstr]["a"],
      pth_inv_root_coeffs[indexstr]["b"],
      pth_inv_root_coeffs[indexstr]["c"],
  )


def r12_schedule(p: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns the coefficients of the R_{m,n} minimax rational approximation.

  The coefficients in this function are hand computed through a minimax
  optimization using Mathematica. I have precomputed the coefficients of the
  rational polynomials for p = 1, 2, 3, 4, 6, and 8 such that it is accurate in
  the interval [1e-5, 1] (within the precision of bfloat16).

  The fixed point iteration with this approximation converges in 3 steps.
  However, I have to perform 2 Cholesky and solves in each step. The
  Cholesky computation is completely independent, so they are parallelizable,
  but it appears we are unable to take advantage of this on the current
  hardware.

  Note: If the parallelization is possible, a R_{3,3} approximation might
  be better because it converges in 2 steps (with 3 parallelizable choleskys
  in each step).

  Args:
    p: Exponent

  Returns:
    Coefficients of the R_{1,2} minimax rational approximation to x^{1/p}.
  """
  pth_inv_root_coeffs = {
      "1": {
          "a": jnp.array([
              jnp.array([
                  0.00728429,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.412371,
                  1.45199,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.0000422318,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0132896,
                  0.359132,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9927684811086588,
              0.40399412424033093,
          ]),
          "alpha": jnp.array([
              0.00031,
          ]),
      },
      "2": {
          "a": jnp.array([
              jnp.array([
                  0.00728429,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.412371,
                  1.45199,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000013169,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0132896,
                  0.359132,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9927684811086588,
              0.40399412424033093,
          ]),
          "alpha": jnp.array([
              0.00031,
          ]),
      },
      "3": {
          "a": jnp.array([
              jnp.array([
                  0.00223321,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.100571,
                  0.736674,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000013169,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0128156,
                  0.34967,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9977720423285437,
              0.6078597971798018,
          ]),
          "alpha": jnp.array([
              0.004580446299383034,
          ]),
      },
      "4": {
          "a": jnp.array([
              jnp.array([
                  0.00127078,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.0482585,
                  0.502309,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.0000715897,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0131364,
                  0.363824,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9987309217849512,
              0.7062434406893869,
          ]),
          "alpha": jnp.array([
              0.01760681686165901,
          ])
      },
      "6": {
          "a": jnp.array([
              jnp.array([
                  0.000693758,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.0213567,
                  0.313303,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000118336,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0139331,
                  0.391343,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9993068094793026,
              0.8024113703961219,
          ]),
          "alpha": jnp.array([
              0.06767899452107008,
          ])
      },
      "8": {
          "a": jnp.array([
              jnp.array([
                  0.000484871,
                  0.0,
                  0.0,
              ]),
              jnp.array([
                  0.0132363,
                  0.229835,
                  0.0,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  0.000151106,
                  1.0,
                  1.0,
              ]),
              jnp.array([
                  0.0145446,
                  0.410795,
                  1.0,
              ]),
          ]),
          "c": jnp.array([
              0.9995154369091881,
              0.8503702052561087,
          ]),
          "alpha": jnp.array([
              0.13269068114098673,
          ])
      },
  }

  assert p in [1, 2, 3, 4, 6, 8], f"Unsupported exponent: {p}"

  indexstr = str(p)
  return (
      pth_inv_root_coeffs[indexstr]["a"],
      pth_inv_root_coeffs[indexstr]["b"],
      pth_inv_root_coeffs[indexstr]["c"],
  )


def r23_schedule(p: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Returns the coefficients of the R_{2,2} and R_{3,3} minimax rational approximation.

  The coefficients in this function are hand computed through a minimax
  optimization using Mathematica. I have precomputed the coefficients of the
  rational polynomials for p = 2, 3, 4, 6, and 8 such that it is accurate in
  the interval [1e-8, 1].

  The fixed point iteration with this approximation converges in 3 steps.
  However, I have to perform 2 Cholesky and solves in each step. The
  Cholesky computation is completely independent, so they are parallelizable,
  but it appears we are unable to take advantage of this on the current
  hardware.

  Note: If the parallelization is possible, a R_{3,3} approximation might
  be better because it converges in 2 steps (with 3 parallelizable choleskys
  in each step).

  Args:
    p: Exponent

  Returns:
    Coefficients of the R_{2,2} and R_{3,3} minimax rational approximation
    to x^{1/p}.
  """
  pth_inv_root_coeffs = {
      "1": {
          "a": jnp.array([
              jnp.array([
                  0.00210496,
                  0.0908775,
                  0.0,
              ]),
              jnp.array([
                  1.16853,
                  1.87484,
                  6.15624,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  1.01161e-6,
                  0.00206034,
                  1.0,
              ]),
              jnp.array([
                  0.0276374,
                  0.347561,
                  2.44542,
              ]),
          ]),
          "c": jnp.array([
              0.9150842419756393,
              0.18814049506324307,
          ]),
          "alpha": jnp.array([
              0.00031,
          ])
      },
      "2": {
          "a": jnp.array([
              jnp.array([
                  0.00210496,
                  0.0908775,
                  0.0,
              ]),
              jnp.array([
                  1.16853,
                  1.87484,
                  6.15624,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  1.01161e-6,
                  0.00206034,
                  1.0,
              ]),
              jnp.array([
                  0.0276374,
                  0.347561,
                  2.44542,
              ]),
          ]),
          "c": jnp.array([
              0.9150842419756393,
              0.18814049506324307,
          ]),
          "alpha": jnp.array([
              0.00031,
          ])
      },
      "3": {
          "a": jnp.array([
              jnp.array([
                  0.000289179,
                  0.0403451,
                  0.0,
              ]),
              jnp.array([
                  0.28355,
                  0.72082,
                  3.60415,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  2.16943e-6,
                  0.0036741,
                  1.0,
              ]),
              jnp.array([
                  0.0338279,
                  0.353954,
                  2.61132,
              ]),
          ]),
          "c": jnp.array([
              0.9610887682064332,
              0.35654864483028464,
          ]),
          "alpha": jnp.array([
              0.00458,
          ])
      },
      "4": {
          "a": jnp.array([
              jnp.array([
                  0.000107688,
                  0.0264996,
                  0.0,
              ]),
              jnp.array([
                  0.133667,
                  0.418105,
                  2.60585,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  3.1798e-6,
                  0.00498007,
                  1.0,
              ]),
              jnp.array([
                  0.0374229,
                  0.365693,
                  2.77366,
              ]),
          ]),
          "c": jnp.array([
              0.9742069320862631,
              0.47047090372825334,
          ]),
          "alpha": jnp.array([
              0.0178,
          ])
      },
      "6": {
          "a": jnp.array([
              jnp.array([
                  0.0000362135,
                  0.0159419,
                  0.0,
              ]),
              jnp.array([
                  0.0568194,
                  0.21657,
                  1.69775,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  4.43759e-6,
                  0.00665465,
                  1.0,
              ]),
              jnp.array([
                  0.0413061,
                  0.381233,
                  2.98353,
              ]),
          ]),
          "c": jnp.array([
              0.9843752941682342,
              0.6106677320601622,
          ]),
          "alpha": jnp.array([
              0.06817314583299908,
          ])
      },
      "8": {
          "a": jnp.array([
              jnp.array([
                  0.0000195364,
                  0.0115332,
                  0.0,
              ]),
              jnp.array([
                  0.034241,
                  0.143596,
                  1.26729,
              ]),
          ]),
          "b": jnp.array([
              jnp.array([
                  5.21531e-6,
                  0.00770372,
                  1.0,
              ]),
              jnp.array([
                  0.0434886,
                  0.391305,
                  3.11437,
              ]),
          ]),
          "c": jnp.array([
              0.9886653900682543,
              0.6925022243284951,
          ]),
          "alpha": jnp.array([
              0.13341664064126335,
          ])
      },
  }

  assert p in [1, 2, 3, 4, 6, 8], f"Unsupported exponent: {p}"

  indexstr = str(p)
  return (
      pth_inv_root_coeffs[indexstr]["a"],
      pth_inv_root_coeffs[indexstr]["b"],
      pth_inv_root_coeffs[indexstr]["c"],
  )

