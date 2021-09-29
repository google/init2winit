# coding=utf-8
# Copyright 2021 The init2winit Authors.
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

# Lint as: python3
r"""NQM Model.

"""
from flax import nn
from flax.nn import initializers
from init2winit.model_lib import base_model
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
from scipy.stats import ortho_group

# small hparams used for unit tests
DEFAULT_HPARAMS = config_dict.ConfigDict(dict(
    optimizer='momentum',
    opt_hparams={
        'momentum': 0.0,
    },
    lr_hparams={
        'base_lr': 0.1,
        'schedule': 'constant'
    },
    batch_size=128,
    rng_seed=-1,
    # Note the dimension is set by input_shape.
    hessian_decay_power=1,
    noise_decay_power=1,
    nqm_mode='diagH_diagC',
    model_dtype='float32',
    l2_decay_factor=None,
))


class NQMLoss(nn.Module):
  """Loss for the NQM model.

  We assume the input is isotropic Gaussian noise, this is used to sample
  noise with covariance matrix C = N^T N with eigenspectrum defined by the
  params. To sample Gaussian noise with covariance C, we first sample
  z ~ N(0, I), then define eps = Nz, where N^T N = C. N is passed to apply as
  the noise_scaling matrix.
  """

  def apply(self,
            noise_input,
            hessian,
            noise_scaling,
            train=True):
    """Returns the NQM loss.

    Args:
      noise_input: Sample from a d-dimensional isotropic Gaussian (this will be
        scaled by the noise_scaling matrix).
      hessian: dxd psd matrix representing the loss hessian.
      noise_scaling: dxd matrix used to scale noise_input. This is a matrix N
        s.t. N^T N = C, where C will be the covariance of the scaled noise.
      train: Ignored, we add this to conform to the i2w model API. noise_input =
        jnp.asarray(noise_input) x = self.param('x', (noise_input.shape[-1],),
        initializers.ones)

    Returns: The resulting scalar valued loss.
    """
    noise_eps = jnp.asarray(noise_input)
    x = self.param('x', (noise_eps.shape[-1],), initializers.ones)

    # NQM loss = 1/2 x^T hessian x + x.T noise_scaling noise_input
    # this gives grad loss = hessian x + eps, where eps ~ N(0, C)
    return jnp.dot(jnp.dot(x.T, hessian), x) / 2 + jnp.mean(
        jnp.dot(jnp.dot(noise_input, noise_scaling), x))


def quadratic_form(u, sigma):
  return np.dot(np.dot(u.T, sigma), u)


def _get_nqm_matrices(dim,
                      hessian_decay_power=1.0,
                      noise_decay_power=1.0,
                      mode='diagH_noC'):
  """Returns a hessian and a noise scaling matrix used in the NQM.

  The corresponding loss will be equal to  1/2 x^T H x + eps sqrtC x.

  Where eps ~ N(0, I). This leads to grad_loss = Hx + sqrtC eps  --- where
  sqrtC eps is noise with covariance = C. The hessian_decay_power will configure
  the eigenvalue values of the matrix H to be = 1 / i^power, i=1,...,d.
  Similarly noise_decay_power will configure the eigenvalues of C. Note, this
  function does not return the matrix C directly, instead we return a matrix
  sqrtC, with the property that sqrtC^T sqrtC = C. To sample noise with
  covariance C, you first sample z ~ N(0, I) then return sqrtC z.

  This function supports various configurations to explore whether or not H and
  C are diagonal themselves, or are codiaginalizable.
  This is configured via the mode as follows:

  diagH_noC: H is diagonal C = 0
  diagH_diagC: H and C both diagonal
  H_noC: H = U^T H_eigs U for random orthonormal U.
  H_codiagC: H = U^T H_eigs U  C = U^T C_eigs U for random orthonormal U.
  H_offdiagC: H = U^T H_eigs U  C = V^T C_eigs V for random orthonormal U, V.
  diagH_offdiagC: H is diagonal  C = V^T C_eigs V for random orthonormal V.

  NOTE: Currently this function uses the python internal random seed, so output
    may be non-deterministic.

  Args:
    dim: dimension of the matrices to generate.
    hessian_decay_power: Hessian eigenvalues will be of the form 1 / i^power.
    noise_decay_power : Noise eigenvalues will be of the form 1 / i^power.
    mode: One of the modes listed above.

  Returns:
    hessian_matrix, noise_matrix (aka sqrtC). Noise matrix has form
      sqrtC^T sqrtC = C.
  """
  hessian_eigs = np.array(
      [1.0 / np.power(i, hessian_decay_power) for i in range(1, dim + 1)])
  hessian_eigs = np.diag(hessian_eigs)
  noise_scaling_eigs = np.array(
      [1.0 / np.power(i, noise_decay_power / 2.0) for i in range(1, dim + 1)])

  noise_scaling_eigs = np.diag(noise_scaling_eigs)

  if mode == 'diagH_noC':
    return hessian_eigs, np.zeros_like(noise_scaling_eigs)

  elif mode == 'diagH_diagC':
    return hessian_eigs, noise_scaling_eigs

  ortho_matrix = ortho_group.rvs(dim=dim)  # H = U^T Sigma U
  if mode == 'H_noC':
    return (quadratic_form(ortho_matrix, hessian_eigs),
            np.zeros_like(noise_scaling_eigs))

  elif mode == 'H_codiagC':
    # noise matrix = noise_scaling_eigs U
    return (quadratic_form(ortho_matrix, hessian_eigs),
            np.dot(noise_scaling_eigs, ortho_matrix))

  elif mode == 'H_offdiagC':
    # Sample a new rotation matrix for the noise
    c_ortho_matrix = ortho_group.rvs(dim=dim)
    return (quadratic_form(ortho_matrix, hessian_eigs),
            np.dot(noise_scaling_eigs, c_ortho_matrix))

  elif mode == 'diagH_offdiagC':
    # Sample a new rotation matrix for the noise
    c_ortho_matrix = ortho_group.rvs(dim=dim)
    return hessian_eigs, np.dot(noise_scaling_eigs, c_ortho_matrix)
  else:
    raise ValueError('Invalid mode: {}'.format(mode))


class NQM(base_model.BaseModel):
  """Defines the loss and eval for the NQM model.

  NOTE: This model is only meant to be used with nqm_noise.py dataset, which
  generates isotropic Gaussian noise.
  """

  def __init__(self, hps, dataset_meta_data, loss_name, metrics_name):
    del loss_name

    # This is ignored, but is needed to satisfy the initializer API.
    self.loss_fn = None
    self.metrics_name = metrics_name

    self.hps = hps
    self.dataset_meta_data = dataset_meta_data
    self.flax_module_def = self.build_flax_module()

  def evaluate_batch(self, flax_module, batch_stats, batch):
    """Evals the NQM loss."""

    metrics = {
        # Trainer eval assumes eval function sums, not averages.
        'loss': flax_module(batch['inputs']) * batch['inputs'].shape[0],
        'denominator': batch['inputs'].shape[0]
    }
    return metrics

  def training_cost(self, flax_module, batch_stats, batch, dropout_rng):
    """Returns the NQM loss.

    L = noise_input^T H x / 2 + x^T sqrtC eps. Where x is the model params
    H is the hessian, and sqrtC is the noise matrix satisfying
    sqrtC^T sqrtC = C.

    Args:
      flax_module: Must be NQMLoss class.
      batch_stats: ignored.
      batch: Assumes 'inputs' is isotropic noise.
      dropout_rng: ignored.

    Returns:
      loss, (batch_stats)
    """

    average_loss = flax_module(batch['inputs'])

    return average_loss, (batch_stats)

  def build_flax_module(self):
    hessian, noise_scaling = _get_nqm_matrices(self.hps.input_shape[0],
                                               self.hps.hessian_decay_power,
                                               self.hps.noise_decay_power,
                                               self.hps.nqm_mode)
    return NQMLoss.partial(hessian=hessian, noise_scaling=noise_scaling)
