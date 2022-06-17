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

"""Code for evaluating the hessian and gradient covariance of i2w models."""
import functools
import itertools

from absl import logging
import flax
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from init2winit.dataset_lib import data_utils
from init2winit.model_lib import partition_tree
import jax
import jax.numpy as jnp
import numpy as np
import optax
from spectral_density import hessian_computation
from spectral_density import lanczos
from spectral_density.hessian_computation import ravel_pytree


DEFAULT_EVAL_CONFIG = {
    'average_hosts': True,
    'compute_interps': False,
    'compute_stats': False,
    'eval_gradient_covariance': False,
    'eval_hess_grad_overlap': True,
    'eval_hessian': True,
    'hparam_overrides': {},
    'max_eig_num_steps': 30,
    'name': 'hessian',
    'num_batches': 10,
    'num_eigens': 5,
    'num_eval_draws': 3,
    'num_lanczos_steps': 60,
    'rng_key': 0,
    'update_stats': False,
    'use_training_gen': True,
    'block_hessian': False,
    'param_partition_fn': None  # only used with block_hessian=True
}


def block_hessians(params,
                   loss_fn,
                   param_partition_fn,
                   batches_gen,
                   rng_key,
                   num_lanczos_steps=20):
  """Computes the loss hessian with respect to subsets of model parameters.

  Subsets are determined by the param_partition_fn, which maps the flattened
  model parameters to a dict mapping keys to subset of the flattened model
  parameters. For example, if the flattened param tree is

  {('a', 'b'): 1.0,
  ('a', 'c'): 1.0,
  ('d', 'b'): 2.0}

  And we partition on the outer key (see partition_tree.outer_key), then
  the output is
  {'a': {('a', 'b'): 1.0, ('a', 'c'): 1.0}
  'd': {('d', 'b'): 2.0}}

  Args:
    params: Replicated pytree of model parameters.
    loss_fn: non pmapped function with API
       loss_fn(unreplicated_params, unreplicated_batch) -> scalar loss.
    param_partition_fn: Maps a flattened pytree to a partitioned dict
       as described above.
    batches_gen: Should yield replicated batch so we can call
      jax.pmap(loss_fn)(params, batch).
    rng_key: Unreplicated jax PRNGKey.
    num_lanczos_steps: How many lanczos iterates to do.

  Returns:
    A dictionary of results, with a key for each partition of
      the model parameters (as determined by param_partition_fn). The key
      maps to another dict with the following key value pairs:
      max_eig_hess -> The hessian max eigenvalue with respect to the sub params.
      tridiag_hess -> The tridiagonal matrix output by lanczos.
      param_names -> The flattened parameter names in this partitian.
  """
  unrep_params = flax.jax_utils.unreplicate(params)
  flat_dict = flax.traverse_util.flatten_dict(unrep_params)

  sub_param_groups = param_partition_fn(flat_dict)

  sub_results = {}

  # I believe this will basically store a copy of the unreplicated
  # parameters in the function definition, which may be costly when
  # len(sub_param_groups) is large? Unclear how garbage colllection will handle
  # this in jax.
  def sub_loss(sub_params, batch_rng):
    new_dict = flat_dict.copy()
    for tup in sub_params:
      new_dict[tup] = sub_params[tup]

    new_params = flax.traverse_util.unflatten_dict(new_dict)
    return loss_fn(new_params, batch_rng)

  for key in sub_param_groups:

    logging.info('Block Hessian eval on %s', key)
    sub_params = unfreeze(jax_utils.replicate(sub_param_groups[key]))

    hvp_fn, _, n_params = hessian_computation.get_hvp_fn(
        sub_loss, sub_params, batches_gen, use_pmap=True)

    # This was needed to avoid the lint [cell-var-from-loop] error. Not sure
    # it's needed but to avoid any python weirdness with defining functions in
    # loop went ahead and implemented this.
    hvp_cl = functools.partial(hvp_fn, unfreeze(sub_params))
    row = {}

    row['tridiag_hess'], _ = lanczos.lanczos_np(
        hvp_cl,
        n_params,
        num_lanczos_steps,
        0,
        rng_key,
        verbose=True)
    evs = np.linalg.eigvalsh(row['tridiag_hess'])
    row['max_eig_hess'] = np.max(evs)
    row['param_names'] = [list(sub_param_groups[key].keys())]

    # The flattened keys are tuples, which doesn't work with the serialization,
    # so to save this dict the keys need to be strings.
    sub_results[str(key)] = row

  return sub_results


def prepare_batches_gen(dataset, eval_config):
  """Returns a data iterator.

  The API for the data iterator will be
    for b in batches_gen():
      pass
  We yield the same "epoch" every time to the data iterator is called.

  Args:
    dataset: An init2winit.dataset_lib.Dataset object. This is ignored if
      eval_config['use_training_gen'] == False.
    eval_config: A dict specifying the parameters for the hessian eval.

  Returns:
    A data generator.
  """
  train_iter = itertools.islice(dataset.train_iterator_fn(), 0,
                                eval_config['num_batches'])
  batches = list(train_iter)
  init_rng = jax.random.PRNGKey(eval_config['rng_key'])
  init_rng = jax.random.fold_in(init_rng, jax.process_index())
  def training_batches_gen():
    for counter, batch in enumerate(batches):
      batch = data_utils.shard(batch)
      rng = jax.random.fold_in(init_rng, counter)
      rng = jax_utils.replicate(rng)
      yield (batch, rng)
  return training_batches_gen


def _tree_sum(tree_left, tree_right):
  """Computes tree_left + tree_right."""
  def f(x, y):
    return x + y
  return jax.tree_util.tree_multimap(f, tree_left, tree_right)


def _tree_sub(tree_left, tree_right):
  """Computes tree_left - tree_right."""
  def f(x, y):
    return x - y
  return jax.tree_util.tree_multimap(f, tree_left, tree_right)


def _tree_zeros_like(tree):
  def f(x):
    return jnp.zeros_like(x)
  return jax.tree_util.tree_map(f, tree)


def _additive_update(params, update):
  """Computes an updated model for interpolation studies."""
  def f(x, y):
    return x + y
  new_params = jax.tree_util.tree_multimap(f, params, update)
  return new_params


def _unreplicate(sharded_array):
  temp = jax.tree_map(lambda x: x[0], sharded_array)
  return jax.device_get(temp)


def _compute_update(
    curr_params, optimizer_state, gradient, optimizer_update_fn):
  model_updates, _ = optimizer_update_fn(
      gradient, optimizer_state, params=curr_params)
  new_params = optax.apply_updates(curr_params, model_updates)
  diff = _tree_sub(curr_params, new_params)
  return diff


def _tree_normalize(tree):
  """Normalizes a pytree by its l_2 norm."""
  norm_vec = jax.tree_map(lambda x: np.linalg.norm(x) ** 2, tree)
  norm_vec, _ = jax.tree_util.tree_flatten(norm_vec)
  norm = np.sqrt(np.sum(norm_vec))
  norm_fn = lambda x: x / norm
  return jax.tree_map(norm_fn, tree)


def precondition_mvp_fn(mvp_fn, p_diag):
  """Matrix-vector product with a diagonal preconditioner.

  Given (1) a function that computes matrix-vector products between some [n x n]
  matrix "M" and arbitrary vectors, and (2) an [n] vector "p_diag", create a
  function that computes matrix-vector products between the matrix
      P^{-1/2} M P^{-1/2}  where P = diag(p_diag)
  and arbitrary vectors.

  Args:
    mvp_fn: (function) a function: ndarray -> ndarray that computes the matrix
      vector product of some matrix M with an arbitrary vector.
    p_diag: (ndarray) an ndarray

  Returns:
    a function that computes the matrix-vector product of P^{-1} M, where
      P = diag(p_diag), with arbitrary vectors.
  """
  p_diag_sqrt = np.sqrt(p_diag)
  def preconditioned_mvp_fn(v):
    return np.divide(mvp_fn(np.divide(v, p_diag_sqrt)), p_diag_sqrt)
  return preconditioned_mvp_fn


# TODO(gilmer): Rewrite API to accept loss as a function of params, batch
class CurvatureEvaluator:
  """Class for evaluating 2nd-order curvature stats of a model's loss surface.

  This class has three main methods:

  1- evaluate_stats: This method computes various statistics of the loss
  curvature. These statistics are divided in two groups: statistics computed
  along mini-batch / full-batch gradient directions and statistics computed
  along the optimizer's update direction. These statistics are returned in a
  dictionary with the following keys / values:
    step: Training global_step corresponding to the checkpoint.
    overlapi: The inner-product between the i_th mini-batch gradient direction
               (g_i) and the (full-batch) gradient, g. The full-batch gradient,
               g, is computed by averaging mini-batch gradients over a series of
               batches as specified by eval_config[num_batches].
    quadi: The quadratic form g_i^T H g_i, where H is the Hessian of the loss.
    normi: The squared norm: ||g_i||_2^2.
    quad_noisei: The quadratic form (g_i - g)^T H (g_i - g). This is used for
                estimating Tr(CH).
    overlap_u: The inner-product between the optimizer's update direction coming
                from the i_th mini-batch, u_i, and the full-batch gradient g.
    quad_u: The quadratic form u_i^T H u_i.
    norm_u: The squared norm: ||u_i||_2^2.
    fb_overlapi: The inner-product between u_i and the full-batch update
                direction coming from the optimizer.
    hTg: A numpy matrix corresponding to the inner-products of Hessian
         eigenvectors and g_i / ||g_i||.
    hTu: A numpy matrix corresponding to the inner-products of Hessian
         eigenvectors and u_i / ||u_i||.
    cTg: A numpy matrix corresponding to the inner-products of grad covariance
         eigenvectors and g_i / ||g_i||.
    cTu: A numpy matrix corresponding to the inner-products of grad covariance
         eigenvectors and u_i / ||u_i||.

  2- compute_interpolations: This method computes linear interpolations of the
  loss across the (normalized) directions passed in lists gdirs and udirs
  (corresponding to respectively the gradients and the optimizer's update
  directions). The method returns a dictionary with the following key / values:
    step_size: A numpy array of size (num_points,) corresponding to the step
               sizes in the interpolation.
    lossi: Numpy array of size (num_points,) representing the interpolation
           of the loss in direction g_i, L(theta + eta g_i / ||g_i||).
    loss_ui: Numpy array of size (num_points,) representing the interpolation of
           the loss in direction u_i, L(theta + eta u_i / ||u_i||).
    loss_hveci: A numpy array of size (num_points,) corresponding to the linear
           interpolation of the loss in direction of the ith Hessian eigenvalue.
    loss_cveci: A numpy array of size (num_points,) corresponding to the linear
           interpolation of the loss in direction of the ith Covariance
           eigenvalue.

  In the dictionaries above, the index i varies from 0 to num_eval_draws.
  num_eval_draws corresponds to the number of independent mini-batches to be
  studied. Indices 0 to num_eval_draws - 1 correspond to mini-batch quantities
  while index i = num_eval_draws corresponds to the full-batch directions.

  3- compute_spectrum: This method uses Lanczos iterations to compute detailed
  measurements of the eigenvalue density of the full Hessian and the gradient
  covariance matrix. Depending on the provided config file, Hessian spectrum,
  gradient covariance spectrum, or both are computed.

  The exact output is toggled from the provided eval_config with following keys:
    num_eval_draws: The number of mini-batch updates studied (int > 0).
    num_points: The number of points used for linear interpolation of the loss.
    update_stats: Whether to compute curvature statistics for the update
                  directions produced by the optimizer.
    num_lanczos_steps: The number of Lanczos steps used for computing the
                  spectra of the Hessian and the gradient covariance.
  """

  def __init__(self,
               params,
               eval_config,
               loss,
               dataset=None,
               batches_gen=None):
    """Creates the CurvatureEvaluator object.

    Args:
      params: A pytree of model parameters.
      eval_config: See DEFAULT_EVAL_CONFIG as an example.
      loss: Any function which satisfies the API loss(params, batch).
      dataset: An init2winit.dataset_lib.datasets object. Optional if
        batches_gen is supplied.
      batches_gen: Any function which yields batches to be fed into the loss.
        Optional if dataset is supplied. API must satisfy
        for batch in batches_gen():
           batch_loss = loss(params, batch)
    """
    if batches_gen is None and dataset is None:
      raise ValueError('Either a dataset or a batches generator must be given'
                       'when constructing the CurvatureEvaluator')
    self.eval_config = eval_config
    if eval_config['num_batches'] < eval_config['num_eval_draws']:
      raise ValueError('Number of draws is larger than number of batches.')

    if batches_gen:
      self.batches_gen = batches_gen
    else:
      self.batches_gen = prepare_batches_gen(dataset, eval_config)

    def avg_loss(params, batch, loss_fn):
      loss_val = loss_fn(params, batch)
      loss_val = jax.lax.pmean(loss_val, axis_name='batch')
      return loss_val

    def avg_grad(params, batch, loss_fn):
      grad_loss = jax.grad(loss_fn)(params, batch)
      grad_loss = jax.lax.pmean(grad_loss, axis_name='batch')
      return grad_loss

    self.grad_loss = jax.pmap(functools.partial(avg_grad, loss_fn=loss),
                              axis_name='batch')
    self.avg_loss = functools.partial(avg_loss, loss_fn=loss)
    self.p_avg_loss = jax.pmap(
        functools.partial(avg_loss, loss_fn=loss), axis_name='batch')
    self.hvp_fn, self.unravel, self.n_params = hessian_computation.get_hvp_fn(
        loss, params, self.batches_gen, use_pmap=True)
    self.update_model = jax.pmap(_additive_update)
    self.gvp_fn, _, n_params = hessian_computation.get_gradient_covariance_vp_fn(
        loss,
        params,
        self.batches_gen,
        use_pmap=True,
        average_hosts=eval_config['average_hosts'])
    assert self.n_params == n_params
    if jax.process_index() == 0:
      logging.info('CurvatureEvaluator build with config: %r', eval_config)
      logging.info('n_params: %d', self.n_params)

  def evaluate_stats(self, params, gdirs, udirs, hvex, cvex, step):
    """Compute the curvature statistics.

    Args:
      params: The current model parameters. Must be already replicated to
        accommodate pmapping.
      gdirs: List of pytrees corresponding to the mini/full batch gradients.
      udirs: List of pytrees corresponding to the optimizer's update directions.
      hvex: List of Hessian Eigenvectors represented as 1D Numpy arrays.
      cvex: List of Covariance Eigenvectors represented as 1D Numpy arrays.
      step: A scalar corresponding to the optimization step.
    Returns:
      row: A dictionary corresponding to the computed statistics.
    """
    row = {'step': step}
    if not self.eval_config['compute_stats']:
      return row
    if not gdirs:
      raise ValueError('Full-batch gradient is necessary!')
    # Returns a 1D vector. If needed, v would be replicated inside the function.
    hvp_cl = lambda v: self.hvp_fn(params, v)
    fullbatch_grad, _ = ravel_pytree(gdirs[-1])
    for i, dir_dict in enumerate(gdirs):
      update, _ = ravel_pytree(dir_dict)
      row['overlap%d' % (i,)] = np.sum(np.multiply(update, fullbatch_grad))
      temp = hvp_cl(update)
      row['quad%d' % (i,)] = np.sum(np.multiply(update, temp))
      row['norm%d' % (i,)] = np.linalg.norm(update) ** 2
      # The mini-batch noise quadratic
      noise = update - fullbatch_grad
      temp = hvp_cl(noise)
      row['quad_noise%d' % (i,)] = np.sum(np.multiply(noise, temp))

    if udirs:
      fullbatch_update, _ = ravel_pytree(udirs[-1])
      for i, dir_dict in enumerate(udirs):
        update, _ = ravel_pytree(dir_dict)
        row['overlap_u%d' % (i,)] = np.sum(np.multiply(update, fullbatch_grad))
        temp = hvp_cl(update)
        row['quad_u%d' % (i,)] = np.sum(np.multiply(update, temp))
        row['norm_u%d' % (i,)] = np.linalg.norm(update) ** 2
        # The inner-product between full-batch and mini-batch updates
        row['fb_overlap%d' % (i,)] = np.sum(np.multiply(update,
                                                        fullbatch_update))
    if hvex:
      inner_prod_g = np.zeros((len(hvex), len(gdirs)))
      inner_prod_u = np.zeros((len(hvex), len(udirs)))
      for i, vec in enumerate(hvex):
        for j, g in enumerate(gdirs):
          g, _ = ravel_pytree(g)
          g = g / np.linalg.norm(g)
          inner_prod_g[i, j] = np.sum(np.multiply(vec, g))
        for j, u in enumerate(udirs):
          u, _ = ravel_pytree(u)
          u = u / np.linalg.norm(u)
          inner_prod_u[i, j] = np.sum(np.multiply(vec, u))
      row['hTg'] = np.copy(inner_prod_g)
      row['hTu'] = np.copy(inner_prod_u)

    if cvex:
      inner_prod_g = np.zeros((len(cvex), len(gdirs)))
      inner_prod_u = np.zeros((len(cvex), len(udirs)))
      for i, vec in enumerate(cvex):
        for j, g in enumerate(gdirs):
          g, _ = ravel_pytree(g)
          g = g / np.linalg.norm(g)
          inner_prod_g[i, j] = np.sum(np.multiply(vec, g))
        for j, u in enumerate(udirs):
          u, _ = ravel_pytree(u)
          u = u / np.linalg.norm(u)
          inner_prod_u[i, j] = np.sum(np.multiply(vec, u))
      row['cTg'] = np.copy(inner_prod_g)
      row['cTu'] = np.copy(inner_prod_u)

    if jax.process_index() == 0:
      logging.info('stats eval finished. Statistics captured:')
      logging.info(row.keys())
    return row

  def compute_dirs(self, params, optimizer_state, optimizer_update_fn):
    """Compute the directions of curvature evaluation.

    The function computes:
      1- mini-batch gradients and full-batch gradients and adds them to gdirs.
      2- mini-batch and full-batch update directions and adds them to udirs.
    Args:
      params: The current model parameters. Must be already replicated to
        accommodate pmapping.
      optimizer_state: The optax optimizer state used for the training. The
        state has to be already replicated to accommodate pmapping.
      optimizer_update_fn: The optax optimizer update function. We assume that
        the learning rate is a constant 1.0.
    Returns:
      gdirs, udirs: two lists of pytrees described above.
    """
    gdirs = []
    udirs = []
    compiled_tree_sum = jax.pmap(_tree_sum)
    compiled_tree_zeros_like = jax.pmap(_tree_zeros_like)
    update_fn = functools.partial(
        _compute_update, optimizer_update_fn=optimizer_update_fn)
    compiled_update = jax.pmap(update_fn)
    count = 0.0
    full_grad = compiled_tree_zeros_like(params)
    for batch in self.batches_gen():
      # Already pmeaned minibatch gradients
      avg_grad = self.grad_loss(params, batch)
      if count < self.eval_config['num_eval_draws']:
        gdirs.append(_unreplicate(avg_grad))
        minibatch_update = compiled_update(
            params, optimizer_state, avg_grad)
        udirs.append(_unreplicate(minibatch_update))
      count += 1.0
      full_grad = compiled_tree_sum(full_grad, avg_grad)
    if count == 0:
      raise ValueError('Provided generator did not yield any data.')
    # Compute the full-batch counterparts
    full_grad = jax.tree_map(lambda x: x / count, full_grad)
    gdirs.append(_unreplicate(full_grad))
    full_batch_update = compiled_update(
        params, optimizer_state, full_grad)
    udirs.append(_unreplicate(full_batch_update))

    if jax.process_index() == 0:
      logging.info('Update directions successfully computed')
    return gdirs, udirs

  def compute_interpolations(self, params, gdirs, udirs, hvex, cvex, step):
    """Compute the linear interpolation along directions of gdirs or udirs."""
    row = {'step': step}
    if not self.eval_config['compute_interps']:
      return row
    lower = self.eval_config['lower_thresh']
    upper = self.eval_config['upper_thresh']
    num_points = self.eval_config['num_points']
    etas = np.linspace(lower, upper, num=num_points, endpoint=True)
    row = {'step_size': etas}
    for i, u_dir in enumerate(gdirs):
      u_dir = _tree_normalize(u_dir)
      loss_values = np.zeros(shape=(num_points,))
      for j in range(num_points):
        eta = etas[j]
        loss_values[j] = self._full_batch_eval(params, u_dir, eta)
      row['loss%d' % (i,)] = np.copy(loss_values)
    if jax.process_index() == 0:
      logging.info('Loss interpolation along gradients finished.')

    for i, u_dir in enumerate(udirs):
      u_dir = _tree_normalize(u_dir)
      loss_values = np.zeros(shape=(num_points,))
      for j in range(num_points):
        eta = etas[j]
        loss_values[j] = self._full_batch_eval(params, u_dir, eta)
      row['loss_u%d' % (i,)] = np.copy(loss_values)
    if jax.process_index() == 0:
      logging.info('Loss interpolation along optimizer directions finished.')

    _, unflatten = ravel_pytree(gdirs[0])
    for i, u_dir in enumerate(hvex):
      loss_values = np.zeros(shape=(num_points,))
      u_dir = unflatten(u_dir)
      for j in range(num_points):
        eta = etas[j]
        loss_values[j] = self._full_batch_eval(params, u_dir, eta)
      row['loss_hvec%d' % (i,)] = np.copy(loss_values)

    for i, u_dir in enumerate(cvex):
      loss_values = np.zeros(shape=(num_points,))
      u_dir = unflatten(u_dir)
      for j in range(num_points):
        eta = etas[j]
        loss_values[j] = self._full_batch_eval(params, u_dir, eta)
      row['loss_cvec%d' % (i,)] = np.copy(loss_values)

    if jax.process_index() == 0:
      logging.info('Loss interpolations finished. Statistics captured:')
      logging.info(row.keys())
    return row

  def _full_batch_eval(self, params, update_dir, eta):
    """Compute the full-batch loss at theta = theta_0 + eta * update_dir.

    The function assumes that self.p_avg_loss is returning the loss averaged
    over the batch.
    Args:
      params: The current model parameters. Must be already replicated to
        accommodate pmapping.
      update_dir: A pytree corresponding to the update direction.
      eta: A scalar corresponding to the step-size.
    Returns:
      A scalar corresponding to the average full-batch loss.
    """
    update_dir = jax.tree_map(lambda x: x * eta, update_dir)
    update_dir = jax_utils.replicate(update_dir)
    new_params = self.update_model(params, update_dir)

    count = 0.0
    total_loss = 0.0
    for batch in self.batches_gen():
      sharded_loss = self.p_avg_loss(new_params, batch)
      avg_loss = _unreplicate(sharded_loss)
      count += 1.0
      total_loss += jax.device_get(avg_loss)
    if count == 0:
      raise ValueError('Provided generator did not yield any data.')
    return total_loss / count

  def _full_batch_grad(self, params):
    """Compute full batch gradient."""
    compiled_tree_sum = jax.pmap(_tree_sum)
    compiled_tree_zeros_like = jax.pmap(_tree_zeros_like)

    count = 0.0
    full_grad = compiled_tree_zeros_like(params)
    for batch in self.batches_gen():
      # Already pmeaned minibatch gradients
      avg_grad = self.grad_loss(params, batch)
      if count < self.eval_config['num_eval_draws']:
        count += 1.0
        full_grad = compiled_tree_sum(full_grad, avg_grad)
    full_grad = jax.tree_map(lambda x: x / count, full_grad)
    full_grad = _unreplicate(full_grad)
    return full_grad

  def evaluate_spectrum(self, params, step, diag_preconditioner=None):
    """Estimate the eigenspectrum of H and C.

    Args:
      params: (replicated pytree) the parameters at which to compute
        the hessian spectrum.
      step: (int) the global step of training.
      diag_preconditioner: (optional unreplicated pytree) if not None, we'll
        compute the spectrum of P^{-1} H and P^{-1} C,
        where P = diag(diag_preconditioner). Our implementation exploits the
        fact that P^{-1} M shares eigenvalues with the "similar" matrix
        P^(-1/2) M P^(-1/2), which is symmetric, and hence can be
        eigendecomposed using Lanczos.

    Returns:
      row
      hess_evecs
      cov_evecs
    """
    # Number of upper and lower eigenvectors to be approximated.
    num_evs = self.eval_config['num_eigens']
    hess_evecs = []
    cov_evecs = []

    if 2 * num_evs >= self.eval_config['num_lanczos_steps']:
      raise ValueError('Too many eigenvectors requested!')
    hvp_cl = lambda v: self.hvp_fn(params, v)
    gvp_cl = lambda v: self.gvp_fn(params, v)

    if diag_preconditioner is not None:
      diag_p = hessian_computation.ravel_pytree(diag_preconditioner)[0]
      hvp_cl = precondition_mvp_fn(hvp_cl, diag_p)
      gvp_cl = precondition_mvp_fn(gvp_cl, diag_p)

    key = jax.random.PRNGKey(0)

    row = {'step': step}

    # Do we want .get here to keep old configs working?
    if self.eval_config.get('block_hessian'):
      param_partition_fn = partition_tree.get_param_partition_fn(
          self.eval_config['param_partition_fn'])
      block_results = block_hessians(params, self.avg_loss, param_partition_fn,
                                     self.batches_gen, key)
      row['block_hessian'] = block_results

    if self.eval_config['eval_hessian']:
      row['tridiag_hess'], hess_evecs = lanczos.lanczos_np(
          hvp_cl,
          self.n_params,
          self.eval_config['num_lanczos_steps'],
          num_evs,
          key,
          verbose=True)
      evs = np.linalg.eigvalsh(row['tridiag_hess'])
      row['max_eig_hess'] = np.max(evs)

      # We assume you run more than 1 step.
      row['max_eig_hess_ratio'] = evs[-1] / evs[-2]
      row['pos_neg_ratio'] = evs[0] / evs[-1]

      if num_evs > 0:
        # Compute the breakdown of the max eigenvector across variables in the
        # model (shown promise in localizing issues causing large curvature).
        max_evec = self.unravel(hess_evecs[-1])
        row['max_evec_decomp'] = jax.tree_map(lambda x: jnp.linalg.norm(x)**2,
                                              max_evec)

    if self.eval_config['eval_hess_grad_overlap']:
      # compute gradient.
      full_grad = self._full_batch_grad(params)
      grad_flat, _ = hessian_computation.ravel_pytree(full_grad)

      row['tridiag_hess_grad_overlap'], _ = lanczos.lanczos_np(
          hvp_cl,
          self.n_params,
          self.eval_config['num_lanczos_steps'],
          num_evs,
          key,
          init_vec=grad_flat,
          verbose=True)

    if self.eval_config['eval_gradient_covariance']:
      row['tridiag_cov'], cov_evecs = lanczos.lanczos_np(
          gvp_cl,
          self.n_params,
          self.eval_config['num_lanczos_steps'],
          num_evs,
          key,
          verbose=True)
      evs = np.linalg.eigvalsh(row['tridiag_cov'])
      row['max_eig_cov'] = np.max(evs)
    return row, hess_evecs, cov_evecs
