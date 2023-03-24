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

"""Unit tests for CurvatureEvaluator class.

"""

import functools
import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from flax import jax_utils
from flax import linen as nn
from init2winit import checkpoint
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
from init2winit.hessian import hessian_eval
from init2winit.hessian import run_lanczos
from init2winit.init_lib import initializers
from init2winit.model_lib import models
from init2winit.trainer_lib import trainer
from jax.config import config as jax_config
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random
from ml_collections import config_dict
import numpy as np
import optax
import tensorflow.compat.v1 as tf  # importing this is needed for tfds mocking.
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

jax_config.parse_flags_with_absl()

CONFIG = {
    'num_batches': 25,
    'rng_key': 0,
    'use_training_gen': True,
    'update_stats': True,
    'num_points': 20,
    'num_eval_draws': 6,
    'compute_stats': True,
    'lower_thresh': -0.1,
    'upper_thresh': 0.1,
    'name': 'stats',
    'eval_hessian': True,
    'eval_hess_grad_overlap': True,
    'eval_gradient_covariance': True,
    'compute_interps': True,
    'num_lanczos_steps': 40,
    'hparam_overrides': {},
    'average_hosts': True,
    'num_eigens': 3}


def _batch_square_loss(flax_module, params, batch):
  """Helper function to compute square loss of model on the given batch.

  The function computes frac{1}{B} sum_{i=1}^B (y - hat{y})^2 where B is the
  batch-size.

  Args:
    flax_module: The flax module representing the model.
    params: The model parameters.
    batch: A dictionary with keys 'inputs' and 'targets'.

  Returns:
    total_loss: The loss averaged over the batch.
  """
  batch, rng = batch
  del rng
  batch_size = batch['targets'].shape[0]
  preds = flax_module.apply(
      {'params': params}, batch['inputs'], train=True).reshape((batch_size, -1))
  batch_targets = batch['targets'].reshape((batch_size, -1))
  square_loss = jnp.mean(jnp.sum(jnp.square(preds - batch_targets), axis=1))
  total_loss = square_loss
  return total_loss


class LinearModel(nn.Module):
  """Defines a simple linear model for the purpose of testing.

  The model assumes the input data has shape
  [batch_size_per_device, feature_dim]. The model flatten the input before
  applying a dense layer.
  """
  num_outputs: int

  @nn.compact
  def __call__(self, x, train):
    del train
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(features=self.num_outputs, use_bias=False)(x)
    return x


def _get_synth_data(num_examples, dim, num_outputs, batch_size):
  """Generates a fake data class for testing."""
  hess = np.ones((1, dim))
  hess[0, :CONFIG['num_eigens']] += np.arange(CONFIG['num_eigens'])
  feature = np.random.normal(size=(num_examples, dim)) / np.sqrt(dim)
  feature = np.multiply(feature, hess)
  feature = feature.astype(np.float32)
  y = np.random.normal(size=(num_examples, num_outputs))
  y = y.astype(np.float32)

  class SynthData(object):

    def train_iterator_fn(self):
      for ind in range(0, num_examples, batch_size):
        batch = {'inputs': feature[ind:ind + batch_size, :],
                 'targets': y[ind:ind + batch_size, :]}
        yield batch
  return SynthData, feature, y


def _to_vec(pytree):
  """Helper function that converts a pytree to a n-by-1 vector."""
  vec, _ = ravel_pytree(pytree)
  n = len(vec)
  vec = vec.reshape((n, 1))
  return vec


def _quad_grad(x, y, beta):
  """Computes the gradient of a linear model with square loss."""
  num_obs = x.shape[0]
  assert len(y.shape) == 2 and y.shape[0] == num_obs
  exact_grad = - np.dot(x.T, y) + np.dot(x.T, np.dot(x, beta))
  exact_grad = 2 * exact_grad / num_obs
  return exact_grad


class TrainerTest(absltest.TestCase):
  """Tests examining the CurvatureEvaluator class."""

  def setUp(self):
    super(TrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    rng = jax.random.PRNGKey(0)
    np.random.seed(0)
    self.feature_dim = 100
    num_outputs = 1
    self.batch_size = 32
    num_examples = 2048

    def create_model(key):
      flax_module = LinearModel(num_outputs=num_outputs)
      model_init_fn = jax.jit(functools.partial(flax_module.init, train=False))
      fake_input_batch = np.zeros((self.batch_size, self.feature_dim))
      init_dict = model_init_fn({'params': key}, fake_input_batch)
      params = init_dict['params']
      return flax_module, params

    flax_module, params = create_model(rng)
    # Linear model coefficients
    self.beta = params['Dense_0']['kernel']
    self.beta = self.beta.reshape((self.feature_dim, 1))
    self.beta = self.beta.astype(np.float32)

    optimizer_init_fn, self.optimizer_update_fn = optax.sgd(1.0)
    self.optimizer_state = jax_utils.replicate(optimizer_init_fn(params))
    self.params = jax_utils.replicate(params)

    data_class, self.feature, self.y = _get_synth_data(num_examples,
                                                       self.feature_dim,
                                                       num_outputs,
                                                       self.batch_size)
    self.evaluator = hessian_eval.CurvatureEvaluator(
        self.params,
        CONFIG,
        dataset=data_class(),
        loss=functools.partial(_batch_square_loss, flax_module))
    # Computing the exact full-batch quantities from the linear model
    num_obs = CONFIG['num_batches'] * self.batch_size
    xb = self.feature[:num_obs, :]
    yb = self.y[:num_obs, :]
    self.fb_grad = _quad_grad(xb, yb, self.beta)
    self.hessian = 2 * np.dot(xb.T, xb) / num_obs

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(TrainerTest, self).tearDown()

  def test_run_lanczos(self):
    """Test training for two epochs on MNIST with a small model."""
    rng = jax.random.PRNGKey(0)

    # Set the numpy seed to make the fake data deterministc. mocking.mock_data
    # ultimately calls numpy.random.
    np.random.seed(0)

    model_name = 'fully_connected'
    loss_name = 'cross_entropy'
    metrics_name = 'classification_metrics'
    initializer_name = 'noop'
    dataset_name = 'mnist'
    model_cls = models.get_model(model_name)
    initializer = initializers.get_initializer(initializer_name)
    dataset_builder = datasets.get_dataset(dataset_name)
    hparam_overrides = {
        'lr_hparams': {
            'base_lr': 0.1,
            'schedule': 'cosine'
        },
        'batch_size': 8,
        'train_size': 160,
        'valid_size': 96,
        'test_size': 80,
    }
    input_pipeline_hps = config_dict.ConfigDict(dict(
        num_tf_data_prefetches=-1,
        num_device_prefetches=0,
        num_tf_data_map_parallel_calls=-1,
    ))
    hps = hyperparameters.build_hparams(
        model_name,
        initializer_name,
        dataset_name,
        hparam_file=None,
        hparam_overrides=hparam_overrides,
        input_pipeline_hps=input_pipeline_hps)
    model = model_cls(hps, datasets.get_dataset_meta_data(dataset_name),
                      loss_name, metrics_name)

    eval_batch_size = 16
    num_examples = 256

    def as_dataset(self, *args, **kwargs):
      del args
      del kwargs

      # pylint: disable=g-long-lambda,g-complex-comprehension
      return tf.data.Dataset.from_generator(
          lambda: ({
              'image': np.ones(shape=(28, 28, 1), dtype=np.uint8),
              'label': 9,
          } for i in range(num_examples)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )

    # This will override the tfds.load(mnist) call to return 100 fake samples.
    with tfds.testing.mock_data(
        as_dataset_fn=as_dataset, num_examples=num_examples):
      dataset = dataset_builder(
          shuffle_rng=jax.random.PRNGKey(0),
          batch_size=hps.batch_size,
          eval_batch_size=eval_batch_size,
          hps=hps)

    num_train_steps = 41
    eval_num_batches = 5
    test_num_batches = 0
    eval_every = 10
    checkpoint_steps = [40]
    _ = list(
        trainer.Trainer(
            train_dir=self.test_dir,
            model=model,
            dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
            initializer=initializer,
            num_train_steps=num_train_steps,
            hps=hps,
            rng=rng,
            eval_batch_size=eval_batch_size,
            eval_num_batches=eval_num_batches,
            test_num_batches=test_num_batches,
            eval_train_num_batches=eval_num_batches,
            eval_frequency=eval_every,
            checkpoint_steps=checkpoint_steps).train())

    checkpoint_dir = os.path.join(self.test_dir, 'checkpoints')
    rng = jax.random.PRNGKey(0)

    run_lanczos.eval_checkpoints(
        checkpoint_dir,
        hps,
        rng,
        eval_num_batches,
        test_num_batches,
        model_cls=model_cls,
        dataset_builder=lambda *unused_args, **unused_kwargs: dataset,
        dataset_meta_data=datasets.get_dataset_meta_data(dataset_name),
        hessian_eval_config=CONFIG,
    )

    # Load the saved file.
    stats_path = os.path.join(checkpoint_dir, 'stats', 'training_metrics')
    pytree_list = checkpoint.load_pytree(stats_path)
    # Test that the logged steps are correct.
    pytree_list = [pytree_list[str(i)] for i in range(len(pytree_list))]
    saved_steps = [row['step'] for row in pytree_list]
    self.assertEqual(saved_steps, checkpoint_steps)

  def test_grads(self):
    """Test the computed gradients using a linear model."""
    dim = self.feature_dim
    bs = self.batch_size
    num_batches = CONFIG['num_batches']
    num_draws = CONFIG['num_eval_draws']

    grads, _ = self.evaluator.compute_dirs(
        self.params, self.optimizer_state, self.optimizer_update_fn)
    # Assert both full and mini batch gradients are accurate
    for i in range(num_draws + 1):
      dir_vec = _to_vec(grads[i])[:, 0]
      self.assertLen(dir_vec, dim)
      # i == num_draws corresponds to full-batch directions
      if i == num_draws:
        start = 0
        end = num_batches * bs
      else:
        start = i * bs
        end = (i + 1) * bs
      xb = self.feature[start:end, :]
      yb = self.y[start:end, :]
      exact_grad = _quad_grad(xb, yb, self.beta)[:, 0]
      add_err = np.max(np.abs(dir_vec - exact_grad))
      self.assertLessEqual(add_err, 1e-6)
      rel_err = np.abs(exact_grad / dir_vec - 1.0)
      rel_err = np.max(rel_err)
      self.assertLessEqual(rel_err, 1e-4)

  def test_statistics(self):
    """Test the computed statistics using a linear model."""
    bs = self.batch_size
    num_batches = CONFIG['num_batches']
    num_draws = CONFIG['num_eval_draws']
    step = 0

    grads, _ = self.evaluator.compute_dirs(
        self.params, self.optimizer_state, self.optimizer_update_fn)
    _, q = np.linalg.eigh(self.hessian)
    evecs = [q[:, -k] for k in range(CONFIG['num_eigens'], 0, -1)]
    q = q[:, -CONFIG['num_eigens']:]
    stats_row = self.evaluator.evaluate_stats(self.params, grads, [],
                                              evecs, [], step)
    # Assert that the statistics are exact
    for i in range(num_draws + 1):
      if i == num_draws:
        start = 0
        end = num_batches * bs
      else:
        start = i * bs
        end = (i + 1) * bs
      xb = self.feature[start:end, :]
      yb = self.y[start:end, :]
      exact_grad = _quad_grad(xb, yb, self.beta)
      exact_overlap = np.sum(np.multiply(exact_grad, self.fb_grad))
      overlap = stats_row['overlap%d'%(i,)]
      self.assertAlmostEqual(exact_overlap, overlap, places=5)
      exact_norm = np.linalg.norm(exact_grad) ** 2
      norm = stats_row['norm%d'%(i,)]
      self.assertAlmostEqual(exact_norm, norm, places=5)
      exact_quad = np.dot(exact_grad.T, np.dot(self.hessian, exact_grad))[0, 0]
      quad = stats_row['quad%d'%(i,)]
      self.assertAlmostEqual(exact_quad, quad, places=5)
      noise = exact_grad - self.fb_grad
      exact_quad = np.dot(noise.T, np.dot(self.hessian, noise))[0, 0]
      quad = stats_row['quad_noise%d'%(i,)]
      self.assertAlmostEqual(exact_quad, quad, places=5)
      inner_prods = np.dot(q.T,
                           exact_grad / np.linalg.norm(exact_grad)).flatten()
      err = np.max(np.abs(inner_prods - stats_row['hTg'][:, i]))
      self.assertAlmostEqual(err, 0.0, places=4)

  def test_interpolation(self):
    """Test the linear interpolations using a linear model."""
    bs = self.batch_size
    num_batches = CONFIG['num_batches']
    num_draws = CONFIG['num_eval_draws']
    num_obs = num_batches * bs
    step = 0
    num_points = CONFIG['num_points']

    grads, _ = self.evaluator.compute_dirs(
        self.params, self.optimizer_state, self.optimizer_update_fn)
    _, q = np.linalg.eigh(self.hessian)
    evecs = [q[:, -k] for k in range(CONFIG['num_eigens'], 0, -1)]
    q = q[:, -CONFIG['num_eigens']:]
    interps_row = self.evaluator.compute_interpolations(self.params,
                                                        grads, [], evecs,
                                                        [], step)

    # Computing the exact full-batch quantities from the linear model
    etas = interps_row['step_size']
    xb = self.feature[:num_obs, :]
    yb = self.y[:num_obs, :]
    for i in range(num_draws + 1):
      exact_values = np.zeros((num_points,))
      dir_vec = _to_vec(grads[i])
      # Normalize:
      dir_vec = dir_vec / np.linalg.norm(dir_vec)
      for j in range(num_points):
        new_param = self.beta + etas[j] * dir_vec
        errs = yb - np.dot(xb, new_param)
        exact_values[j] = np.dot(errs.T, errs)[0, 0] / num_obs
      values = interps_row['loss%d'%(i,)]
      self.assertTrue(np.allclose(exact_values, values, atol=1e-6, rtol=1e-5))
    # Checking interpolations for the eigenvectors
    for i in range(CONFIG['num_eigens']):
      exact_values = np.zeros((num_points,))
      dir_vec = evecs[i].reshape(len(self.beta), 1)
      for j in range(num_points):
        new_param = self.beta + etas[j] * dir_vec
        errs = yb - np.dot(xb, new_param)
        exact_values[j] = np.dot(errs.T, errs)[0, 0] / num_obs
      values = interps_row['loss_hvec%d'%(i,)]
      self.assertTrue(np.allclose(exact_values, values, atol=1e-6, rtol=1e-5))

  def test_spectrum(self):
    """Test Hessian / Covariance spectrum using a linear model."""
    bs = self.batch_size
    num_batches = CONFIG['num_batches']
    vec_counts = CONFIG['num_eigens']
    step = 0
    # Compute the spectra
    row, hvex, cvex = self.evaluator.evaluate_spectrum(self.params, step)
    # Comparing with the exact Hessian
    num_obs = num_batches * bs
    xb = self.feature[:num_obs, :]
    hessian = 2 * np.dot(xb.T, xb) / num_obs
    lambda_max = np.max(np.linalg.eigvalsh(hessian))
    # Assert that the calculated top Hessian eigenvalue is accurate
    self.assertAlmostEqual(row['max_eig_hess'] / lambda_max, 1.0, 2)

    # Assert that the extreme Hessian eigenvectors are accurate
    w, _ = np.linalg.eigh(hessian)
    dim = len(w)
    w = np.sort(w)
    for i in range(vec_counts):
      # Bottom Eigenvectors
      ev = hvex[i].reshape((dim, 1))
      lambda_hat = np.dot(ev.T, np.dot(hessian, ev))[0, 0]
      self.assertAlmostEqual(lambda_hat / w[i], 1.0, 3)
      # Top Eigenvectors
      ev = hvex[-i - 1].reshape((dim, 1))
      lambda_hat = np.dot(ev.T, np.dot(hessian, ev))[0, 0]
      self.assertAlmostEqual(lambda_hat / w[-i - 1], 1.0, 3)

    # Assert that the calculated top covariance eigenvalue is accurate
    yb = self.y[:num_obs, :]
    fb_grad = _quad_grad(xb, yb, self.beta)
    exact_cov = 0.0
    for i in range(num_batches):
      start = i * bs
      end = (i + 1) * bs
      xb = self.feature[start:end, :]
      yb = self.y[start:end, :]
      exact_grad = _quad_grad(xb, yb, self.beta)
      exact_cov += np.outer(exact_grad, exact_grad)
    exact_cov = exact_cov / (num_batches + 0.0)
    exact_cov += - np.outer(fb_grad, fb_grad)
    lambda_max = np.max(np.linalg.eigvalsh(exact_cov))
    self.assertAlmostEqual(row['max_eig_cov'], lambda_max, 4)

    # Assert that the extreme Covariance eigenvectors are accurate
    w, _ = np.linalg.eigh(exact_cov)
    dim = len(w)
    w = np.sort(w)
    for i in range(vec_counts):
      # Bottom Eigenvectors
      ev = cvex[i].reshape((dim, 1))
      lambda_hat = np.dot(ev.T, np.dot(exact_cov, ev))[0, 0]
      self.assertAlmostEqual(lambda_hat, 0.0, 4)
      # Top Eigenvectors
      ev = cvex[-i - 1].reshape((dim, 1))
      lambda_hat = np.dot(ev.T, np.dot(exact_cov, ev))[0, 0]
      self.assertAlmostEqual(lambda_hat / w[-i - 1], 1.0, 4)

  def test_update_dirs(self):
    """Testing the statistics computed from optimizer's update directions."""
    step = 0
    grads, updates = self.evaluator.compute_dirs(
        self.params, self.optimizer_state, self.optimizer_update_fn)
    stats_row = self.evaluator.evaluate_stats(self.params,
                                              grads, updates, [], [], step)
    interps_row = self.evaluator.compute_interpolations(self.params,
                                                        grads, updates, [], [],
                                                        step)

    # Since the optimizer is vanilla sgd, the gradient and the update stats
    # should match exactly.
    for i in range(CONFIG['num_eval_draws'] + 1):
      gdir_vec = _to_vec(grads[i])
      udir_vec = _to_vec(updates[i])
      add_err = np.max(np.abs(gdir_vec - udir_vec))
      self.assertLessEqual(add_err, 1e-6)
      diff = interps_row['loss%d'%(i,)] - interps_row['loss_u%d'%(i,)]
      add_err = np.max(np.abs(diff))
      self.assertLessEqual(add_err, 1e-6)
      self.assertAlmostEqual(stats_row['quad%d'%(i,)],
                             stats_row['quad_u%d'%(i,)], 5)
      self.assertAlmostEqual(stats_row['norm%d'%(i,)],
                             stats_row['norm_u%d'%(i,)], 5)
      self.assertAlmostEqual(stats_row['overlap%d'%(i,)],
                             stats_row['overlap_u%d'%(i,)], 5)
      self.assertAlmostEqual(stats_row['overlap%d'%(i,)],
                             stats_row['fb_overlap%d'%(i,)], 5)

  def test_eval_hess_grad_overlap(self):
    """Test gradient overlap calculations."""

    if jax.devices()[0].platform == 'tpu':
      atol = 1e-3
      rtol = 0.1
    else:
      atol = 1e-5
      rtol = 1e-5

    # dimension of space
    n_params = 4
    num_batches = 5

    eval_config = hessian_eval.DEFAULT_EVAL_CONFIG
    eval_config['eval_hessian'] = False
    eval_config['eval_gradient_covariance'] = False
    eval_config['num_eigens'] = 0
    eval_config['num_lanczos_steps'] = n_params  # max iterations
    eval_config['num_batches'] = num_batches
    eval_config['num_eval_draws'] = 1

    key = jax.random.PRNGKey(0)
    key, split = jax.random.split(key)

    # Diagonal matrix values
    mat_diag = jax.random.normal(split, (n_params,))

    def batches_gen():
      for _ in range(num_batches):
        yield None

    # Model
    class QuadraticLoss(nn.Module):
      """Loss function which only depends on parameters."""

      @nn.compact
      def __call__(self, x):
        del x
        w = self.param('w', jax.random.normal, (n_params,))
        return jnp.sum((w ** 2) * mat_diag)

    flax_module = QuadraticLoss()

    def loss(params, _):
      # 1.0 is required but unused.
      return flax_module.apply({'params': params}, 1.0)

    # Model initialization.
    model_init_fn = jax.jit(flax_module.init)
    init_dict = model_init_fn(
        {'params': key}, np.zeros((num_batches,), jnp.float32))
    params = init_dict['params']

    # replicate
    replicated_params = jax_utils.replicate(params)

    curve_eval = hessian_eval.CurvatureEvaluator(
        replicated_params,
        eval_config,
        loss,
        dataset=None,
        batches_gen=batches_gen)
    row, _, _ = curve_eval.evaluate_spectrum(replicated_params, 0)

    tridiag = row['tridiag_hess_grad_overlap']
    eigs_triag, vecs_triag = np.linalg.eigh(tridiag)

    # Test eigenvalues
    np.testing.assert_allclose(
        eigs_triag, 2*np.sort(mat_diag), atol=atol, rtol=rtol)

    # Compute overlaps
    weights_triag = vecs_triag[0, :]**2
    grad_true = 2 * params['w'] * mat_diag
    weight_idx = np.argsort(mat_diag)
    weights_true = (grad_true)**2 / jnp.dot(grad_true, grad_true)
    weights_true = weights_true[weight_idx]

    # Test overlaps
    np.testing.assert_allclose(
        weights_triag, weights_true, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
