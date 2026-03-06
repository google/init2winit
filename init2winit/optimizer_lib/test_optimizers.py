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

"""Tests for optimizers."""

import shutil
import tempfile

from absl.testing import absltest
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
from init2winit.optimizer_lib import muon
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib import utils as optimizers_utils
from init2winit.optimizer_lib.kitchen_sink._src.transform import ScaleByAdapropState  # pylint: disable=g-importing-member
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import optax
from optax._src.transform import ScaleByAdamState  # pylint: disable=g-importing-member

# TODO(b/385225663): add test for nadamw.

ParameterType = model_utils.ParameterType


class GenericMultiOptimizerTest(absltest.TestCase):
  """Integration tests for learning.clair.alise.optimizers.experimental.noise_adaptive."""

  def test_generic_multi_optimizer_init(self):
    """Tests that the gradient aggregation tag propagates correctly."""
    adam_opt_hparams = {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
        },
    }

    raga_opt_hparams = {
        'optimizer': 'raga',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
            'beta3': 1.0,
            'beta4': 0.99,
            'nesterov': True,
            'power': 2.0,
            'quantized_dtype': 'float32',
        },
    }

    experiment_config = config_dict.ConfigDict({
        'dataset': 'c4',
        'model': 'nanodo',
        'loss': 'cross_entropy',
        'metrics': 'classification_metrics',
        'initializer': 'noop',
        'hparam_overrides': config_dict.ConfigDict({
            'optimizer': 'generic_multi_optimizer',
            'l2_decay_factor': None,
            'batch_size': 50,
            'opt_hparams': {
                'param_type_to_optimizer_and_hparams': {
                    ParameterType.DEFAULT.value: adam_opt_hparams,
                    ParameterType.ATTENTION_Q.value: raga_opt_hparams,
                    ParameterType.ATTENTION_K.value: raga_opt_hparams,
                    ParameterType.ATTENTION_OUT.value: raga_opt_hparams,
                }
            },
        }),
    })

    model_cls = models.get_model(experiment_config.model)
    dataset_meta_data = datasets.get_dataset_meta_data(
        experiment_config.dataset
    )

    merged_hps = hyperparameters.build_hparams(
        experiment_config.model,
        experiment_config.initializer,
        experiment_config.dataset,
        hparam_file=None,
        hparam_overrides=experiment_config.hparam_overrides,
    )

    model = model_cls(
        merged_hps,
        dataset_meta_data,
        experiment_config.loss,
        experiment_config.metrics,
    )

    noop = initializers.get_initializer('noop')
    rng = jax.random.PRNGKey(0)
    init_dict = model.initialize(
        rng=rng, metrics_logger=None, initializer=noop, hps=merged_hps
    )
    unreplicated_params, _ = init_dict

    opt_init_fn, _ = optimizers.get_optimizer(merged_hps, model)

    unreplicated_optimizer_state = opt_init_fn(unreplicated_params)
    self.assertIsInstance(
        unreplicated_optimizer_state, optax.transforms.PartitionState
    )

    # unreplicated_optimizer_state should be a Dict mapping param type
    # to opt_state where only params mapping to that param_type have non-empty
    # states.
    self.assertEqual(
        len(unreplicated_optimizer_state.inner_states),
        len(set(jax.tree_util.tree_leaves(model.params_types))),
    )

    # Check that the optax multi transforms correctly maps param types to
    # optimizers.
    self.assertIsInstance(
        unreplicated_optimizer_state.inner_states[
            ParameterType.ATTENTION_K.value
        ].inner_state.inner_state[0],
        ScaleByAdapropState,
    )

    self.assertIsInstance(
        unreplicated_optimizer_state.inner_states[
            ParameterType.WEIGHT.value
        ].inner_state.inner_state[0],
        ScaleByAdamState,
    )


class OptimizersTrainerTest(absltest.TestCase):
  """Tests for optimizers.py that require starting a trainer object."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    self.trainer.wait_until_orbax_checkpointer_finished()
    shutil.rmtree(self.test_dir)
    super().tearDown()




class MuonTest(absltest.TestCase):
  """Tests for Muon optimizer."""

  def test_muon_split(self):
    """Verifies that Muon handles 2D params and RMSProp handles 1D params."""
    params = {
        'p1d': jnp.ones((10,)),
        'p2d': jnp.ones((10, 10)),
    }
    grads = {
        'p1d': jnp.full((10,), 0.1),
        'p2d': jnp.eye(10) * 0.1,
    }

    lr = 0.1
    beta = 0.95
    wd = 0.01

    muon_tx = muon.scale_by_muon(
        learning_rate=lr,
        beta=beta,
        weight_decay=wd,
    )
    rms_prop = optax.chain(
        optax.scale_by_rms(decay=beta, eps=1e-7),
        optax.add_decayed_weights(wd),
        optax.scale_by_learning_rate(lr, flip_sign=True),
    )
    muon_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
    rmsprop_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim < 2, p)
    tx = optax.chain(
        optax.masked(muon_tx, mask=muon_mask),
        optax.masked(rms_prop, mask=rmsprop_mask),
    )

    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)

    expected_1d = -0.4482136
    self.assertTrue(jnp.allclose(updates['p1d'], expected_1d, rtol=1e-3))

    self.assertFalse(jnp.allclose(updates['p2d'], -0.4482136, rtol=1e-1))

  def test_muon_get_optimizer(self):
    """Verifies get_optimizer('muon') matches manual construction."""
    params = {
        'p1d': jnp.ones((10,)),
        'p2d': jnp.ones((10, 10)),
    }
    grads = {
        'p1d': jnp.full((10,), 0.1),
        'p2d': jnp.eye(10) * 0.1,
    }

    lr = 0.1
    beta = 0.95
    wd = 0.01

    hps = config_dict.ConfigDict({
        'optimizer': 'muon',
        'l2_decay_factor': None,
        'batch_size': 8,
        'opt_hparams': {
            'muon_hparams': {
                'beta': beta,
                'weight_decay': wd,
            },
            'pair_optimizer': 'sgd',
            'pair_hparams': {
                'weight_decay': wd,
            },
        },
    })

    opt_init, opt_update = optimizers.get_optimizer(hps)
    state = opt_init(params)
    state = optimizers.inject_learning_rate(state, lr)
    updates, _ = opt_update(grads, state, params=params)

    muon_tx = muon.scale_by_muon(
        learning_rate=lr,
        beta=beta,
        weight_decay=wd,
    )
    sgd_tx = optax.chain(
        optax.add_decayed_weights(wd),
        optax.sgd(learning_rate=lr, momentum=None, nesterov=False),
    )
    muon_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
    pair_mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim < 2, p)
    manual_tx = optax.chain(
        optax.masked(muon_tx, mask=muon_mask),
        optax.masked(sgd_tx, mask=pair_mask),
    )
    manual_state = manual_tx.init(params)
    manual_updates, _ = manual_tx.update(grads, manual_state, params)

    self.assertTrue(
        jnp.allclose(updates['p2d'], manual_updates['p2d'], atol=1e-5)
    )
    self.assertTrue(
        jnp.allclose(updates['p1d'], manual_updates['p1d'], atol=1e-5)
    )

  def test_muon_lr_multiplier(self):
    """Verifies lr_multiplier scales Muon's effective learning rate."""
    params = {'w': jnp.ones((10, 10))}
    grads = {'w': jnp.eye(10) * 0.1}

    lr = 0.1
    base = muon.scale_by_muon(learning_rate=lr, lr_multiplier=1.0)
    scaled = muon.scale_by_muon(learning_rate=lr, lr_multiplier=2.0)

    base_state = base.init(params)
    scaled_state = scaled.init(params)

    base_updates, _ = base.update(grads, base_state, params)
    scaled_updates, _ = scaled.update(grads, scaled_state, params)

    self.assertTrue(
        jnp.allclose(scaled_updates['w'], 2.0 * base_updates['w'], atol=1e-6)
    )

  def test_muon_3d_reshape(self):
    """Verifies Muon reshapes >2D params to (s0, -1) for orthogonalization."""
    params_3d = {'w': jnp.ones((4, 5, 6))}
    grads_3d = {'w': jax.random.normal(jax.random.PRNGKey(0), (4, 5, 6))}

    tx = muon.scale_by_muon(learning_rate=0.1)
    state = tx.init(params_3d)
    updates, _ = tx.update(grads_3d, state, params_3d)

    self.assertEqual(updates['w'].shape, (4, 5, 6))

  def test_muon_embed_mask(self):
    """Verifies default mask excludes embed params and 1D params."""
    params = {
        'embed': jnp.ones((100, 64)),
        'dense': jnp.ones((64, 64)),
        'bias': jnp.ones((64,)),
    }

    muon_mask = lambda p: jax.tree_util.tree_map_with_path(
        lambda path, x: (
            x.ndim >= 2 and 'embed' not in jax.tree_util.keystr(path).lower()
        ),
        p,
    )
    mask = muon_mask(params)

    self.assertTrue(mask['dense'])
    self.assertFalse(mask['embed'])
    self.assertFalse(mask['bias'])


if __name__ == '__main__':
  absltest.main()
