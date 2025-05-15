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

"""Tests for optimizers."""
import shutil
import tempfile

from absl.testing import absltest
from init2winit import hyperparameters
from init2winit.dataset_lib import datasets
from init2winit.init_lib import initializers
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
from init2winit.optimizer_lib import optimizers
from init2winit.optimizer_lib import utils as optimizers_utils
from init2winit.optimizer_lib.kitchen_sink._src.transform import ScaleByAdapropState  # pylint: disable=g-importing-member
import jax
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
        hparam_overrides=experiment_config.hparam_overrides)

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
        unreplicated_optimizer_state,
        optax.transforms.PartitionState)

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



if __name__ == '__main__':
  absltest.main()
