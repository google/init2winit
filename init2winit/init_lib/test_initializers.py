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

r"""Tests for initializers.py.

"""
import copy

from absl.testing import absltest
from absl.testing import parameterized
from init2winit import utils
from init2winit.init_lib import initializers
from init2winit.init_lib import meta_init
from init2winit.model_lib import losses
from init2winit.model_lib import model_utils
from init2winit.model_lib import models
import jax.numpy as jnp
import jax.tree_util


BATCH_SIZE = 10
OUTPUT_SHAPE = (5,)
MODEL_TO_INPUT_SHAPE = {
    'fully_connected': (5,),
    'simple_cnn': (8, 8, 3),
    'max_pooling_cnn': (16, 16, 3),
}


TEST_PARAMETRIZATION = (('test_{}'.format(init), init)
                        for init in initializers._ALL_INITIALIZERS.keys())  # pylint: disable=protected-access


def _load_model(model_name):
  """Load a test model."""
  rng = jax.random.PRNGKey(0)
  model_cls = models.get_model(model_name)
  loss_name = 'cross_entropy'
  metrics_name = 'classification_metrics'
  model_hps = models.get_model_hparams(model_name)

  hps = copy.copy(model_hps)
  hps.update({'output_shape': OUTPUT_SHAPE})
  model = model_cls(hps, {}, loss_name, metrics_name)

  input_shape = (BATCH_SIZE,) + MODEL_TO_INPUT_SHAPE[model_name]
  _, flax_module = model.flax_module_def.create_by_shape(
      rng, [input_shape], train=True)
  utils.log_pytree_shape_and_statistics(flax_module.params)
  return flax_module, input_shape


class InitializersTest(parameterized.TestCase):
  """Tests for initializers.py."""

  @parameterized.named_parameters(params for params in TEST_PARAMETRIZATION)
  def test_initializers(self, init):
    """Test that each initializer runs, and the output is a valid pytree."""

    rng = jax.random.PRNGKey(0)
    flax_module, input_shape = _load_model('fully_connected')
    _, init_rng = jax.random.split(rng)
    initializer = initializers.get_initializer(init)
    init_hps = initializers.get_initializer_hparams(init)
    loss_name = 'cross_entropy'
    loss_fn = losses.get_loss_fn(loss_name)
    new_model = initializer(
        loss_fn=loss_fn,
        model=flax_module,
        hps=init_hps,
        input_shape=input_shape[1:],
        output_shape=OUTPUT_SHAPE,
        rng_key=init_rng)

    # Check new params are still valid params
    outputs = new_model(jnp.ones(input_shape))
    utils.log_pytree_shape_and_statistics(new_model.params)
    self.assertEqual(outputs.shape, (input_shape[0], OUTPUT_SHAPE[-1]))

  @parameterized.named_parameters(('test_{}'.format(model_name), model_name)
                                  for model_name in MODEL_TO_INPUT_SHAPE.keys())
  def test_meta_loss(self, model_name):
    """Test that meta_init does not update the bias scalars."""

    rng = jax.random.PRNGKey(0)
    flax_module, input_shape = _load_model(model_name)
    norms = jax.tree_map(lambda node: jnp.linalg.norm(node.reshape(-1)),
                         flax_module.params)
    normalized_params = jax.tree_map(meta_init.normalize,
                                     flax_module.params)
    loss_name = 'cross_entropy'
    loss_fn = losses.get_loss_fn(loss_name)
    learned_norms, _ = meta_init.meta_optimize_scales(
        loss_fn=loss_fn,
        fprop=flax_module.module.call,
        normalized_params=normalized_params,
        norms=norms,
        hps=meta_init.DEFAULT_HPARAMS,
        input_shape=input_shape[1:],
        output_shape=OUTPUT_SHAPE,
        rng_key=rng)

    # Check that all learned bias scales are 0, the meta loss should be
    # independent of these terms.
    learned_norms_flat = model_utils.flatten_dict(learned_norms)
    for layer_key in learned_norms_flat:
      if 'bias' in layer_key:
        self.assertEqual(learned_norms_flat[layer_key], 0.0)


if __name__ == '__main__':
  absltest.main()
