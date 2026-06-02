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

"""Base class for training algorithms."""

import abc
import collections

from absl import logging
from init2winit import schedules
from init2winit.model_lib import model_utils
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import optimizers
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import optax


_GRAD_CLIP_EPS = 1e-6


def optax_update_params_helper(
    params,
    model_state,
    optimizer_state,
    optimizer_update_fn,
    batch,
    lr,
    rng,
    grad_clip,
    training_cost_fn,
):
  """Helper function for updating parameters using optax.

  Args:
    params: The current model parameters.
    model_state: The current state of the model.
    optimizer_state: The current state of the optimizer.
    optimizer_update_fn: The optimizer update function.
    batch: The current batch of data.
    lr: The learning rate.
    rng: The random number generator.
    grad_clip: The gradient clipping value.
    training_cost_fn: The training cost function.

  Returns:
    A tuple containing the new optimizer state, the new model parameters,
    the new model state, and a dictionary of metrics.
  """
  optimizer_state = optimizers.inject_learning_rate(optimizer_state, lr)

  def opt_cost(params):
    return training_cost_fn(
        params, batch=batch, batch_stats=model_state, dropout_rng=rng
    )

  grad_fn = jax.value_and_grad(opt_cost, has_aux=True)
  (cost_value, new_batch_stats), grad = grad_fn(params)
  new_batch_stats = new_batch_stats.get('batch_stats', None)

  grad_norm = jnp.sqrt(model_utils.l2_regularization(grad, 0))
  if grad_clip:
    scaled_grad = jax.tree.map(
        lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad
    )
    grad = jax.lax.cond(
        grad_norm > grad_clip, lambda _: scaled_grad, lambda _: grad, None
    )
  model_updates, new_optimizer_state = optimizer_update_fn(
      grad,
      optimizer_state,
      params=params,
      batch=batch,
      batch_stats=new_batch_stats,
      cost_fn=opt_cost,
      grad_fn=grad_fn,
      value=cost_value,
  )

  update_norm = jnp.sqrt(model_utils.l2_regularization(model_updates, 0))

  new_params = optax.apply_updates(params, model_updates)
  return (
      new_optimizer_state,
      new_params,
      new_batch_stats,
      cost_value,
      grad,
      grad_norm,
      update_norm,
  )


class TrainingAlgorithm(metaclass=abc.ABCMeta):
  """Base class for training algorithms."""

  def __init__(self, hps, model, num_train_steps):
    self.model = model
    self.num_train_steps = num_train_steps
    self.hps = hps
    self.eval_report_metrics = collections.defaultdict()

  @classmethod
  def get_default_training_hparams(cls):
    """Returns default training hyperparameters.

    The base class provides infrastructure-level defaults. Subclasses should
    call super() and merge in their own optimizer/lr defaults.

    Returns:
      A ConfigDict of default training hyperparameters.
    """
    return config_dict.ConfigDict({
        'batch_size': None,
        'total_accumulated_batch_size': None,
        'l2_decay_factor': None,
        'l2_decay_rank_threshold': 2,
        'label_smoothing': None,
        'rng_seed': -1,
        'use_shallue_label_smoothing': False,
        'layer_rescale_factors': {},
    })

  @abc.abstractmethod
  def update_params(
      self,
      params,
      model_state,
      optimizer_state,
      batch,
      global_step,
      rng,
      hyperparameters=None,
      workload=None,
      param_types=None,
      loss_type=None,
      train_state=None,
      eval_results=None,
  ):
    """Updates the model parameters.

    Args:
      params: The current model parameters.
      model_state: The current state of the model.
      optimizer_state: The current state of the optimizer.
      batch: The current batch of data.
      global_step: The current training step.
      rng: The random number generator.
      hyperparameters: The hyperparameters for the training.
      workload: The workload being trained.
      param_types: The types of the parameters.
      loss_type: The type of loss function to use.
      train_state: The optional training state.
      eval_results: The optional evaluation results.

    Returns:
      A tuple containing:
        new_optimizer_state: Pytree of optimizer state.
        new_params: Pytree of model parameters.
        new_model_state: Pytree of model state.
        cost_value: The training cost.
        grad: The gradient.
    """

  @abc.abstractmethod
  def init_optimizer_state(
      self,
      model=None,
      params=None,
      model_state=None,
      hyperparameters=None,
      rng=None,
  ):
    """Initializes the optimizer state.

    Args:
      model: The model being trained.
      params: The initial model parameters.
      model_state: The initial state of the model.
      hyperparameters: The hyperparameters for the training.
      rng: The random number generator.

    Returns:
      Optimizer state: Pytree of optimizer state.
    """

  def restore_optimizer_state(self, optimizer_state):
    """Post-processes optimizer state after checkpoint restore.

    Override this method in subclasses that use runtime wrappers (e.g.,
    CpuOffloaded) which are stripped during serialization and need to be
    re-applied after deserialization.

    Args:
      optimizer_state: The restored optimizer state pytree (plain numpy arrays).

    Returns:
      The post-processed optimizer state, ready for sharding.
    """
    return optimizer_state


# Per-optimizer default opt_hparams for OptaxTrainingAlgorithm.
# These consolidate all the inline defaults from get_optimizer() in
# optimizer_lib/optimizers.py so that configs don't need to redundantly
# specify values that match the defaults.
_OPTAX_OPTIMIZER_DEFAULTS = {
    'adam': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'sgd': {
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'momentum': {
        'momentum': 0.9,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'nesterov': {
        'momentum': 0.9,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'nadam': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'debias': True,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'generalized_adam': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'nesterov': True,
        'power': 2.0,
        'disable_preconditioning': False,
        'epsilon_root': 0.0,
        'debias': True,
        'weight_decay': 0.0,
        'disable_multiply_wd_by_base_lr': False,
        'grad_clip': None,
    },
    'nadamp': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'debias': True,
        'power': 2.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'adaprop': {
        'beta1': 0.9,
        'beta2': 0.999,
        'beta3': 1.0,
        'beta4': 0.999,
        'epsilon': 1e-8,
        'power': 2.0,
        'nesterov': True,
        'quantized_dtype': 'float32',
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'adafactor': {
        'min_dim_size_to_factor': 128,
        'adafactor_decay_rate': 0.8,
        'decay_offset': 0,
        'multiply_by_parameter_scale': True,
        'clipping_threshold': 1.0,
        'momentum': None,
        'epsilon': 1e-30,
        'factored': True,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'lamb': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-6,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'lars': {
        'trust_coefficient': 0.001,
        'epsilon': 1e-6,
        'momentum': 0.9,
        'nesterov': False,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'adabelief': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'radam': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'adam_relative_update': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'herolion': {
        'beta1': 0.9,
        'beta2': 0.99,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'bubbles': {
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'nesterov': True,
        'min_steps': 100,
        'grad_rms_threshold': 10.0,
        'precond_grad_clip': None,
        'bias_correction': True,
        'grad_clip': None,
    },
    'lora_bubbles': {
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'nesterov': True,
        'eps': 1e-7,
        'lora_min_steps': 100,
        'lora_update_steps': 20,
        'lora_rank': 64,
        'grad_rms_threshold': 10.0,
        'precond_grad_clip': None,
        'bias_correction': True,
        'grad_clip': None,
    },
    'diag_bubbles': {
        'beta1': None,
        'beta2': 0.999,
        'eps': 1e-8,
        'precond_grad_clip': None,
        'nesterov': False,
        'bias_correction': True,
        'weight_decay': 1e-4,
        'grad_clip': None,
    },
    'decoupled_adam': {
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'grad_clip': None,
    },
    'adan': {
        'beta1': 0.98,
        'beta2': 0.92,
        'beta3': 0.99,
        'epsilon': 1e-8,
        'epsilon_root': 0.0,
        'weight_decay': 0.0,
        'use_adamw_wd': True,
        'tie_b1_b2': False,
        'grad_clip': None,
    },
    'sm3': {
        'beta1': 0.9,
        'beta2': 0.999,
        'diagonal_epsilon': 1e-10,
        'weight_decay': 0.0,
        'normalize_grads': False,
        'grad_clip': None,
    },
}

# UNet piecewise_constant schedule constants (originally in unet.py).
_FASTMRI_TRAIN_SIZE = 34742
_UNET_BATCH_SIZE = 8
_UNET_NUM_EPOCHS = 50
_UNET_STEPS_PER_EPOCH = int(_FASTMRI_TRAIN_SIZE / _UNET_BATCH_SIZE)
_UNET_NUM_TRAIN_STEPS = _UNET_NUM_EPOCHS * _UNET_STEPS_PER_EPOCH
_UNET_LR_GAMMA = 0.1
_UNET_LR_STEP_SIZE = 40 * _UNET_STEPS_PER_EPOCH
_UNET_DECAY_EVENTS = list(
    range(_UNET_LR_STEP_SIZE, _UNET_NUM_TRAIN_STEPS, _UNET_LR_STEP_SIZE)
)
_UNET_DECAY_FACTORS = [
    _UNET_LR_GAMMA**i for i in range(1, len(_UNET_DECAY_EVENTS) + 1)
]

# Per-model training defaults, preserving the historical optimizer configuration
# that each model was originally tuned with. These are used as Tier 2 fallback
# when no explicit optimizer is specified in hparam_overrides.
_MODEL_TRAINING_DEFAULTS = {
    # Vision models with momentum
    'adabelief_densenet': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.2, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0001,
    },
    'adabelief_resnet': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.2, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0001,
    },
    'adabelief_vgg': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.2, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0001,
    },
    'resnet': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.2, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0001,
    },
    'fully_connected': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0005,
    },
    'simple_cnn': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.001, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0005,
    },
    'max_pooling_cnn': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.001, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 0.0005,
    },
    'wide_resnet': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.9},
        'lr_hparams': {'base_lr': 0.001, 'schedule': 'cosine'},
        'batch_size': 128,
        'l2_decay_factor': 0.0001,
    },
    'convolutional_autoencoder': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.0},
        'lr_hparams': {'base_lr': 0.02, 'schedule': 'constant'},
        'batch_size': 128,
    },
    'nqm': {
        'optimizer': 'momentum',
        'opt_hparams': {'momentum': 0.0},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 128,
    },
    # Vision models with adam
    'vit': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.1},
        'lr_hparams': {'base_lr': 1e-3, 'schedule': 'cosine_warmup'},
        'batch_size': 1024,
    },
    # Speech models
    'conformer': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0, 'grad_clip': 5.0},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 1e-6,
        'l2_decay_rank_threshold': 0,
    },
    'mlcommons_conformer': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0, 'grad_clip': 5.0},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 1e-6,
        'l2_decay_rank_threshold': 0,
    },
    'deepspeech': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0, 'grad_clip': 10.0},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 1e-6,
        'l2_decay_rank_threshold': 0,
    },
    'mlcommons_deepspeech': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0, 'grad_clip': 10.0},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 1e-6,
        'l2_decay_rank_threshold': 0,
    },
    'transformer': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
        },
        'lr_hparams': {
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup',
        },
        'batch_size': 512,
        'l2_decay_rank_threshold': 0,
    },
    'performer': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
        },
        'lr_hparams': {
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup',
        },
        'batch_size': 512,
        'l2_decay_rank_threshold': 0,
    },
    'transformer_stu': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
        },
        'lr_hparams': {
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup',
        },
        'batch_size': 512,
        'l2_decay_rank_threshold': 0,
    },
    'transformer_stu_tensordot': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.1,
        },
        'lr_hparams': {
            'base_lr': 0.0016,
            'warmup_steps': 1000,
            'squash_steps': 1000,
            'schedule': 'rsqrt_normalized_decay_warmup',
        },
        'batch_size': 512,
        'l2_decay_rank_threshold': 0,
    },
    'xformer_translate': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.0,
        },
        'lr_hparams': {
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound',
        },
        'batch_size': 64,
        'l2_decay_rank_threshold': 0,
    },
    'mlcommons_xformer_translate': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.0,
        },
        'lr_hparams': {
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound',
        },
        'batch_size': 64,
        'l2_decay_rank_threshold': 0,
    },
    'xformer_translate_binary': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.0,
        },
        'lr_hparams': {
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound',
        },
        'batch_size': 64,
        'l2_decay_rank_threshold': 0,
    },
    'xformer_translate_mlc_variant': {
        'optimizer': 'adam',
        'opt_hparams': {
            'beta1': 0.9,
            'beta2': 0.98,
            'epsilon': 1e-9,
            'weight_decay': 0.0,
        },
        'lr_hparams': {
            'base_lr': 0.05,
            'warmup_steps': 8000,
            'factors': 'constant * linear_warmup * rsqrt_decay',
            'schedule': 'compound',
        },
        'batch_size': 64,
        'l2_decay_rank_threshold': 0,
    },
    'lstm': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {'base_lr': 1e-3, 'schedule': 'constant'},
        'batch_size': 256,
    },
    'local_attention_transformer': {
        'optimizer': 'adafactor',
        'opt_hparams': {},
        'lr_hparams': {
            'base_lr': 0.01,
            'defer_steps': 10000,
            'schedule': 't2t_rsqrt_normalized_decay',
        },
        'batch_size': 8,
    },
    # GNN / tabular
    'gnn': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 0.0005,
    },
    'dlrm': {
        'optimizer': 'adam',
        'opt_hparams': {},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 1e-5,
    },
    'dlrm_resnet': {
        'optimizer': 'adam',
        'opt_hparams': {},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 1e-5,
    },
    # Nanodo family
    'nanodo': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 0.0005,
    },
    'rope_nanodo': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 0.0005,
    },
    'mdlm_rope_nanodo': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {'base_lr': 0.01, 'schedule': 'constant'},
        'batch_size': 256,
        'l2_decay_factor': 0.0005,
    },
    # Special
    'autoencoder': {
        'optimizer': 'hessian_free',
        'opt_hparams': {'damping_lb': 1e-6},
        'lr_hparams': {'base_lr': 0.1, 'schedule': 'constant'},
        'batch_size': 128,
        'l2_decay_factor': 2e-5,
        'l2_decay_rank_threshold': 1,
    },
    'mlperf_resnet': {
        'optimizer': 'lars',
        'opt_hparams': {'weight_decay': 2e-4, 'momentum': 0.9},
        'lr_hparams': {
            'base_lr': 10.0,
            'schedule': 'mlperf_polynomial',
            'start_lr': 0.0,
            'end_lr': 1e-4,
            'power': 2.0,
            'decay_end': -1,
            'warmup_steps': 18,
            'warmup_power': 1,
        },
        'batch_size': 128,
    },
    'fake_resnet': {
        'optimizer': 'lars',
        'opt_hparams': {'weight_decay': 2e-4, 'momentum': 0.9},
        'lr_hparams': {
            'base_lr': 10.0,
            'schedule': 'mlperf_polynomial',
            'start_lr': 0.0,
            'end_lr': 1e-4,
            'power': 2.0,
            'decay_end': -1,
            'warmup_steps': 18,
            'warmup_power': 1,
        },
        'batch_size': 128,
    },
    'unet': {
        'optimizer': 'adam',
        'opt_hparams': {'weight_decay': 0.0},
        'lr_hparams': {
            'schedule': 'piecewise_constant',
            'base_lr': 1e-3,
            'decay_events': _UNET_DECAY_EVENTS,
            'decay_factors': _UNET_DECAY_FACTORS,
        },
        'batch_size': 8,
    },
}


class OptaxTrainingAlgorithm(TrainingAlgorithm):
  """Class for training algorithms implemented with optax and defined in optimizer_lib.optimizers.py."""

  @classmethod
  def get_default_training_hparams(cls, optimizer_name=None, model_name=None):
    """Returns default training hparams for optax-based training.

    Resolution hierarchy:
      1. If optimizer_name is provided, use optimizer-specific defaults.
      2. Else if model_name is in _MODEL_TRAINING_DEFAULTS, use model-specific
         defaults (which include the model's historical optimizer choice).
      3. Else fall back to 'adam' defaults.

    Args:
      optimizer_name: Optional name of the optimizer to get defaults for. When
        provided, takes precedence over model_name.
      model_name: Optional name of the model. Used to look up historical
        per-model training defaults when no optimizer_name is specified.

    Returns:
      A ConfigDict of default training hyperparameters including
      optimizer-specific opt_hparams looked up from the per-optimizer defaults
      table.
    """
    training_hparams = super().get_default_training_hparams()

    if optimizer_name is not None:
      # Tier 1: explicit optimizer override.
      logging.info(
          'Using optimizer-specific defaults for optimizer=%s',
          optimizer_name,
      )
      opt_defaults = dict(_OPTAX_OPTIMIZER_DEFAULTS.get(optimizer_name, {}))
      training_hparams.update({
          'optimizer': optimizer_name,
          'opt_hparams': opt_defaults,
          'lr_hparams': {
              'base_lr': 0.001,
              'schedule': 'cosine',
          },
      })
    elif model_name is not None and model_name in _MODEL_TRAINING_DEFAULTS:
      # Tier 2: model-specific historical defaults.
      logging.info(
          'Using model-specific training defaults for model=%s', model_name
      )
      model_defaults = dict(_MODEL_TRAINING_DEFAULTS[model_name])
      model_optimizer = model_defaults.pop('optimizer', 'adam')
      # Start with the optimizer's own defaults, then overlay model-specific.
      opt_defaults = dict(_OPTAX_OPTIMIZER_DEFAULTS.get(model_optimizer, {}))
      opt_defaults.update(model_defaults.pop('opt_hparams', {}))
      training_hparams.update({
          'optimizer': model_optimizer,
          'opt_hparams': opt_defaults,
          'lr_hparams': model_defaults.pop(
              'lr_hparams', {'base_lr': 0.001, 'schedule': 'cosine'}
          ),
      })
      # Apply remaining model-level overrides (batch_size, l2_decay, etc.)
      training_hparams.update(model_defaults)
    else:
      # Tier 3: generic adam fallback.
      opt_defaults = dict(_OPTAX_OPTIMIZER_DEFAULTS.get('adam', {}))
      training_hparams.update({
          'optimizer': 'adam',
          'opt_hparams': opt_defaults,
          'lr_hparams': {
              'base_lr': 0.001,
              'schedule': 'cosine',
          },
      })

    return training_hparams

  def __init__(self, hps, model, num_train_steps):
    super().__init__(hps, model, num_train_steps)
    self._optimizer_state = None
    self._update_fn = None
    self._lr_fn = None
    self.training_cost_fn = model.training_cost

  def update_params(
      self,
      params,
      model_state,
      optimizer_state,
      batch,
      global_step,
      rng,
      hyperparameters=None,
      workload=None,
      param_types=None,
      loss_type=None,
      train_state=None,
      eval_results=None,
  ):
    """Updates the model parameters.

    Args:
      params: The current model parameters.
      model_state: The current state of the model.
      optimizer_state: The current state of the optimizer.
      batch: The current batch of data.
      global_step: The current training step.
      rng: The random number generator.
      hyperparameters: The hyperparameters for the training.
      workload: The workload being trained.
      param_types: The types of the parameters.
      loss_type: The type of loss function to use.
      train_state: The optional training state.
      eval_results: The optional evaluation results.

    Returns:
      A tuple containing:
        new_optimizer_state: Pytree of optimizer state.
        new_params: Pytree of model parameters.
        new_model_state: Pytree of model state.
    """
    del (
        workload,
        hyperparameters,
        param_types,
        loss_type,
        train_state,
        eval_results,
    )  # Unused
    grad_clip = self.hps.opt_hparams.get('grad_clip', None)
    # We pass the lr directly because the lr functions from sehedules.py
    # have numpy dependencies and can't be jitted.
    lr = self._lr_fn(global_step)
    jitted_update_fn = jax.jit(
        optax_update_params_helper,
        static_argnames=(
            'training_cost_fn',
            'optimizer_update_fn',
        ),
        donate_argnums=(0, 1, 2),
    )
    (
        new_optimizer_state,
        new_params,
        new_batch_stats,
        cost_value,
        grad,
        grad_norm,
        update_norm,
    ) = jitted_update_fn(
        params,
        model_state,
        optimizer_state,
        self._update_fn,
        batch,
        lr,
        rng,
        grad_clip,
        self.training_cost_fn,
    )

    self.eval_report_metrics.update(
        learning_rate=lr,
        grad_norm=grad_norm.item(),
        update_norm=update_norm.item(),
    )
    self._optimizer_state = new_optimizer_state

    return new_optimizer_state, new_params, new_batch_stats, cost_value, grad

  def init_optimizer_state(
      self,
      model=None,
      params=None,
      model_state=None,
      hyperparameters=None,
      rng=None,
  ):
    """Initializes the optimizer state.

    Args:
      model: The model being trained.
      params: The initial model parameters.
      model_state: The initial state of the model.
      hyperparameters: The hyperparameters for the training.
      rng: The random number generator.

    Returns:
      Optimizer state: Pytree of optimizer state.
    """
    del model, model_state, hyperparameters, rng  # Unused
    stretch_factor = 1
    if self.hps.get('total_accumulated_batch_size') is not None:
      stretch_factor = (
          self.hps.total_accumulated_batch_size // self.hps.batch_size
      )

    self._lr_fn = schedules.get_schedule_fn(
        self.hps.lr_hparams,
        max_training_updates=self.num_train_steps // stretch_factor,
        stretch_factor=stretch_factor,
    )

    optimizer_init_fn, optax_optimizer_update_fn = optimizers.get_optimizer(
        self.hps, self.model, batch_axis_name='batch'
    )
    # Wrapping init in jax.jit fuses per-parameter state creation ops into
    # a single compilation instead of compiling each one individually.
    optax_optimizer_state = jax.jit(optimizer_init_fn)(params)
    self._optimizer_state = optax_optimizer_state
    self._update_fn = optax_optimizer_update_fn
    return optax_optimizer_state

  # TODO(b/436634470): Consolidate this with the prepare_for_eval API
  def get_ema_eval_params(self, optimizer_state, params):
    """Extracts the exponential moving average (EMA) parameters from the optimizer state.

    Args:
      optimizer_state: The current state of the optimizer.
      params: The current model parameters.

    Returns:
      The EMA parameters.

    Raises:
      ValueError: If the EMA parameters cannot be extracted from the optimizer
        state.
    """
    del params  # Unused
    if isinstance(optimizer_state, optax.InjectStatefulHyperparamsState):
      eval_params = optimizer_state.inner_state[0][0].ema
    elif isinstance(
        optimizer_state, gradient_accumulator.GradientAccumulatorState
    ):
      eval_params = optimizer_state.base_state.inner_state[0][0].ema
    else:
      raise ValueError(
          'EMA computation should be the very first transformation in defined'
          ' kitchensink optimizer.'
      )
    return eval_params
