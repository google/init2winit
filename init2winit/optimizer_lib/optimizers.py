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

"""Getter function for selecting optimizers."""

import copy

from absl import logging
import flax
from init2winit.model_lib.model_utils import ParameterType  # pylint: disable=g-importing-member
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import kitchen_sink
from init2winit.optimizer_lib import muon
from init2winit.optimizer_lib import online_newton_step
from init2winit.optimizer_lib import pax_adafactor
from init2winit.optimizer_lib import samuel
from init2winit.optimizer_lib import sharpness_aware_minimization
from init2winit.optimizer_lib import sla
from init2winit.optimizer_lib import utils
import jax
import jax.numpy as jnp
import optax










def _filtering(path, _) -> bool:
  """Filter to ensure that we inject/fetch lrs from 'InjectHyperparamsState'-like states."""
  if (
      (len(path) > 1)
      and isinstance(path[-2], optax.tree_utils.NamedTupleKey)
      and path[-2].name == 'hyperparams'
  ):
    return True
  else:
    return False


def inject_learning_rate(optimizer_state, lr):
  """Inject the given LR into any optimizer state that will accept it.

  We require that the optimizer state exposes an 'InjectHyperparamsState'-like
  interface, i.e., it should contain a `hyperparams` dictionary with a
  'learning_rate' key where the learning rate can be set. We need to do this
  to allow arbitrary (non-jittable) LR schedules.

  Args:
    optimizer_state: optimizer state returned by an optax optimizer
    lr: learning rate to inject

  Returns:
    new_optimizer_state
      optimizer state with the same structure as the input. The learning_rate
      entry in the state has been set to lr.
  """
  return optax.tree_utils.tree_set(
      optimizer_state, _filtering, learning_rate=lr
  )


def fetch_learning_rate(optimizer_state):
  """Fetch the LR from any optimizer state."""
  lrs_with_path = optax.tree_utils.tree_get_all_with_path(
      optimizer_state, 'learning_rate', _filtering
  )
  if not lrs_with_path:
    raise ValueError(f'No learning rate found in {optimizer_state}.')
  all_equal = all(
      jnp.array_equal(lr, lrs_with_path[0][1]) for _, lr in lrs_with_path
  )
  if all_equal:
    lr_array = lrs_with_path[0][1]
    return lr_array
  else:
    raise ValueError(
        'All learning rates in the optimizer state must be the same.'
        f'Found {lrs_with_path} in {optimizer_state}.'
    )


def sgd(learning_rate, weight_decay, momentum=None, nesterov=False):
  r"""A customizable gradient descent optimizer.

  NOTE: We apply weight decay **before** computing the momentum update.
  This is equivalent to applying WD after for heavy-ball momentum,
  but slightly different when using Nesterov accelleration. This is the same as
  how the Flax optimizers handle weight decay
  https://flax.readthedocs.io/en/latest/_modules/flax/optim/momentum.html.

  Args:
    learning_rate: The learning rate. Expected as the positive learning rate,
      for example `\alpha` in `w -= \alpha * u` (as opposed to `\alpha`).
    weight_decay: The weight decay hyperparameter.
    momentum: The momentum hyperparameter.
    nesterov: Whether or not to use Nesterov momentum.

  Returns:
    An optax gradient transformation that applies weight decay and then one of a
    {SGD, Momentum, Nesterov} update.
  """
  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      optax.sgd(
          learning_rate=learning_rate, momentum=momentum, nesterov=nesterov))


def get_optimizer(hps, model=None, batch_axis_name=None):
  """Constructs the optax optimizer from the given HParams.

  We use optax.inject_hyperparams to wrap the optimizer transformations that
  accept learning rates. This allows us to "inject" the learning rate at each
  step in a training loop by manually setting it in the optimizer_state,
  calculating it using whatever (Python or Jax) logic we want. This is why we
  set learning_rate=0.0 for all optimizers below. Note that all optax
  transformations returned from this function need to have
  `optax.inject_hyperparams` as the top level transformation.

  Args:
    hps: the experiment hyperparameters, as a ConfigDict.
    model: the model to be trained.
    batch_axis_name: the axis to pmap over.

  Returns:
    A tuple of the initialization and update functions returned by optax.
  """
  # We handle hps.l2_decay_factor in the training cost function base_model.py
  # and hps.weight_decay in the optimizer. It is almost certainly an error if
  # both are set.
  weight_decay = hps.opt_hparams.get('weight_decay', 0)
  assert hps.l2_decay_factor is None or weight_decay == 0.0

  opt_init = None
  opt_update = None
  # When set to True, the optimizer's update function will be called with
  # "grad_fn_params_tuple=(grad_fn, params)" instead of the `params`
  # argument, where grad_fn is the result of `jax.value_and_grad(opt_cost)`.
  optimizer_requires_grad_fn = False
  # When set to True, the optimizer's update function will be called with
  # "cost_fn_params_tuple=(cost_fn, params)" instead of the `params`
  # argument, where cost_fn is the model's cost function.
  optimizer_requires_cost_fn = False
  # When set to True, the optimizer's update function will be called with
  # an extra keyword argument value=cost_fn(params).
  optimizer_requires_value = False

  if hps.optimizer == 'sgd':
    opt_init, opt_update = utils.static_inject_hyperparams(sgd)(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=weight_decay,
    )
  elif hps.optimizer == 'muon':
    opt_init, opt_update = utils.static_inject_hyperparams(muon.scale_by_muon)(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=hps.opt_hparams.get('weight_decay', 0.01),
        beta=hps.opt_hparams.get('beta', 0.95),
        nesterov=hps.opt_hparams.get('nesterov', True),
        ns_coeffs=hps.opt_hparams.get('ns_coeffs', (3.4445, -4.7750, 2.0315)),
        ns_steps=hps.opt_hparams.get('ns_steps', 5),
        eps=hps.opt_hparams.get('eps', 1e-7),
        bias_correction=hps.opt_hparams.get('bias_correction', False),
    )
  elif hps.optimizer == 'diag_bubbles':
    opt_init, opt_update = utils.static_inject_hyperparams(
        lora_bubbles.diag_bubbles
    )(
        learning_rate=0.0,  # Manually injected on each train step.
        beta1=hps.opt_hparams.get('beta1', None),
        beta2=hps.opt_hparams.get('beta2', 0.999),
        eps=hps.opt_hparams.get('eps', 1e-8),
        precond_grad_clip=hps.opt_hparams.get('precond_grad_clip', None),
        nesterov=hps.opt_hparams.get('nesterov', False),
        bias_correction=hps.opt_hparams.get('bias_correction', True),
        weight_decay=hps.opt_hparams.get('weight_decay', 1e-4),
    )
  elif hps.optimizer == 'lora_bubbles':
    opt_init, opt_update = utils.static_inject_hyperparams(
        lora_bubbles.scale_by_lora_bubbles
    )(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=hps.opt_hparams.get('weight_decay', 0.01),
        beta1=hps.opt_hparams.get('beta1', 0.9),
        beta2=hps.opt_hparams.get('beta2', 0.999),
        nesterov=hps.opt_hparams.get('nesterov', True),
        eps=hps.opt_hparams.get('eps', 1e-7),
        lora_min_steps=hps.opt_hparams.get('lora_min_steps', 100),
        lora_update_steps=hps.opt_hparams.get('lora_update_steps', 20),
        lora_rank=hps.opt_hparams.get('lora_rank', 64),
        grad_rms_threshold=hps.opt_hparams.get('grad_rms_threshold', 10.0),
        precond_grad_clip=hps.opt_hparams.get('precond_grad_clip', None),
        bias_correction=hps.opt_hparams.get('bias_correction', True),
    )
  elif hps.optimizer == 'bubbles':
    opt_init, opt_update = utils.static_inject_hyperparams(
        lora_bubbles.scale_by_bubbles
    )(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=hps.opt_hparams.get('weight_decay', 0.01),
        beta1=hps.opt_hparams.get('beta1', 0.9),
        beta2=hps.opt_hparams.get('beta2', 0.999),
        nesterov=hps.opt_hparams.get('nesterov', True),
        min_steps=hps.opt_hparams.get('min_steps', 100),
        grad_rms_threshold=hps.opt_hparams.get('grad_rms_threshold', 10.0),
        precond_grad_clip=hps.opt_hparams.get('precond_grad_clip', None),
        bias_correction=hps.opt_hparams.get('bias_correction', True),
    )
  elif hps.optimizer == 'generic_multi_optimizer':
    param_type_to_optimizer_and_hparams = hps.opt_hparams[
        'param_type_to_optimizer_and_hparams'
    ]

    # With sweeps, we need to be able to modify the hparams at each leaf
    # We introduce a special dict `to_merge` that we merge into the hparams
    # of each leaf optimizer.
    hparams_to_merge = {}
    if 'to_merge' in hps.opt_hparams:
      hparams_to_merge = copy.deepcopy(hps.opt_hparams['to_merge'])
      del hps.opt_hparams['to_merge']

    param_type_to_optimizer_and_hparams = {
        int(k): v for k, v in param_type_to_optimizer_and_hparams.items()
    }

    if ParameterType.DEFAULT.value not in param_type_to_optimizer_and_hparams:
      raise ValueError(
          f'Fallback default optimizer not found in param_type_to_grad_tx.'
          f' Please add a fallback optimizer to param_type_to_grad_tx ='
          f' {param_type_to_optimizer_and_hparams}'
      )
    param_type_to_grad_tx = {}

    for param_type, opt_hparams in param_type_to_optimizer_and_hparams.items():
      hps_copy = copy.deepcopy(hps)
      del hps_copy.opt_hparams['param_type_to_optimizer_and_hparams']
      if hparams_to_merge:
        opt_hparams.opt_hparams.update(hparams_to_merge)
      hps_copy.update(opt_hparams)
      logging.info('HPS_COPY %s', hps_copy)
      param_type_to_grad_tx[param_type] = (
          optax.GradientTransformation(*get_optimizer(hps_copy, model))
      )

    param_to_type = model.params_types
    param_to_type = jax.tree_util.tree_map(lambda x: x.value, param_to_type)

    param_types = jax.tree_util.tree_leaves(param_to_type)

    for param_type in param_types:
      if param_type not in param_type_to_grad_tx:
        param_type_to_grad_tx[param_type] = param_type_to_grad_tx[
            ParameterType.DEFAULT.value
        ]

    del param_type_to_grad_tx[ParameterType.DEFAULT.value]

    opt_init, opt_update = optax.multi_transform(
        param_type_to_grad_tx, param_labels=param_to_type
    )
  elif hps.optimizer == 'multiple_optimizer':
    hps_network = hps.opt_hparams['hps_network']
    hps_last_layer = hps.opt_hparams['hps_last_layer']

    # l2_decay_factor is checked, but we
    # want it to be None for the suboptimizers.
    hps_network['l2_decay_factor'] = None
    hps_last_layer['l2_decay_factor'] = None

    network_optimizer = optax.GradientTransformation(
        *get_optimizer(hps_network))
    last_layer_optimizer = optax.GradientTransformation(
        *get_optimizer(hps_last_layer))

    opt_init, opt_update = online_newton_step.multiple_optimizer(
        last_layer_name=hps.opt_hparams['last_layer_name'],
        network_optimizer=network_optimizer,
        last_layer_optimizer=last_layer_optimizer,
        last_layer_base_lr=hps.opt_hparams['last_layer_base_lr'],
        base_lr=hps.lr_hparams['base_lr'])
  elif hps.optimizer == 'online_newton_step':
    opt_init, opt_update = utils.static_inject_hyperparams(
        online_newton_step.online_newton_step)(
            learning_rate=0.0,  # Manually injected on each train step.
            alpha=hps.opt_hparams['alpha'],
            weight_decay=weight_decay)
  elif hps.optimizer == 'diag_ons':
    opt_init, opt_update = utils.static_inject_hyperparams(
        online_newton_step.diag_ons)(
            learning_rate=1.0,  # Set to 1.0 to use as a last layer optimizer.
            weight_decay=weight_decay,
            b1=hps.opt_hparams['beta1'],
            b2=hps.opt_hparams['beta2'])
  elif hps.optimizer == 'momentum' or hps.optimizer == 'nesterov':
    opt_init, opt_update = utils.static_inject_hyperparams(sgd)(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=weight_decay,
        momentum=hps.opt_hparams['momentum'],
        nesterov=(hps.optimizer == 'nesterov'))
  elif hps.optimizer == 'tearfree':
    sketch_size = hps.opt_hparams.get('sketchy_rank')
    if sketch_size is not None and sketch_size > 0:
      opts = tearfree_sketchy.Options(
          update_freq=hps.opt_hparams['update_preconditioners_freq'],
          second_moment_decay=hps.opt_hparams['beta2'],
          rank=sketch_size,
          epsilon=hps.opt_hparams.get('matrix_epsilon', 1e-16),
          relative_epsilon=hps.opt_hparams.get(
              'matrix_relative_epsilon', False
          ),
          add_ggt=hps.opt_hparams.get('add_ggt', False),
          memory_alloc=hps.opt_hparams.get('memory_alloc', None),
          ekfac_svd=hps.opt_hparams.get('ekfac_svd', False),
      )
      opts = {
          'sketchy_options': opts,
          'shampoo_options': None,
          'second_order_type': tearfree_second_order.SecondOrderType.SKETCHY,
      }
    else:
      opts = tearfree_shampoo.Options(
          second_moment_decay=hps.opt_hparams['beta2'],
          block_size=hps.opt_hparams['block_size'],
          update_statistics_freq=hps.opt_hparams['update_statistics_freq'],
          update_preconditioners_freq=hps.opt_hparams[
              'update_preconditioners_freq'
          ],
      )
      opts = {
          'shampoo_options': opts,
          'second_order_type': tearfree_second_order.SecondOrderType.SHAMPOO,
      }

    opt_init, opt_update = utils.static_inject_hyperparams(
        tearfree_optimizer.tearfree
    )(
        learning_rate=0.0,
        options=tearfree_optimizer.TearfreeOptions(
            grafting_options=tearfree_grafting.Options(
                grafting_type=tearfree_grafting.GraftingType(
                    hps.opt_hparams['graft_type']
                ),
                second_moment_decay=hps.opt_hparams['beta2'],
                start_preconditioning_step=hps.opt_hparams[
                    'start_preconditioning_step'
                ],
                skip_preconditioning_any_dim_gt=hps.opt_hparams.get(
                    'skip_precond_dim', 4096
                ),
                multiply_by_parameter_scale=hps.opt_hparams.get(
                    'param_scale', True
                ),
                clipping_threshold=hps.opt_hparams.get(
                    'clipping_threshold', 1.0
                ),
            ),
            second_order_options=tearfree_second_order.Options(
                merge_dims=hps.opt_hparams['merge_dims'],
                **opts,
            ),
            momentum_options=tearfree_momentum.Options(
                momentum_decay=hps.opt_hparams['beta1'],
                weight_decay=hps.opt_hparams['weight_decay'],
                weight_decay_after_momentum=hps.opt_hparams[
                    'weight_decay_after_momentum'
                ],
                nesterov=hps.opt_hparams['nesterov'],
                ema=hps.opt_hparams['ema'],
            ),
        ),
    )

  elif hps.optimizer == 'distributed_shampoo':
    raise ValueError(
        'distributed_shampoo implementation is broken in init2winit after we'
        ' migrated to jit, do not use it for the time being.'
    )
    # pylint: enable=line-too-long
  elif hps.optimizer == 'adam':
    opt_init, opt_update = utils.static_inject_hyperparams(optax.adamw)(
        learning_rate=0.0,  # Manually injected on each train step.
        b1=hps.opt_hparams['beta1'],
        b2=hps.opt_hparams['beta2'],
        eps=hps.opt_hparams['epsilon'],
        weight_decay=weight_decay)
  elif hps.optimizer == 'adafactor':
    opt_init, opt_update = utils.static_inject_hyperparams(optax.adafactor)(
        learning_rate=0.0,
        min_dim_size_to_factor=hps.opt_hparams['min_dim_size_to_factor'],
        decay_rate=hps.opt_hparams['adafactor_decay_rate'],
        decay_offset=hps.opt_hparams['decay_offset'],
        multiply_by_parameter_scale=hps
        .opt_hparams['multiply_by_parameter_scale'],
        clipping_threshold=hps.opt_hparams['clipping_threshold'],
        momentum=hps.opt_hparams['momentum'],
        weight_decay_rate=weight_decay,
        eps=hps.opt_hparams['epsilon'],
        factored=hps.opt_hparams['factored'],
        # NOTE(dsuo): we provide this wiring, but specifying a weight decay
        # mask in a config file / serializing properly is not completely
        # straightforward.
        weight_decay_mask=hps.opt_hparams.get('weight_decay_mask', None),
    )
  elif hps.optimizer == 'polyak_sgd':
    base_opt = utils.static_inject_hyperparams(
        optax.polyak_sgd, injectable_args=('scaling',)
    )(
        scaling=0.0,  # Manually injected on each train step.
        f_min=hps.opt_hparams['f_min'],
        max_learning_rate=hps.opt_hparams['max_learning_rate'],
        eps=hps.opt_hparams['eps'],
    )
    # Enables modifying scaling through learning_rate to comply with the
    # fetching/injecting semantics of init2winit pipeline
    opt_init, opt_update = utils.overwrite_hparam_names(
        base_opt, scaling='learning_rate'
    )
    optimizer_requires_value = True
  elif hps.optimizer == 'dadapt_adamw':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.dadapt_adamw
    )(
        learning_rate=0.0,
        betas=(hps.opt_hparams['beta1'], hps.opt_hparams['beta2']),
        eps=hps.opt_hparams['eps'],
        estim_lr0=hps.opt_hparams['estim_lr0'],
        weight_decay=weight_decay,
    )
  elif hps.optimizer == 'prodigy':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.prodigy
    )(
        learning_rate=0.0,
        betas=(hps.opt_hparams['beta1'], hps.opt_hparams['beta2']),
        eps=hps.opt_hparams['eps'],
        estim_lr0=hps.opt_hparams['estim_lr0'],
        estim_lr_coef=hps.opt_hparams['estim_lr_coef'],
        weight_decay=weight_decay,
        safeguard_warmup=hps.opt_hparams['safeguard_warmup'],
    )
  elif hps.optimizer == 'cocob':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.cocob
    )(
        learning_rate=0.0,
        weight_decay=weight_decay,
        alpha=hps.opt_hparams['alpha'],
        eps=hps.opt_hparams['eps'],
    )
  elif hps.optimizer == 'momo':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.momo
    )(
        learning_rate=0.0,
        beta=hps.opt_hparams['beta'],
        lower_bound=hps.opt_hparams['lower_bound'],
        weight_decay=weight_decay,
        adapt_lower_bound=hps.opt_hparams['adapt_lower_bound'],
    )
    optimizer_requires_value = True
  elif hps.optimizer == 'momo_adam':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.momo_adam
    )(
        learning_rate=0.0,
        b1=hps.opt_hparams['beta1'],
        b2=hps.opt_hparams['beta2'],
        eps=hps.opt_hparams['eps'],
        lower_bound=hps.opt_hparams['lower_bound'],
        weight_decay=weight_decay,
        adapt_lower_bound=hps.opt_hparams['adapt_lower_bound'],
    )
    optimizer_requires_value = True
  elif hps.optimizer == 'dog':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.dog
    )(
        learning_rate=0.0,
        reps_rel=hps.opt_hparams['reps_rel'],
        eps=hps.opt_hparams['eps'],
        weight_decay=hps.opt_hparams['weight_decay'],
    )
  elif hps.optimizer == 'dowg':
    opt_init, opt_update = utils.static_inject_hyperparams(
        optax.contrib.dowg
    )(
        learning_rate=0.0,
        eps=hps.opt_hparams['eps'],
        weight_decay=hps.opt_hparams['weight_decay'],
    )

  elif hps.optimizer == 'kitchen_sink':
    opt_init, opt_update = utils.static_inject_hyperparams(
        kitchen_sink.kitchen_sink)(
            learning_rate=0.0, config=hps.opt_hparams)
  elif hps.optimizer == 'samuel':
    opt_init, opt_update = samuel.from_hparams(hps.opt_hparams)

  batch_size = hps.batch_size
  accum_size = hps.get('total_accumulated_batch_size')
  if accum_size is not None and accum_size != batch_size:
    # We do not synchronize batch norm stats across devices, so if there is no
    # virtual_batch_size set in the hyperparameters, the per-core batch size
    # (hps.batch_size // num_hosts) is used as the virtual batch size.
    virtual_batch_size = hps.get('virtual_batch_size', None)
    if virtual_batch_size is None:
      virtual_batch_size = hps.batch_size // jax.process_count()
    opt_init, opt_update = gradient_accumulator.accumulate_gradients(
        per_step_batch_size=hps.batch_size,
        total_batch_size=hps.total_accumulated_batch_size,
        virtual_batch_size=virtual_batch_size,
        base_opt_init_fn=opt_init,
        base_opt_update_fn=opt_update)

  if hps.opt_hparams.get('use_sam', False):
    opt_init, opt_update = (
        sharpness_aware_minimization.sharpness_aware_minimization(
            rho=hps.opt_hparams['rho'],
            grad_clip=hps.get('grad_clip', None),
            base_opt_init_fn=opt_init,
            base_opt_update_fn=opt_update,
        )
    )
    optimizer_requires_grad_fn = True
  elif hps.opt_hparams.get('use_pal', False):
    opt_init, opt_update = (
        parabolic_approximation_line_search.parabolic_approximation_line_search(
            mu=hps.opt_hparams['mu'],
            alpha=hps.opt_hparams['alpha'],
            s_max=hps.opt_hparams['s_max'],
            start_step=hps.opt_hparams['start_step'],
            stop_step=hps.opt_hparams['stop_step'],
            batch_axis_name=batch_axis_name,
            base_opt_init_fn=opt_init,
            base_opt_update_fn=opt_update,
        )
    )
    optimizer_requires_cost_fn = True
  elif hps.opt_hparams.get('use_mechanic', False):
    opt_init, opt_update = optax.contrib.mechanize(
        weight_decay=hps.opt_hparams['mech_weight_decay'],
        eps=hps.opt_hparams['mech_eps'],
        s_init=hps.opt_hparams['mech_s_init'],
        num_betas=hps.opt_hparams['mech_num_betas'],
        base_optimizer=optax.GradientTransformationExtraArgs(
            opt_init, opt_update
        ),
    )

  if opt_init is None or opt_update is None:
    raise NotImplementedError(
        'Optimizer {} not implemented'.format(hps.optimizer)
    )
  return opt_init, _wrap_update_fn(
      hps.optimizer,
      opt_update,
      send_grad_fn=optimizer_requires_grad_fn,
      send_cost_fn=optimizer_requires_cost_fn,
      send_value=optimizer_requires_value,
  )


def _wrap_update_fn(
    opt_name,
    opt_update,
    send_grad_fn=False,
    send_cost_fn=False,
    send_value=False,
):
  """Wraps the optimizer update function to have the same function signature.

  Args:
    opt_name: The optimizer name.
    opt_update: The optimizer update function.
    send_grad_fn: When set to True will pass `value_and_grad` to the optimizer's
      update function.
    send_cost_fn: When set to True will pass `cost_fn` to the optimizer's update
      function. This must not be set to True if `send_grad_fn` is True (note
      that grad_fn already returns the cost value).
    send_value: When set to True will pass the current value to the optimizer's
      update function.

  Returns:
    A wrapped optimizer update function.
  """
  del opt_name

  def update_fn(grads,
                optimizer_state,
                params,
                batch=None,
                batch_stats=None,
                cost_fn=None,
                grad_fn=None,
                value=None):
    del batch, batch_stats
    if send_grad_fn and send_cost_fn:
      # Note that `value_and_grad` already returns the cost, so there is no need
      # to set both send_grad_fn and send_cost_fn to True.
      raise ValueError('send_grad_fn and send_cost_fn must not both be True.')
    if send_grad_fn:
      return opt_update(
          grads, optimizer_state, grad_fn_params_tuple=(grad_fn, params))
    elif send_cost_fn:
      return opt_update(
          grads, optimizer_state, cost_fn_params_tuple=(cost_fn, params))
    elif send_value:
      return opt_update(
          grads,
          optimizer_state,
          params=params,
          value=value,
          value_fn=cost_fn,
          grad=grad_fn,
      )
    return opt_update(grads, optimizer_state, params=params)

  if not utils.requires_gradient_aggregation(opt_update):
    return utils.no_cross_device_gradient_aggregation(update_fn)
  return update_fn
