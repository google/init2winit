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

"""Getter function for selecting optimizers."""

from absl import logging
import flax
from init2winit.model_lib.model_utils import ParameterType  # pylint: disable=g-importing-member
from init2winit.optimizer_lib import gradient_accumulator
from init2winit.optimizer_lib import kitchen_sink
from init2winit.optimizer_lib import online_newton_step
from init2winit.optimizer_lib import pax_adafactor
from init2winit.optimizer_lib import samuel
from init2winit.optimizer_lib import sharpness_aware_minimization
from init2winit.optimizer_lib import utils
from init2winit.optimizer_lib.hessian_free import CGIterationTrackingMethod
from init2winit.optimizer_lib.hessian_free import hessian_free
import jax
import optax







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

  if hps.optimizer == 'sgd':
    opt_init, opt_update = utils.static_inject_hyperparams(sgd)(
        learning_rate=0.0,  # Manually injected on each train step.
        weight_decay=weight_decay)
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
    if hps.opt_hparams.get('frequent_directions', False):
      statistics_compute_steps = hps.opt_hparams[
          'preconditioning_compute_steps']
    else:
      statistics_compute_steps = hps.opt_hparams['statistics_compute_steps']
    # pylint: disable=line-too-long
    opt_init, opt_update = utils.static_inject_hyperparams(
        distributed_shampoo.distributed_shampoo
    )(
        learning_rate=0.0,
        block_size=hps.opt_hparams['block_size'],
        beta1=hps.opt_hparams['beta1'],
        beta2=hps.opt_hparams['beta2'],
        diagonal_epsilon=hps.opt_hparams['diagonal_epsilon'],
        matrix_epsilon=hps.opt_hparams['matrix_epsilon'],
        weight_decay=hps.opt_hparams['weight_decay'],
        start_preconditioning_step=hps
        .opt_hparams['start_preconditioning_step'],
        preconditioning_compute_steps=hps
        .opt_hparams['preconditioning_compute_steps'],
        decay_preconditioning_compute_steps=hps
        .opt_hparams.get('decay_preconditioning_compute_steps', False),
        end_preconditioning_compute_steps=hps
        .opt_hparams.get('end_preconditioning_compute_steps', None),
        statistics_compute_steps=statistics_compute_steps,
        best_effort_shape_interpretation=hps
        .opt_hparams['best_effort_shape_interpretation'],
        nesterov=hps.opt_hparams['nesterov'],
        exponent_override=hps.opt_hparams['exponent_override'],
        batch_axis_name=batch_axis_name,
        graft_type=hps.opt_hparams['graft_type'],
        num_devices_for_pjit=hps.opt_hparams['num_devices_for_pjit'],
        shard_optimizer_states=hps.opt_hparams['shard_optimizer_states'],
        best_effort_memory_usage_reduction=hps
        .opt_hparams['best_effort_memory_usage_reduction'],
        inverse_failure_threshold=hps.opt_hparams['inverse_failure_threshold'],
        moving_average_for_momentum=hps
        .opt_hparams['moving_average_for_momentum'],
        skip_preconditioning_dim_size_gt=hps
        .opt_hparams['skip_preconditioning_dim_size_gt'],
        relative_matrix_epsilon=hps.opt_hparams.get('relative_matrix_epsilon',
                                                    True),
        clip_by_scaled_gradient_norm=hps
        .opt_hparams['clip_by_scaled_gradient_norm'],
        merge_small_dims_block_size=hps.opt_hparams.get(
            'merge_small_dims_block_size', 4096),
        generate_fd_metrics=hps.opt_hparams.get('generate_fd_metrics', False),
        compression_rank=hps.opt_hparams.get('compression_rank', 0),
        frequent_directions=hps.opt_hparams.get('frequent_directions', False),
        average_grad=hps.opt_hparams.get('average_grad', False),
        eigh=hps.opt_hparams.get('eigh', False),
        skip_preconditioning_rank_lt=hps.opt_hparams.get(
            'skip_preconditioning_rank_lt', 1),
        decoupled_learning_rate=hps.opt_hparams.get('decoupled_learning_rate',
                                                    True),
        decoupled_weight_decay=hps.opt_hparams.get('decoupled_weight_decay',
                                                   False),
        generate_training_metrics=hps.opt_hparams.get(
            'generate_training_metrics', True),
        reuse_preconditioner=hps.opt_hparams.get('reuse_preconditioner', False),
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
        base_opt_update_fn=opt_update,
        batch_axis_name=batch_axis_name)

  if hps.opt_hparams.get('use_sam', False):
    opt_init, opt_update = sharpness_aware_minimization.sharpness_aware_minimization(
        rho=hps.opt_hparams['rho'],
        grad_clip=hps.get('grad_clip', None),
        batch_axis_name=batch_axis_name,
        base_opt_init_fn=opt_init,
        base_opt_update_fn=opt_update,
    )
  elif hps.opt_hparams.get('use_pal', False):
    opt_init, opt_update = parabolic_approximation_line_search.parabolic_approximation_line_search(
        mu=hps.opt_hparams['mu'],
        alpha=hps.opt_hparams['alpha'],
        s_max=hps.opt_hparams['s_max'],
        start_step=hps.opt_hparams['start_step'],
        stop_step=hps.opt_hparams['stop_step'],
        batch_axis_name=batch_axis_name,
        base_opt_init_fn=opt_init,
        base_opt_update_fn=opt_update
    )

  if opt_init is None or opt_update is None:
    raise NotImplementedError('Optimizer {} not implemented'.format(
        hps.optimizer))
  return opt_init, _wrap_update_fn(hps.optimizer, opt_update,
                                   hps.opt_hparams.get('use_sam', False),
                                   hps.opt_hparams.get('use_pal', False))


def _wrap_update_fn(opt_name, opt_update, use_sam=False, use_pal=False):
  """Wraps the optimizer update function to have the same function signiture.

  Args:
    opt_name: the optimizer name.
    opt_update: the optimizer update function.
    use_sam: flag to use sharpness aware minimization updates.
    use_pal: flag to use parabolic approximation line search updates.

  Returns:
    A wrapped optimizer update function.
  """

  def update_fn(grads,
                optimizer_state,
                params,
                batch=None,
                batch_stats=None,
                cost_fn=None,
                grad_fn=None):
    if opt_name == 'hessian_free':
      variables = {'params': params}
      if batch_stats is not None:
        variables['batch_stats'] = batch_stats
      return opt_update(grads, optimizer_state, params=(variables, batch))
    if use_sam:
      return opt_update(
          grads, optimizer_state, grad_fn_params_tuple=(grad_fn, params))
    elif use_pal:
      return opt_update(
          grads, optimizer_state, cost_fn_params_tuple=(cost_fn, params))
    return opt_update(grads, optimizer_state, params=params)

  return update_fn
