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

"""Implementation of the SAMUEL optimizer.

Paper: https://arxiv.org/pdf/2203.01400.pdf
"""

import copy
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple

from init2winit.optimizer_lib.utils import static_inject_hyperparams
import jax
import jax.numpy as jnp
import optax


class SamuelState(NamedTuple):
  inner_state: NamedTuple
  expert_weights: jnp.array
  key: jnp.array
  current_expert: int = 0
  step: int = 0


def samuel(
    optimizers: List[str],
    hps: List[Dict[str, Any]],
    mw_etas: jnp.array,
    seed: int = 0,
    train_loss: float = 0.,
    learning_rate: float = 0.,
):
  """Samuel optimizer.

  NOTES
  - This implementation assumes each host is an expert (i.e., holds a copy of
    the model). As a consequence, we must modify the input and training
    pipelines to forgot data parallelism across hosts and limit to only the
    local devices available to a given host.
  - We synchronize after each batch. This is not always necessary and can be
    a point of future performance optimization.

  TODO(dsuo): add LR schedules for optimizers.

  Args:
    optimizers: list of strings indicating optax optimizers.
    hps: list of hps for each optimizer.
    mw_etas: list of multiplicative weight etas.
    seed: initial jax random seed.
    train_loss: train loss to be injected at update time.
    learning_rate: for compatability, but ignored for now.

  Returns:
    samuel optimizer
  """
  del learning_rate

  num_experts = len(optimizers)
  mw_etas = jnp.array(mw_etas)

  if num_experts != jax.process_count():
    raise ValueError(
        'This implementation of SAMUEL requires the number of optimizers to be '
        'equal to the number of hosts (one host per expert).')

  optimizer = optimizers[jax.process_index()]
  hps = hps[jax.process_index()]

  optimizer = getattr(optax, optimizer)(**hps)

  def init_fn(params):
    return SamuelState(
        inner_state=optimizer.init(params),
        expert_weights=jnp.repeat(mw_etas, num_experts, axis=1),
        # NOTE(dsuo): each init gives the same key, but given the changing
        # params for each model, this is not an issue.
        key=jax.random.PRNGKey(seed),
    )

  def update_fn(updates, state, params):
    del params

    key, subkey = jax.random.split(state.key)

    # Compute updates based on inner optimizer
    updates, inner_state = optimizer.update(updates, state.inner_state)

    prob = state.expert_weights.sum(axis=0) / state.expert_weights.sum()

    # NOTE(dsuo): we rely on jax determinism for each host to behave the same.
    current_expert = jax.random.choice(subkey, jnp.arange(prob.size), p=prob)

    # Synchronize train_losses across hosts.
    # NOTE(dsuo): since we are already insider a pmap, we can't use
    # jax.experimental.multihost_utils.
    # NOTE(dsuo): train_losses is of shape (jax.process_count(),).
    train_losses = jax.lax.all_gather(train_loss, 'batch').reshape(
        jax.process_count(), jax.local_device_count())[:, 0]

    # Compute loss regret and update expert weights.
    loss_regret = train_losses.at[current_expert].get() - train_losses
    expert_weights = state.expert_weights * jnp.exp(mw_etas * loss_regret)

    state = SamuelState(
        inner_state=inner_state,
        expert_weights=expert_weights,
        key=key,
        current_expert=current_expert,
        step=state.step + 1,
    )
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def from_hparams(opt_hparams):
  """Create SAMUEL optimizer from init2winit."""
  opt_hparams_optimizers = opt_hparams['optimizers']

  optimizers = []
  hps = []
  index = 0
  while str(index) in opt_hparams_optimizers:
    hparams = opt_hparams_optimizers[str(index)]
    hp = hparams.get('hps', {})

    for h in copy.deepcopy(hp).keys():
      if 'one_minus_' in h:
        hp[h.replace('one_minus_', '')] = 1 - hp[h]
        del hp[h]

    optimizers.append(hparams.get('optimizer'))
    hps.append(hp)
    index += 1

  return static_inject_hyperparams(samuel)(
      optimizers=optimizers, hps=hps, **opt_hparams['args'])
