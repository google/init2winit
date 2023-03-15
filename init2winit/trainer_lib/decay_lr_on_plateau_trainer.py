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

"""Trainer that decays learning rate when target metric plateaus."""
import itertools
import time

from absl import logging
from init2winit import checkpoint
from init2winit import utils
from init2winit.trainer_lib import base_trainer
from init2winit.trainer_lib import trainer
from init2winit.trainer_lib import trainer_utils
import jax

_GRAD_CLIP_EPS = 1e-6


class DecayLROnPlateauTrainer(base_trainer.BaseTrainer):
  """Trainer that decays learning rate when target metric plateaus."""

  def train(self):
    """All training logic.

    The only side-effects are:
      - Initiailizing self._time_at_prev_eval_end to the current time
      - Initiailizing self._prev_eval_step to the current step

    Yields:
      metrics: A dictionary of all eval metrics from the given epoch.
    """
    # NOTE: the initialization RNG should *not* be per-host, as this will create
    # different sets of weights per host. However, all other RNGs should be
    # per-host.
    # TODO(znado,gilmer,gdahl): implement replicating the same initialization
    # across hosts.
    rng, init_rng = jax.random.split(self._rng)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, data_rng = jax.random.split(rng)
    rng, callback_rng = jax.random.split(rng)

    if jax.process_index() == 0:
      logging.info('Let the training begin!')
      logging.info('Dataset input shape: %r', self._hps.input_shape)
      logging.info('Hyperparameters: %s', self._hps)

    self._setup_and_maybe_restore(init_rng, data_rng, callback_rng,
                                  trainer.update)

    if jax.process_index() == 0:
      trainer_utils.log_message(
          'Starting training!', self._logging_pool, self._xm_work_unit)

    # Start at the resumed step and continue until we have finished the number
    # of training steps. If building a dataset iterator using a tf.data.Dataset,
    # in the case of a batch size that does not evenly divide the training
    # dataset size, if using `ds.batch(..., drop_remainer=True)` on the training
    # dataset then the final batch in this iterator will be a partial batch.
    # However, if `drop_remainer=False`, then this iterator will always return
    # batches of the same size, and the final batch will have elements from the
    # start of the (num_epochs + 1)-th epoch.
    if self._hps.get('use_grain'):
      train_iter = itertools.islice(
          self._dataset.train_iterator_fn(), self._num_train_steps
      )
    else:
      train_iter = itertools.islice(
          self._dataset.train_iterator_fn(),
          self._global_step,
          self._num_train_steps,
      )


    train_iter = trainer_utils.prefetch_input_pipeline(
        train_iter, self._hps.num_device_prefetches)

    start_time = time.time()
    start_step = self._global_step

    # NOTE(dsuo): record timestamps for run_time since we don't have a duration
    # that we can increment as in the case of train_time.
    self._time_at_prev_eval_end = start_time
    self._prev_eval_step = self._global_step

    for _ in range(start_step, self._num_train_steps):
      with jax.profiler.StepTraceAnnotation('train',
                                            step_num=self._global_step):
        # NOTE(dsuo): to properly profile each step, we must include batch
        # creation in the StepTraceContext (as opposed to putting `train_iter`
        # directly in the top-level for loop).
        batch = next(train_iter)

        if (self._global_step in self._checkpoint_steps
            and jax.process_index() == 0):
          self._save(self._checkpoint_dir, max_to_keep=None)
        lr = self._lr_fn(self._global_step)
        # It looks like we are reusing an rng key, but we aren't.
        # TODO(gdahl): Make it more obvious that passing rng is safe.
        # TODO(gdahl,gilmer,znado): investigate possibly merging the member
        # variable inputs/outputs of this function into a named tuple.
        (self._optimizer_state, self._params, self._batch_stats,
         self._sum_train_cost,
         self._metrics_state, self._grad_norm) = self._update_pmapped(
             self._optimizer_state, self._params, self._batch_stats,
             self._metrics_state, batch, self._global_step, lr, rng,
             self._local_device_indices, self._sum_train_cost)
        self._global_step += 1
        # TODO(gdahl, gilmer): consider moving this test up.
        # NB: Since this test is after we increment self._global_step, having 0
        # in eval_steps does nothing.
        if trainer_utils.should_eval(
            self._global_step, self._eval_frequency, self._eval_steps):
          report = self._eval(lr, start_step, start_time)
          yield report
          if self._check_early_stopping(report):
            return
          self._lr_fn.decay(report)

    # Always log and checkpoint on host 0 at the end of training.
    # If we moved where in the loop body evals happen then we would not need
    # this test.
    if self._prev_eval_step != self._num_train_steps:
      report = self._eval(lr, start_step, start_time)
      yield report
    # To make sure the last checkpoint was correctly saved.
    checkpoint.wait_for_checkpoint_save()
