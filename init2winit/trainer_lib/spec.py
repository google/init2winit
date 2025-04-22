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

"""MLPerf™ Algorithmic Efficiency API."""

import abc
import enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


class LossType(enum.Enum):
  SOFTMAX_CROSS_ENTROPY = 0
  SIGMOID_CROSS_ENTROPY = 1
  MEAN_SQUARED_ERROR = 2
  CTC_LOSS = 3
  MEAN_ABSOLUTE_ERROR = 4


class ForwardPassMode(enum.Enum):
  TRAIN = 0
  EVAL = 1


class ParameterType(enum.Enum):
  """Types of model parameters."""
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM_SCALE = 3
  BATCH_NORM_BIAS = 4
  LAYER_NORM_SCALE = 5
  LAYER_NORM_BIAS = 6
  EMBEDDING = 7
  ATTENTION_Q = 8
  ATTENTION_K = 9
  ATTENTION_V = 10
  ATTENTION_OUT = 11
  ATTENTION_QKV = 12  # This is used for implementations that fuse QKV together.
  ATTENTION_KV = 13  # This is used for implementations that fuse KV together.
  # We sometimes need to split this out because otherwise fused models will have
  # a different number of biases.
  ATTENTION_BIAS = 14


# Of course, Tensor knows its shape and dtype.
# Tensor = Union[jnp.array, np.array, tf.Tensor, torch.Tensor, ...]
Tensor = Any


# Define this so that if using pytree iteration utilities, can iterate over the
# model shapes pytree without iterating over the shape tuples.
class ShapeTuple:

  def __init__(self, shape_tuple):
    self.shape_tuple = shape_tuple

  def __repr__(self):
    return f'ShapeTuple({self.shape_tuple})'

  def __eq__(self, other):
    return self.shape_tuple == other.shape_tuple


Shape = Union[Tuple[int],
              Tuple[int, int],
              Tuple[int, int, int],
              Tuple[int, int, int, int],
              ShapeTuple]
ParameterShapeTree = Dict[str, Dict[str, Shape]]

# If necessary, these can be zipped together easily given they have the same
# structure, to get an iterator over pairs of leaves.
ParameterKey = str
# Dicts can be arbitrarily nested.
ParameterContainer = Union[Dict[ParameterKey, Dict[ParameterKey, Tensor]]]
ParameterTypeTree = Dict[ParameterKey, Dict[ParameterKey, ParameterType]]

RandomState = Any  # Union[jax.random.PRNGKey, int, bytes, ...]

OptimizerState = Union[Dict[str, Any], Tuple[Any, Any]]
Hyperparameters = Any
Timing = int
Steps = int

# BN EMAs.
ModelAuxiliaryState = Any
ModelInitState = Tuple[ParameterContainer, ModelAuxiliaryState]


class Workload(metaclass=abc.ABCMeta):
  """Base class for workloads."""

  def __init__(self, *args, **kwargs) -> None:
    del args
    del kwargs
    self._param_shapes: Optional[ParameterShapeTree] = None
    self._param_types: Optional[ParameterTypeTree] = None
    self._eval_iters: Dict[str, Iterator[Dict[str, Any]]] = {}
    self.metrics_logger = None

  @property
  @abc.abstractmethod
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""

  @property
  @abc.abstractmethod
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""

  @property
  def param_shapes(self):
    """The shapes of the parameters in the workload model."""
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    """The types of the parameters in the workload model."""
    if self._param_types is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_types!')
    return self._param_types


class TrainingCompleteError(Exception):
  pass


# Training algorithm track submission functions, to be filled in by the
# submitter.

InitOptimizerFn = Callable[[
    Workload,
    ParameterContainer,
    ModelAuxiliaryState,
    Hyperparameters,
    RandomState
], OptimizerState]


# pylint: disable=unused-argument
def init_optimizer_state(workload: Workload,
                         model_params: ParameterContainer,
                         model_state: ModelAuxiliaryState,
                         hyperparameters: Hyperparameters,
                         rng: RandomState) -> OptimizerState:
  # return initial_optimizer_state
  pass


UpdateReturn = Tuple[OptimizerState, ParameterContainer, ModelAuxiliaryState]
UpdateParamsFn = Callable[[
    Workload,
    ParameterContainer,
    ParameterTypeTree,
    ModelAuxiliaryState,
    Hyperparameters,
    Dict[str, Tensor],
    LossType,
    OptimizerState,
    List[Tuple[int, float]],
    int,
    RandomState,
    Optional[Dict[str, Any]]
],
                          UpdateReturn]


# Each call to this function is considered a "step".
# Can raise a TrainingCompleteError if it believes it has achieved the goal and
# wants to end the run and receive a final free eval. It will not be restarted,
# and if has not actually achieved the goal then it will be considered as not
# achieved the goal and get an infinite time score. Most submissions will likely
# wait until the next free eval and not use this functionality.
def update_params(workload: Workload,
                  current_param_container: ParameterContainer,
                  current_params_types: ParameterTypeTree,
                  model_state: ModelAuxiliaryState,
                  hyperparameters: Hyperparameters,
                  batch: Dict[str, Tensor],
                  loss_type: LossType,
                  optimizer_state: OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: RandomState,
                  train_state: Optional[Dict[str, Any]] = None) -> UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  pass


PrepareForEvalFn = Callable[[
    Workload,
    ParameterContainer,
    ParameterTypeTree,
    ModelAuxiliaryState,
    Hyperparameters,
    LossType,
    OptimizerState,
    List[Tuple[int, float]],
    int,
    RandomState
], UpdateReturn]


# Prepare model and optimizer for evaluation.
def prepare_for_eval(workload: Workload,
                     current_param_container: ParameterContainer,
                     current_params_types: ParameterTypeTree,
                     model_state: ModelAuxiliaryState,
                     hyperparameters: Hyperparameters,
                     loss_type: LossType,
                     optimizer_state: OptimizerState,
                     eval_results: List[Tuple[int, float]],
                     global_step: int,
                     rng: RandomState) -> UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  pass


DataSelectionFn = Callable[[
    Workload,
    Iterator[Dict[str, Any]],
    OptimizerState,
    ParameterContainer,
    LossType,
    Hyperparameters,
    int,
    RandomState
], Tuple[Tensor, Tensor]]


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: Workload,
                   input_queue: Iterator[Dict[str, Any]],
                   optimizer_state: OptimizerState,
                   current_param_container: ParameterContainer,
                   model_state: ModelAuxiliaryState,
                   hyperparameters: Hyperparameters,
                   global_step: int,
                   rng: RandomState) -> Dict[str, Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.

  Args:
    workload: The workload being trained.
    input_queue: An iterator over the training data.
    optimizer_state: The current optimizer state.
    current_param_container: The current model parameters.
    model_state: The current model state (e.g., for batch norm).
    hyperparameters: The current hyperparameters.
    global_step: The current training step.
    rng: The current random number generator state.

  Returns:
    A batch of training data.
  """
  # return next(input_queue)
  pass


def get_batch_size(workload_name: str) -> int:
  """Return the global batch size to use for a given workload."""
  pass
