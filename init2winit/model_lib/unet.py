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

"""Jax / Flax implementation of FastMRI U-Net.

Original implementation:
github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py

Training:
github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/unet_module.py

Data:
github.com/facebookresearch/fastMRI/tree/main/fastmri/data
"""

import flax.linen as nn
from init2winit.model_lib import base_model
from init2winit.model_lib import model_utils
import jax
import jax.numpy as jnp
from ml_collections import config_dict


# NOTE(dsuo): We use the Kitchen Sink optimizer to match the RMSProp
# implementation found in the reference FastMRI U-Net code. Specifically,
# epsilon in optax's scale_by_rms places its epsilon inside the square root,
# whereas the reference code epsilon outside.
opt_hparams = {
    'weight_decay': 0.0,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-8,
}

# NOTE(dsuo): This lives here because decay_events / decay_factors is too large
# to pass via the config file.
_FASTMRI_TRAIN_SIZE = 34742
_FASTMRI_VALID_SIZE = 7135

batch_size = 8
num_epochs = 50
steps_per_epoch = int(_FASTMRI_TRAIN_SIZE / batch_size)
num_train_steps = num_epochs * steps_per_epoch
lr_gamma = 0.1
lr_step_size = 40 * steps_per_epoch
decay_events = list(range(lr_step_size, num_train_steps, lr_step_size))
decay_factors = [lr_gamma] * len(decay_events)
decay_factors = [
    decay_factor**i
    for decay_factor, i in zip(decay_factors, range(1,
                                                    len(decay_events) + 1))
]

lr_hparams = {
    'schedule': 'piecewise_constant',
    'base_lr': 1e-3,
    'decay_events': decay_events,
    'decay_factors': decay_factors
}

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        dropout_rate=0.0,
        activation='leaky_relu',
        optimizer='adam',
        opt_hparams=opt_hparams,
        lr_hparams=lr_hparams,
        l2_decay_factor=None,
        batch_size=batch_size,
        rng_seed=-1,
        model_dtype='float32',
        grad_clip=None,
        total_accumulated_batch_size=None,
        normalizer='unet_instance_norm',
    ))


def _compute_stats(x, axes):
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(jnp.square(x), axes)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - jnp.square(mean))
  return mean, var


def _normalize(x, axes, mean, var, epsilon):
  stats_shape = list(x.shape)
  for axis in axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  y = x - mean
  mul = jnp.sqrt(var + epsilon)
  y /= mul
  return y


def _simple_instance_norm2d(x, axes, epsilon=1e-5):
  mean, var = _compute_stats(x, axes)
  return _normalize(x, axes, mean, var, epsilon)


class UNet(nn.Module):
  """Jax / Flax implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.

    out_chans: Number of channels in the output to the U-Net model.
    chans: Number of output channels of the first convolution layer.
    num_pool_layers: Number of down-sampling and up-sampling layers.
    drop_prob: Dropout probability.
  """
  out_chans: int
  chans: int = 32
  num_pool_layers: int = 4
  drop_prob: float = 0.0
  activation: str = 'leaky_relu'
  normalizer: str = 'unet_instance_norm'

  @nn.compact
  def __call__(self, x, train=True):
    down_sample_layers = [
        ConvBlock(self.chans, self.drop_prob, self.activation, self.normalizer)
    ]

    ch = self.chans
    for _ in range(self.num_pool_layers - 1):
      down_sample_layers.append(
          ConvBlock(ch * 2, self.drop_prob, self.activation, self.normalizer)
      )
      ch *= 2
    conv = ConvBlock(ch * 2, self.drop_prob, self.activation, self.normalizer)

    up_conv = []
    up_transpose_conv = []
    for _ in range(self.num_pool_layers - 1):
      up_transpose_conv.append(TransposeConvBlock(ch, self.activation))
      up_conv.append(
          ConvBlock(ch, self.drop_prob, self.activation, self.normalizer)
      )
      ch //= 2

    up_transpose_conv.append(TransposeConvBlock(ch, self.activation))
    up_conv.append(
        ConvBlock(ch, self.drop_prob, self.activation, self.normalizer)
    )

    final_conv = nn.Conv(self.out_chans, kernel_size=(1, 1), strides=(1, 1))

    stack = []
    output = jnp.expand_dims(x, axis=-1)

    # apply down-sampling layers
    for layer in down_sample_layers:
      output = layer(output, train)
      stack.append(output)
      output = nn.avg_pool(output, window_shape=(2, 2), strides=(2, 2))

    output = conv(output, train)

    # apply up-sampling layers
    for transpose_conv, conv in zip(up_transpose_conv, up_conv):
      downsample_layer = stack.pop()
      output = transpose_conv(output)

      # reflect pad on the right/botton if needed to handle odd input dimensions
      padding_right = 0
      padding_bottom = 0
      if output.shape[-2] != downsample_layer.shape[-2]:
        padding_right = 1  # padding right
      if output.shape[-3] != downsample_layer.shape[-3]:
        padding_bottom = 1  # padding bottom

      if padding_right or padding_bottom:
        padding = ((0, 0), (0, padding_bottom), (0, padding_right), (0, 0))
        output = jnp.pad(output, padding, mode='reflect')

      output = jnp.concatenate((output, downsample_layer), axis=-1)
      output = conv(output, train)

    output = final_conv(output)

    return output.squeeze(-1)


class ConvBlock(nn.Module):
  """A Convolutional Block.

  out_chans: Number of channels in the output.
  drop_prob: Dropout probability.
  """
  out_chans: int
  drop_prob: float
  activation: str
  normalizer: str

  @nn.compact
  def __call__(self, x, train=True):
    """Forward function.

    Note: Pytorch is NCHW and jax/flax is NHWC.

    Args:
        x: Input 4D tensor of shape `(N, H, W, in_chans)`.
        train: deterministic or not (use init2winit naming).

    Returns:
        jnp.array: Output tensor of shape `(N, H, W, out_chans)`.
    """
    x = nn.Conv(
        features=self.out_chans,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False)(
            x)
    # InstanceNorm2d was implemented with no learnable params in reference code
    # so this is a simple normalization along channels
    if self.normalizer == 'unet_instance_norm':
      x = _simple_instance_norm2d(x, (1, 2))
    elif self.normalizer == 'layer_norm':
      # Layer Norm typically normalizes across channels as well
      x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))
    if self.activation == 'leaky_relu':
      x = jax.nn.leaky_relu(x, negative_slope=0.2)
    elif self.activation in model_utils.ACTIVATIONS:
      x = model_utils.ACTIVATIONS[self.activation](x)
    else:
      raise ValueError('Unsupported activation: {}'.format(self.activation))
    # Ref code uses dropout2d which applies the same mask for the entire channel
    # Replicated by using broadcast dims to have the same filter on HW
    x = nn.Dropout(
        self.drop_prob, broadcast_dims=(1, 2), deterministic=not train)(
            x)
    x = nn.Conv(
        features=self.out_chans,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False)(
            x)
    # InstanceNorm2d was implemented with no learnable params in reference code
    # so this is a simple normalization along channels
    if self.normalizer == 'unet_instance_norm':
      x = _simple_instance_norm2d(x, (1, 2))
    elif self.normalizer == 'layer_norm':
      # Layer Norm typically normalizes across channels as well
      x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
    else:
      raise ValueError('Unsupported normalizer: {}'.format(self.normalizer))
    if self.activation == 'leaky_relu':
      x = jax.nn.leaky_relu(x, negative_slope=0.2)
    elif self.activation in model_utils.ACTIVATIONS:
      x = model_utils.ACTIVATIONS[self.activation](x)
    else:
      raise ValueError('Unsupported activation: {}'.format(self.activation))
    x = nn.Dropout(
        self.drop_prob, broadcast_dims=(1, 2), deterministic=not train)(
            x)

    return x


class TransposeConvBlock(nn.Module):
  """A Transpose Convolutional Block.

  out_chans: Number of channels in the output.
  """
  out_chans: int
  activation: str

  @nn.compact
  def __call__(self, x):
    """Forward function.

    Args:
        x: Input 4D tensor of shape `(N, H, W, in_chans)`.

    Returns:
        jnp.array: Output tensor of shape `(N, H*2, W*2, out_chans)`.
    """
    x = nn.ConvTranspose(
        self.out_chans, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(
            x)
    x = _simple_instance_norm2d(x, (1, 2))
    if self.activation == 'leaky_relu':
      x = jax.nn.leaky_relu(x, negative_slope=0.2)
    elif self.activation in model_utils.ACTIVATIONS:
      x = model_utils.ACTIVATIONS[self.activation](x)
    else:
      raise ValueError('Unsupported activation: {}'.format(self.activation))

    return x


class UNetModel(base_model.BaseModel):
  """U-Net model for fastMRI knee single-coil data."""

  def evaluate_batch(self, params, batch_stats, batch):
    """Evaluates metrics under self.metrics_name on the given_batch."""
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = self.flax_module.apply(
        variables, batch['inputs'], mutable=False, train=False)
    targets = batch['targets']

    # map the dict values (which are functions) to function(targets, logits)
    weights = batch.get('weights')  # Weights might not be defined.
    eval_batch_size = targets.shape[0]
    if weights is None:
      weights = jnp.ones(eval_batch_size)

    # We don't use CLU's `mask` argument here, we handle it ourselves through
    # `weights`.
    return self.metrics_bundle.single_from_model_output(
        logits=logits,
        targets=targets,
        weights=weights,
        mean=batch.get('mean'),
        std=batch.get('std'),
        volume_max=batch.get('volume_max'),
        axis_name='batch')

  def build_flax_module(self):
    """Unet implementation."""
    return UNet(
        out_chans=self.hps.out_chans,
        chans=self.hps.chans,
        num_pool_layers=self.hps.num_pool_layers,
        drop_prob=self.hps.dropout_rate,
        activation=self.hps.activation,
        normalizer=self.hps.normalizer)

  def get_fake_inputs(self, hps):
    """Helper method solely for the purpose of initializing the model."""
    dummy_inputs = [
        jnp.zeros((hps.batch_size, *hps.input_shape), dtype=hps.model_dtype)
    ]
    return dummy_inputs
