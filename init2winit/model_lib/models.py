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

"""Registry for the available models we can train."""

from init2winit.model_lib import adabelief_densenet
from init2winit.model_lib import adabelief_resnet
from init2winit.model_lib import adabelief_vgg
from init2winit.model_lib import autoencoder
from init2winit.model_lib import conformer
from init2winit.model_lib import convolutional_autoencoder
from init2winit.model_lib import deepspeech
from init2winit.model_lib import dlrm
from init2winit.model_lib import fully_connected
from init2winit.model_lib import gnn
from init2winit.model_lib import local_attention_transformer
from init2winit.model_lib import lstm_lm
from init2winit.model_lib import max_pooling_cnn
from init2winit.model_lib import mlperf_resnet
from init2winit.model_lib import nqm
from init2winit.model_lib import resnet
from init2winit.model_lib import simple_cnn
from init2winit.model_lib import transformer_lm
from init2winit.model_lib import transformer_stu_lm
from init2winit.model_lib import transformer_stu_tensordot_lm
from init2winit.model_lib import unet
from init2winit.model_lib import vit
from init2winit.model_lib import wide_resnet
from init2winit.model_lib import xformer_translate
from init2winit.model_lib import xformer_translate_binary
from init2winit.model_lib import xformer_translate_mlc_variant


_ALL_MODELS = {
    'fully_connected':
        (fully_connected.FullyConnectedModel, fully_connected.DEFAULT_HPARAMS),
    'simple_cnn': (simple_cnn.SimpleCNNModel, simple_cnn.DEFAULT_HPARAMS),
    'max_pooling_cnn':
        (max_pooling_cnn.MaxPoolingCNNModel, max_pooling_cnn.DEFAULT_HPARAMS),
    'wide_resnet': (wide_resnet.WideResnetModel, wide_resnet.DEFAULT_HPARAMS),
    'resnet': (resnet.ResnetModel, resnet.DEFAULT_HPARAMS),
    'adabelief_densenet': (adabelief_densenet.AdaBeliefDensenetModel,
                           adabelief_densenet.DEFAULT_HPARAMS),
    'adabelief_resnet': (adabelief_resnet.AdaBeliefResnetModel,
                         adabelief_resnet.DEFAULT_HPARAMS),
    'adabelief_vgg':
        (adabelief_vgg.AdaBeliefVGGModel, adabelief_vgg.DEFAULT_HPARAMS),
    'autoencoder': (autoencoder.AutoEncoderModel, autoencoder.DEFAULT_HPARAMS),
    'convolutional_autoencoder':
        (convolutional_autoencoder.ConvAutoEncoderModel,
         convolutional_autoencoder.DEFAULT_HPARAMS),
    'fake_resnet':
        (mlperf_resnet.FakeModel, mlperf_resnet.FAKE_MODEL_DEFAULT_HPARAMS),
    'mlperf_resnet':
        (mlperf_resnet.ResnetModelMLPerf, mlperf_resnet.MLPERF_DEFAULT_HPARAMS),
    'transformer':
        (transformer_lm.TransformerLM1B, transformer_lm.DEFAULT_HPARAMS),
    'transformer_stu': (
        transformer_stu_lm.TransformerLM1B,
        transformer_stu_lm.DEFAULT_HPARAMS,
    ),
    'transformer_stu_tensordot': (
        transformer_stu_tensordot_lm.TransformerLM1B,
        transformer_stu_tensordot_lm.DEFAULT_HPARAMS,
    ),
    'conformer': (conformer.ConformerModel, conformer.DEFAULT_HPARAMS),
    'mlcommons_conformer': (conformer.MLCommonsConformerModel,
                            conformer.MLCOMMONS_DEFAULT_HPARAMS),
    'local_attention_transformer':
        (local_attention_transformer.LocalAttentionTransformer,
         local_attention_transformer.DEFAULT_HPARAMS),
    'deepspeech': (deepspeech.DeepSpeechModel, deepspeech.DEFAULT_HPARAMS),
    'mlcommons_deepspeech': (deepspeech.MLCommonsDeepSpeechModel,
                             deepspeech.MLCOMMONS_DEFAULT_HPARAMS),
    'nqm': (nqm.NQM, nqm.DEFAULT_HPARAMS),
    'xformer_translate': (xformer_translate.TransformerTranslate,
                          xformer_translate.DEFAULT_HPARAMS),
    'mlcommons_xformer_translate':
        (xformer_translate.MLCommonsTransformerTranslate,
         xformer_translate.MLCOMMONS_DEFAULT_HPARAMS),
    'xformer_translate_mlc_variant':
        (xformer_translate_mlc_variant.TransformerTranslate,
         xformer_translate_mlc_variant.DEFAULT_HPARAMS),
    'xformer_translate_binary': (xformer_translate_binary.TransformerTranslate,
                                 xformer_translate_binary.DEFAULT_HPARAMS),
    'gnn': (gnn.GNNModel, gnn.DEFAULT_HPARAMS),
    'dlrm': (dlrm.DLRMModel, dlrm.DEFAULT_HPARAMS),
    'dlrm_resnet': (dlrm.DLRMResNetModel, dlrm.DEFAULT_RESNET_HPARAMS),
    'vit': (vit.ViTModel, vit.DEFAULT_HPARAMS),
    'unet': (unet.UNetModel, unet.DEFAULT_HPARAMS),
    'lstm': (lstm_lm.LSTMModel, lstm_lm.DEFAULT_HPARAMS),
}


def get_model(model_name):
  """Get the corresponding model class based on the model string.

  API:
  model_builder, hps = get_model("fully_connected")
  ... modify/parse hparams
  model = model_builder(hps, num_classes)

  Args:
    model_name: (str) e.g. fully_connected.

  Returns:
    The model architecture (currently a flax Model) along with its
    default hparams.
  Raises:
    ValueError if model is unrecognized.
  """
  try:
    return _ALL_MODELS[model_name][0]
  except KeyError:
    raise ValueError('Unrecognized model: {}'.format(model_name)) from None


def get_model_hparams(model_name):
  """Get the corresponding model hyperparameters based on the model string.

  Args:
    model_name: (str) e.g. fully_connected.

  Returns:
    The model architecture (currently a flax Model) along with its
    default hparams.
  Raises:
    ValueError if model is unrecognized.
  """
  try:
    return _ALL_MODELS[model_name][1]
  except KeyError:
    raise ValueError('Unrecognized model: {}'.format(model_name)) from None
