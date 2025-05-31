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

"""Module containing hyperparameters, metadata and dataset getter for Wikitext-103 dataset."""

import functools
from init2winit.dataset_lib import wikitext103
from init2winit.dataset_lib import wikitext103_input_pipeline
from ml_collections.config_dict import config_dict

SPM_TOKENIZER_VOCAB_SIZE = wikitext103_input_pipeline.SPM_TOKENIZER_VOCAB_SIZE
SPM_TOKENIZER_VOCAB_PATH = wikitext103_input_pipeline.SPM_TOKENIZER_VOCAB_PATH
PAD_ID = -1
get_wikitext103 = functools.partial(wikitext103.get_wikitext103, pad_id=PAD_ID)

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        sequence_length=128,
        max_target_length=128,
        max_eval_target_length=128,
        eval_sequence_length=128,
        input_shape=(128,),
        output_shape=(SPM_TOKENIZER_VOCAB_SIZE,),
        tokenizer='sentencepiece',
        tokenizer_vocab_path=SPM_TOKENIZER_VOCAB_PATH,
        vocab_size=SPM_TOKENIZER_VOCAB_SIZE,
        train_size=800210,  # TODO(kasimbeg): Update this
    )
)


METADATA = {
    'apply_one_hot_in_loss': True,
    'shift_inputs': True,
    'causal': True,
    'pad_token': -1,
}
