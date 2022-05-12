# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle

from paddle3d.utils.logger import logger


def load_pretrained_model(model: paddle.nn.Layer, pretrained_model_path: str):

    if os.path.exists(pretrained_model_path):
        para_state_dict = paddle.load(pretrained_model_path)
        model_state_dict = model.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            if k not in para_state_dict:
                logger.warning("{} is not in pretrained model".format(k))
            elif list(para_state_dict[k].shape) != list(
                    model_state_dict[k].shape):
                logger.warning(
                    "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                    .format(k, para_state_dict[k].shape,
                            model_state_dict[k].shape))
            else:
                model_state_dict[k] = para_state_dict[k]
                num_params_loaded += 1

        model.set_dict(model_state_dict)
        logger.info("There are {}/{} variables loaded into {}.".format(
            num_params_loaded, len(model_state_dict), model.__class__.__name__))

    else:
        raise ValueError(
            'The pretrained model directory is not Found: {}'.format(
                pretrained_model_path))
