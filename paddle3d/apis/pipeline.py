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

import paddle


def trainning_step(model: paddle.nn.Layer,
                   optimizer: paddle.optimizer.Optimizer, data: dict,
                   label: dict):
    model.train()
    outputs = model(data, label)

    loss = outputs['loss']
    # model backward
    loss.backward()

    optimizer.step()
    model.clear_gradients()

    return loss


def validation_step(model: paddle.nn.Layer, data: dict, label: dict):
    model.eval()

    with paddle.no_grad():
        outputs = model(data, label)


def predict(model: paddle.nn.Layer, data):
    outputs = model(data)
