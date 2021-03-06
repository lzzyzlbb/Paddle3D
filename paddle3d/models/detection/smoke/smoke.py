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
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.utils.logger import logger
from paddle3d.models.detection.smoke.smoke_loss import SMOKELossComputation


@manager.MODELS.add_component
class SMOKE(nn.Layer):
    """
    """

    def __init__(self, backbone, head, post_process=None):
        super().__init__()
        self.backbone = backbone
        self.heads = head
        self.post_process = post_process
        self.init_weight()
        self.loss_computation = SMOKELossComputation(
            depth_ref=[28.01, 16.32],
            dim_ref=[[3.88, 1.63, 1.53], [1.78, 1.70, 0.58], [0.88, 1.73,
                                                              0.67]],
            reg_loss="DisL1",
            loss_weight=[1., 10.],
            max_objs=50)

    def forward(self, data, targets=None):
        images = data["image"]
        features = self.backbone(images)
        predictions = self.heads(features)
        if not self.training:
            return self.post_process(predictions, targets)

        loss = self.loss_computation(predictions, targets)
        return {'loss': loss}

    def init_weight(self, bias_lr_factor=2):
        for sublayer in self.sublayers():
            if hasattr(sublayer, 'bias') and sublayer.bias is not None:
                sublayer.bias.optimize_attr['learning_rate'] = bias_lr_factor
