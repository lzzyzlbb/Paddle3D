# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle3d.apis import manager
from paddle3d.models.layers import ASPPModule, ConvBNReLU, SeparableConvBNReLU

@manager.HEADS.add_component
class DeepLabV3Head(nn.Layer):
    """
    The DeepLabV3Head implementation based on PaddlePaddle.
    Args:
        Please Refer to DeepLabV3PHead above.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, align_corners, pretrained=None):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels,
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True,
            bias_attr=False)
        
        self.conv_bn_relu = ConvBNReLU(
            in_channels=aspp_out_channels,
            out_channels=aspp_out_channels,
            kernel_size=3,
            bias_attr=False)

        self.cls = nn.Conv2D(
            in_channels=aspp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        x = self.conv_bn_relu(x)
        logit = self.cls(x)
        
        return logit

