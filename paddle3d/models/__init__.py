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

from .backbones.dla import *
from .detection.smoke.smoke import SMOKE
from .detection.smoke.smoke_coder import SMOKECoder
from .detection.smoke.smoke_predictor import SMOKEPredictor
from .detection.smoke.processor import PostProcessor
from .detection.smoke.processorhm import PostProcessorHm
from .losses.focal_loss import FocalLoss

from .backbones.resnet import ResNet
from .detection.caddn.caddn import CADDN
from .heads.dense_heads.anchor_head import AnchorHeadSingle
from .heads.class_heads.deeplabv3_head import DeepLabV3Head

from .optimizers import AdamWOnecycle

