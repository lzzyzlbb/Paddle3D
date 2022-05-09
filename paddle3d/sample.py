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

from typing import Generic, List, Optional

import numpy as np


class _EasyDict(dict):
    def __getattr__(self, key: str):
        if key in self:
            return self[key]

        return None

    def __setattr__(self, key: str, value: Generic):
        self[key] = value


class SampleMeta(_EasyDict):
    """
    """
    __slots__ = ["camera_intrinsic", "image_format", "image_reader", "id"]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Sample(_EasyDict):
    """
    """
    _VALID_MODALITIES = ["image", "lidar", "radar", "multimodal", "multiview"]

    def __init__(self, path: str, modality: str):
        if modality not in self._VALID_MODALITIES:
            raise ValueError

        self.path = path
        self.modality = modality.lower()
        self.meta = SampleMeta()
        self.data = None
        self.sweeps = []
        self.bboxes_2d = None
        self.bboxes_3d = None
        self.labels = None
