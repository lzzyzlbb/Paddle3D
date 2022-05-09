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

import numpy as np

from paddle3d.geometries.structure import _Structure


class PointCloud(_Structure):
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError

        if data.shape[1] < 3:
            raise ValueError

    def scale(self, factor: float):
        """
        """
        ...

    def translate(self, translation: np.ndarray):
        """
        """
        self[..., :3] = self[..., :3] + translation

    def rotate(self, rotation: np.ndarray):
        """
        """
        self[..., :3] = (rotation @ self[..., :3].T).T

    def flip(self, axis: int):
        """
        """
