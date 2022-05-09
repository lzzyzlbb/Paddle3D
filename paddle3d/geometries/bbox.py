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

from typing import List

import numpy as np

from paddle3d.utils.camera import roty
from paddle3d.geometries.structure import _Structure


class BBoxes2D(_Structure):
    """
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError

        if data.shape[1] != 4:
            raise ValueError

    def scale(self, factor: float):
        ...

    def translate(self, translation: np.ndarray):
        ...

    def rotate(self, rotation: np.ndarray):
        ...

    def horizontal_flip(self, image_width: float):
        ...


class BBoxes3D(_Structure):
    """
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError

        if data.shape[1] != 7:
            raise ValueError

    @property
    def corners(self):
        # shape --> n
        h, w, l = self[:, 3:6].T
        b = h.shape[0]

        # shape --> n * 8
        x_corners = np.array([[1, 1, -1, -1, 1, 1, -1, -1]]).repeat(b, axis=0)
        y_corners = np.array([[1, 1, 1, 1, -1, -1, -1, -1]]).repeat(b, axis=0)
        z_corners = np.array([[1, -1, -1, 1, 1, -1, -1, 1]]).repeat(b, axis=0)

        # shape --> n * 3 * 8
        x_corners = (l / 2 * x_corners.T).T[:, np.newaxis, :]
        y_corners = (h / 2 * y_corners.T).T[:, np.newaxis, :]
        z_corners = (w / 2 * z_corners.T).T[:, np.newaxis, :]
        corners = np.concatenate([x_corners, y_corners, z_corners], axis=1)

        # shape --> n * 3 * 3
        R = roty(self[:, 6])

        corners = R @ corners
        centers = self[:, 0:3][:, :, np.newaxis]
        corners += centers

        return corners.transpose(0, 2, 1)

    def scale(self, factor: float):
        ...

    def translate(self, translation: np.ndarray):
        self[..., :3] = self[..., :3] + translation

    def rotate(self, rotation: np.ndarray):
        self[..., :3] = (rotation @ self[..., :3].T).T

    def horizontal_flip(self):
        ...
