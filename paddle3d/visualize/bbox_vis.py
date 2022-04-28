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

from typing import Tuple

import cv2
import numpy as np

from paddle3d.utils.camera import project
from paddle3d.object import BBox2D, BBox3D


def draw_bbox2d(img: np.ndarray,
                bbox2d: BBox2D,
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 1):
    """
    """
    pt1 = np.array(bbox2d.point1).astype('int')
    pt2 = np.array(bbox2d.point2).astype('int')

    return cv2.rectangle(
        img.astype('uint8'), pt1, pt2, color=color, thickness=thickness)


def draw_bbox3d(img: np.ndarray,
                bbox3d: BBox3D,
                K: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 0),
                thickness: int = 1):
    """
    """
    points = np.array([np.array(p) for p in bbox3d.points]).transpose((1, 0))
    points = project(points, K).transpose((1, 0))

    lines = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for idx1, idx2 in lines:
        img = cv2.line(
            img,
            points[idx1].astype('int').tolist(),
            points[idx2].astype('int').tolist(),
            color=color,
            thickness=thickness)
    return img
