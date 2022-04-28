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

from typing import Optional, Tuple

import numpy as np

from paddle3d.utils.camera import roty


class Point2D(dict):
    """
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: "Point2D"):
        self.x += other.x
        self.y += other.y

    def __truediv__(self, num):
        self.x /= num
        self.y /= num

    def horizontal_flip(self, image_width):
        self.x = image_width - self.x

    def __array__(self):
        return np.array([self.x, self.y])


class Point3D(dict):
    """
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Point3D"):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __truediv__(self, num):
        self.x /= num
        self.y /= num
        self.z /= num

    def horizontal_flip(self):
        self.x *= -1

    def __array__(self):
        return np.array([self.x, self.y, self.z])


class BBox2D(dict):
    """
    """

    def __init__(self, *, pt1: Tuple[float, float], pt2: Tuple[float, float]):
        self.point1 = Point2D(*pt1)
        self.point2 = Point2D(*pt2)

    @property
    def center(self) -> Point2D:
        return (self.point1 + self.point2) / 2

    @property
    def height(self) -> float:
        return self.point2.y - self.point1.y

    @property
    def width(self) -> float:
        return self.point2.x - self.point1.x

    def horizontal_flip(self, image_width):
        self.point1.horizontal_flip(image_width)
        self.point2.horizontal_flip(image_width)
        self.point1, self.point2 = self.point2, self.point1
        self.point1.y, self.point2.y = self.point2.y, self.point1.y

    def __array__(self):
        pt1 = np.array(self.point1)
        pt2 = np.array(self.point2)
        return np.array([pt1, pt2]).flatten()


class BBox3D(dict):
    """
    """

    def __init__(self, pt: Tuple[float, float, float], w: float, h: float,
                 l: float, theta: float):
        self.theta = theta
        self.length = l
        self.height = h
        self.width = w
        self.pt = Point3D(*pt)

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(self.R, corners)

        corners_3d += np.array(self.pt)[:, np.newaxis]
        self.points = [Point3D(*corners_3d[:, i]) for i in range(8)]

    @property
    def center(self) -> Point3D:
        return (self.points[0] + self.points[6]) / 2

    @property
    def R(self) -> np.ndarray:
        return roty(self.theta)

    def horizontal_flip(self):
        for i in range(8):
            self.points[i].horizontal_flip()

        self.pt.horizontal_flip()
        self.theta *= -1

    def __array__(self):
        return np.append(self.pt,
                         [self.width, self.height, self.length, self.theta])


class DetObject(dict):
    """
    """

    def __init__(self,
                 label: str,
                 label_id: int,
                 bbox2d: Optional[BBox2D] = None,
                 bbox3d: Optional[BBox3D] = None):
        self.bbox2d = bbox2d
        self.bbox3d = bbox3d
        self.label = label
        self.label_id = label_id

    def horizontal_flip(self, image_width=None):
        if self.bbox2d:
            self.bbox2d.horizontal_flip(image_width)

        if self.bbox3d:
            self.bbox3d.horizontal_flip()


empty_box2d = BBox2D(pt1=(0., 0.), pt2=(0., 0.))
empty_box3d = BBox3D((0., 0., 0.), 0., 0., 0., 0.)
empty_object = DetObject(0., 0., empty_box2d, empty_box3d)
