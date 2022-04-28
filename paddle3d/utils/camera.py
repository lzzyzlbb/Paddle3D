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


def roty(theta: float) -> np.ndarray:
    """
    """
    c = np.cos(theta, dtype='float32')
    s = np.sin(theta, dtype='float32')
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project(points: np.ndarray, K: np.ndarray):
    """
    """
    points = np.matmul(K, points)
    points = points[:2, :] / points[2:, :]
    return points
