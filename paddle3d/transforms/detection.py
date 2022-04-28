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

import cv2
import numpy as np
from PIL import Image

from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.apis import manager


@manager.TRANSFORMS.add_component
class Normalize(TransformABC):
    """
    """

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))

        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, data: dict, label: Optional[dict] = None):
        """
        """
        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
        std = np.array(self.std)[:, np.newaxis, np.newaxis]
        data.image = F.normalize(data.image, mean, std)

        return data, label


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(TransformABC):
    """
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, data: dict, label: Optional[dict] = None):
        if np.random.random() < self.prob:

            data.image = F.horizontal_flip(data.image)
            h, w, c = data.image.shape

            # flip camera intrinsics
            if 'K' in label:
                label['K'][0, 2] = w - label['K'][0, 2] - 1

            # flip bbox
            for obj in label.get('objects', []):
                obj.horizontal_flip(image_width=w)

        return data, label
