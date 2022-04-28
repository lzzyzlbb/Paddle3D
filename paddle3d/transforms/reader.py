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

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from paddle3d.apis import manager
from paddle3d.transforms.base import TransformABC


@manager.TRANSFORMS.add_component
class LoadImage(TransformABC):
    """
    """
    _READER_MAPPER = {"cv2": cv2.imread, "pillow": Image.open}

    def __init__(self,
                 to_chw: bool = True,
                 to_rgb: bool = True,
                 reader: str = "cv2"):
        if not reader in self._READER_MAPPER.keys():
            raise ValueError

        self.reader = reader
        self.to_rgb = to_rgb
        self.to_chw = to_chw

    def __call__(self, data: dict, label: Optional[dict] = None):
        """
        """
        if not "image" in data:
            raise ValueError

        data["image"] = self._READER_MAPPER[self.reader](data["image"])

        data["image_reader"] = self.reader
        data["image_format"] = "bgr" if self.reader == "cv2" else "rgb"

        if self.reader == "cv2" and self.to_rgb:
            data["image"] = cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB)
            data["image_format"] = "rgb"

        # if self.to_chw:
        #     data["image"] = data["image"].transpose((2, 0, 1))

        return data, label
