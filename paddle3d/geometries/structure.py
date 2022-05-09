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

from json import JSONEncoder

import numpy as np


class StructureEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _Structure):
            return obj.tolist()
        return super().default(obj)


class _Structure(np.ndarray):
    """
    """

    def __new__(cls, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        data = data.astype(np.float32)

        obj = np.asarray(data).view(cls)
        return obj
