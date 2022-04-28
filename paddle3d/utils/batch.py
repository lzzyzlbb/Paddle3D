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

import numbers
from collections.abc import Sequence, Mapping

import numpy as np
import paddle

from paddle3d.object import DetObject, empty_object


def collate_fn(batch):
    """
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in sample}
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            # if not isinstance(sample, DetObject):
            #     raise RuntimeError(
            #         "fileds number not same among samples in a batch")

            maxnum = max([len(sample) for sample in iter(batch)])

            for sample in batch:
                sample_num = len(sample)

                if sample_num == maxnum:
                    continue

                sample += [empty_object] * (maxnum - sample_num)

        return [collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))
