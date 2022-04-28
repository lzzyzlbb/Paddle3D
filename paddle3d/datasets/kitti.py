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

import csv
import os
from collections import namedtuple
from typing import List, Union

import pandas
import numpy as np
import paddle
from easydict import EasyDict

import paddle3d.transforms as T
from paddle3d.transforms import TransformABC
from paddle3d.apis import manager
from paddle3d.object import BBox3D, BBox2D, DetObject

_kitti_label_fields = [
    'type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax',
    'dh', 'dw', 'dl', 'lx', 'ly', 'lz', 'ry'
]


class KittiObject(DetObject):
    _CLASS_MAP = {'Car': 0, 'Cyclist': 1, 'Pedestrian': 2}

    def __init__(self, *args):
        bbox2d = BBox2D(pt1=args[4:6], pt2=args[6:8])
        bbox3d = BBox3D(
            pt=args[11:14], w=args[9], h=args[8], l=args[10], theta=args[14])
        super().__init__(
            label=args[0],
            label_id=self._CLASS_MAP[args[0]],
            bbox2d=bbox2d,
            bbox3d=bbox3d)


class KittiDetDataset(paddle.io.Dataset):
    """
    """
    CLASSES = ['Car', 'Cyclist', 'Pedestrian']

    def __init__(self,
                 dataset_root: str,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 with_intrinsics: bool = False):
        super().__init__()
        self.dataset_root = dataset_root
        self.mode = mode.lower()

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms
        self.with_intrinsics = with_intrinsics

        if self.mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval' or 'test', but got {}."
                .format(self.mode))

        # parse camera intrinsics from calibration table
        txtfile = '000000.txt'
        with open(os.path.join(self.calib_dir, txtfile)) as csv_file:
            reader = list(csv.reader(csv_file, delimiter=' '))
            K = [float(i) for i in reader[2][1:]]
            K = np.array(K, dtype=np.float32).reshape(3, 4)
            self.K = K[:3, :3]

        # get file list
        with open(self.imagesets_path) as file:
            self.data = file.read().split('\n')[:-1]

    def __len__(self):
        return len(self.data)

    @property
    def is_test_mode(self) -> bool:
        return self.mode == 'test'

    @property
    def base_dir(self) -> str:
        dirname = 'testing' if self.is_test_mode else 'training'
        return os.path.join(self.dataset_root, dirname)

    @property
    def label_dir(self) -> str:
        return os.path.join(self.base_dir, 'label_2')

    @property
    def calib_dir(self) -> str:
        return os.path.join(self.base_dir, 'calib')

    @property
    def imagesets_path(self) -> str:
        return os.path.join(self.base_dir, 'ImageSets',
                            '{}.txt'.format(self.mode))

    def load_annotation(self, index: int) -> List[KittiObject]:
        filename = '{}.txt'.format(self.data[index])
        objs = []

        with open(os.path.join(self.label_dir, filename), 'r') as csv_file:
            df = pandas.read_csv(
                csv_file, sep=' ', header=None, names=_kitti_label_fields)
            array = np.array(df)

            for row in array:
                if row[0] not in self.CLASSES:
                    continue

                obj = KittiObject(*row)
                objs.append(obj)

        return objs


@manager.DATASETS.add_component
class KittiPCDataset(KittiDetDataset):
    """
    """

    def __getitem__(self, index: int):
        filename = '{}.bin'.format(self.data[index])
        path = os.path.join(self.pointcloud_dir, filename)

        data = {'pointcloud': path}
        label = dict()

        if self.with_intrinsics:
            label["K"] = self.K

        if not self.is_test_mode:
            label["objects"] = self.load_annotation(index)

        if self.transforms:
            data, label = self.transforms(data, label)
        return data, label

    @property
    def pointcloud_dir(self) -> str:
        return os.path.join(self.base_dir, 'velodyne')


@manager.DATASETS.add_component
class KittiMonoDataset(KittiDetDataset):
    """
    """

    def __getitem__(self, index: int):
        filename = '{}.png'.format(self.data[index])
        path = os.path.join(self.image_dir, filename)

        data = EasyDict(image=path)
        label = dict()

        if self.with_intrinsics:
            label["K"] = self.K

        if not self.is_test_mode:
            label["objects"] = self.load_annotation(index)

        if self.transforms:
            data, label = self.transforms(data, label)
        return data, label

    @property
    def image_dir(self) -> str:
        return os.path.join(self.base_dir, 'image_2')
