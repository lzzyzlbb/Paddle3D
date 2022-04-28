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

from typing import List, Union

import paddle
import nuscenes
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits as nuscenes_split
from pyquaternion import Quaternion

import paddle3d.transforms as T
from paddle3d.transforms import TransformABC
from paddle3d.object import BBox3D, DetObject


class NuscenesDetDataset(paddle.io.Dataset):
    """
    """

    _VERSION = {
        'train': 'v1.0-trainval',
        'val': 'v1.0-trainval',
        'trainval': 'v1.0-trainval',
        'test': 'v1.0-test'
    }

    _LABEL_MAP = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone'
    }

    _CLASS_MAP = {
        'pedestrian': 0,
        'car': 1,
        'motorcycle': 2,
        'bicycle': 3,
        'bus': 4,
        'truck': 5,
        'construction_vehicle': 6,
        'trailer': 7,
        'barrier': 8,
        'traffic_cone': 9
    }

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

        self.version = self._VERSION[self.mode]
        self.nusc = NuScenes(
            version=self.version, dataroot=self.dataset_root, verbose=False)
        self._build_data()

    def _build_data(self):
        scenes = getattr(nuscenes_split, self.mode)
        self.data = []

        for scene in self.nusc.scene:
            if scene['name'] not in scenes:
                continue

            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            cur_token = first_sample_token
            first_sample = self.nusc.get('sample', first_sample_token)

            while True:
                sample = self.nusc.get('sample', cur_token)
                self.data.append(sample)

                if cur_token == last_sample_token:
                    break

                cur_token = sample['next']

    def __len__(self):
        return len(self.data)

    @property
    def is_test_mode(self) -> bool:
        return self.mode == 'test'

    def load_annotation(self, index: int) -> List[DetObject]:
        res = []

        for anno in self.data[index]['anns']:
            anno = self.nusc.get('sample_annotation', anno)

            if not anno['category_name'] in self._LABEL_MAP:
                continue

            label = self._LABEL_MAP[anno['category_name']]
            label_id = self._CLASS_MAP[label]

            size = anno['size']
            quaternion = Quaternion(anno['rotation'])
            bbox3d = BBox3D(
                pt=anno['translation'],
                w=size[0],
                h=size[2],
                l=size[1],
                theta=quaternion.degrees)
            obj = DetObject(label=label, label_id=label_id, bbox3d=bbox3d)

            res.append(obj)

        return res


class NuscenesMonoDataset(NuscenesDetDataset):
    """
    """

    _SUPPORT_CAMERAS = [
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT",
        "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"
    ]

    def __init__(self,
                 dataset_root: str,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 with_intrinsics: bool = False,
                 camera: str = "CAM_FRONT"):
        super().__init__(
            dataset_root=dataset_root,
            mode=mode,
            transforms=transforms,
            with_intrinsics=with_intrinsics)

        if camera not in self._SUPPORT_CAMERAS:
            raise ValueError

        self.camera = camera

    def __getitem__(self, index: int):
        camera_token = self.data[index]['data'][self.camera]
        sample_data = self.nusc.get('sample_data', camera_token)

        data = {'image': sample_data['filename']}
        label = dict()

        if self.with_intrinsics:
            calibrated_sensor_token = sample_data['calibrated_sensor_token']
            calibrated_sensor = self.nusc.get('calibrated_sensor',
                                              calibrated_sensor_token)
            label["K"] = np.array(calibrated_sensor['camera_intrinsic'])

        if not self.is_test_mode:
            label["objects"] = self.load_annotation(index)

        if self.transforms:
            data, label = self.transforms(data, label)
        return data, label
