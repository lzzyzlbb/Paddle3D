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

import abc
import copy
import os
import shutil
from typing import Hashable, Generic, Optional

import paddle
import yaml
from easydict import EasyDict

from paddle3d.utils.logger import logger


class CheckpointABC(abc.ABC):
    """
    """

    @abc.abstractmethod
    def get(self, tag: Optional[str] = None) -> dict:
        """
        """

    @abc.abstractmethod
    def push(self, state_dict: dict, tag: Optional[str] = None):
        """
        """

    @abc.abstractmethod
    def pop(self) -> str:
        """
        """

    @abc.abstractproperty
    def empty(self) -> bool:
        """
        """

    @abc.abstractmethod
    def record(self, key: Hashable, value: Generic) -> bool:
        """
        """

    @abc.abstractproperty
    def meta(self) -> dict:
        """
        """

    @abc.abstractproperty
    def metafile(self) -> str:
        """
        """

    @abc.abstractproperty
    def rootdir(self) -> str:
        """
        """


class Checkpoint(CheckpointABC):
    """
    """

    def __init__(self,
                 save_dir: str,
                 keep_checkpoint_max: int = 5,
                 overwrite: bool = True):
        self.save_dir = save_dir
        self._meta = EasyDict()

        self._meta.overwrite = overwrite
        self._meta.keep_checkpoint_max = keep_checkpoint_max
        self._meta.counter = 0

        self._meta.queue = []

        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(self.metafile):
            with open(self.metafile) as file:
                dic = yaml.load(file, Loader=yaml.FullLoader)
                self._meta.update(dic)

        self._sync_to_file()

    def get(self, tag: Optional[str] = None) -> dict:
        """
        """
        if tag is None:
            if len(self.meta.queue) == 0:
                raise RuntimeError
            tag = self.meta.queue[-1]

        if tag not in self.meta.queue:
            raise ValueError

        path = os.path.join(self.rootdir, tag, 'model.params')
        return paddle.load(path)

    def push(self,
             state_dict: dict,
             tag: Optional[str] = None,
             enqueue: bool = True) -> str:
        """
        """
        tag = str(self._meta.counter) if tag is None else tag
        dirname = os.path.join(self.rootdir, tag)
        path = os.path.join(dirname, 'model.params')

        if enqueue:
            if self._meta.keep_checkpoint_max > 0 and len(
                    self._meta.queue) >= self._meta.keep_checkpoint_max:
                self.pop()

            self._meta.queue.append(tag)
            self._meta.counter += 1
        else:
            if os.path.exists(path) and not self._meta.overwrite:
                raise RuntimeError

        os.makedirs(dirname, exist_ok=True)
        paddle.save(state_dict, path)
        self._sync_to_file()

        return tag

    def pop(self) -> str:
        """
        """
        if len(self._meta.queue) == 0:
            raise RuntimeError

        pop_idx = self._meta.queue[0]
        pop_dir = os.path.join(self.rootdir, pop_idx)
        shutil.rmtree(pop_dir)
        self._meta.queue = self._meta.queue[1:]
        self._sync_to_file()

        return pop_idx

    @property
    def empty(self):
        """
        """
        return len(self._meta.queue) == 0

    def record(self, key: Hashable, value: Generic) -> bool:
        """
        """
        if key in self._meta and not self._meta.overwrite:
            return False

        self._meta[key] = value
        self._sync_to_file()
        return True

    def _sync_to_file(self):
        with open(self.metafile, 'w') as file:
            yaml.dump(dict(self.meta), file)

    @property
    def meta(self) -> dict:
        """
        """
        return copy.deepcopy(self._meta)

    @property
    def metafile(self) -> str:
        """
        """
        return os.path.join(self.rootdir, 'meta.yaml')

    @property
    def rootdir(self) -> str:
        """
        """
        return self.save_dir
