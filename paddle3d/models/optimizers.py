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


import math
import paddle
import paddle.nn as nn
from functools import partial
import numpy as np

import paddle.optimizer as optimizer
import paddle.regularizer as regularizer

from paddle.optimizer import AdamW
from paddle.optimizer.lr import LRScheduler
from paddle3d.apis import manager

__all__ = ['AdamWOnecycle']


class LRSchedulerCycle(LRScheduler):
    def __init__(self, total_step, lr_phases,
                 mom_phases):

        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0    
        super().__init__() 

    
def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out

@manager.OPTIMIZERS.add_component
class OneCycle(LRSchedulerCycle):
    def __init__(self, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.last_moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        self.learning_rate = low_lr
        super().__init__(total_step, lr_phases, mom_phases)
    
    def get_lr(self):
        lr = self.last_lr
        for start, end, func in self.lr_phases:
            if self.last_epoch >= start:
                lr = func((self.last_epoch - start) / (end - start))
        return lr
    
    def set_mom(self):
        mom = self.last_moms[0]
        for start, end, func in self.mom_phases:
            if self.last_epoch >= start:
                mom = func((self.last_epoch - start) / (end - start))
        self.last_moms[0] = mom

    
    def step(self, epoch=None):
        super().step()
        self.set_mom()
    
    def get_mom(self):
        return self.last_moms



@manager.OPTIMIZERS.add_component
class AdamWOnecycle(AdamW):
    
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9, 
                 beta2=0.999,
                 clip_grad_by_norm=None,
                 parameters=None,
                 **optim_args):
        if clip_grad_by_norm is not None:
            grad_clip = paddle.nn.ClipGradByNorm(
                clip_norm=clip_grad_by_norm) 
        self.learning_rate = learning_rate
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1, 
            beta2=beta2,
            grad_clip=grad_clip,
            **optim_args)

    def step(self):
        if isinstance(self._learning_rate, OneCycle):
            self._beta1 = self._learning_rate.get_mom()[0]
        super().step()

