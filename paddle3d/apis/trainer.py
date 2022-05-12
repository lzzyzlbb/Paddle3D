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

import os
from enum import Enum
from typing import Callable, Union

import paddle
from visualdl import LogWriter

import paddle3d.env as env
from paddle3d.batch import collate_fn
from paddle3d.apis.checkpoint import Checkpoint, CheckpointABC
from paddle3d.apis.scheduler import Scheduler, SchedulerABC
from paddle3d.apis.pipeline import trainning_step, validation_step
from paddle3d.utils.logger import logger
from paddle3d.utils.timer import Timer


def default_dataloader_build_fn(**kwargs) -> paddle.io.DataLoader:
    """
    """

    def _generate_loader(dataset: paddle.io.Dataset):
        batch_size = kwargs.get('batch_size', 1)
        shuffle = kwargs.get('shuffle',
                             False if not dataset.is_train_mode else True)
        drop_last = kwargs.get('drop_last',
                               False if not dataset.is_train_mode else True)
        collate_fn =  dataset.collate_fn if callable(getattr(dataset ,"collate_fn")) else collate_fn
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last)

        return paddle.io.DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    return _generate_loader


def default_checkpoint_build_fn(**kwargs) -> Checkpoint:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('save_dir', 'output')
    kwargs.setdefault('keep_checkpoint_max', 5)
    kwargs.setdefault('overwrite', True)
    return Checkpoint(**kwargs)


def default_scheduler_build_fn(**kwargs) -> Scheduler:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('save_interval', 1000)
    kwargs.setdefault('log_interval', 10)
    kwargs.setdefault('do_eval', False)
    return Scheduler(**kwargs)


class Trainer:
    """
    """

    def __init__(
            self,
            model: paddle.nn.Layer,
            iters: int,
            optimizer: paddle.optimizer.Optimizer,
            train_dataset: paddle.io.Dataset,
            val_dataset: paddle.io.Dataset = None,
            # TODO: Default parameters should not use mutable objects, there is a risk
            checkpoint: Union[dict, CheckpointABC] = dict(),
            scheduler: Union[dict, SchedulerABC] = dict(),
            dataloader_fn: Union[dict, Callable] = dict()):

        self.model = model
        self.optimizer = optimizer
        self.iters = iters
        self.cur_iter = 0
        vdl_file_name = None

        if env.nranks > 1:
            env.init_distributed()
            self.optimizer = paddle.distributed.fleet.distributed_optimizer(
                optimizer)
            self.model = paddle.distributed.fleet.distributed_model(model)

        self.checkpoint = default_checkpoint_build_fn(
            **checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.scheduler = default_scheduler_build_fn(
            **scheduler) if isinstance(scheduler, dict) else scheduler

        _dataloader_build_fn = default_dataloader_build_fn(
            **dataloader_fn) if isinstance(dataloader_fn,
                                           dict) else dataloader_fn

        self.train_dataloader = _dataloader_build_fn(train_dataset)
        self.eval_dataloader = _dataloader_build_fn(
            val_dataset) if val_dataset else None
        self.val_dataset = val_dataset

        if not self.checkpoint.empty:
            state_dict = self.checkpoint.get()
            self.model.set_dict(state_dict['params'])
            self.optimizer.set_state_dict(state_dict['opts'])
            self.cur_iter = self.checkpoint.meta.get('iters')

            logger.info(
                'Resume model from checkpoint {}, current iter set to {}'.
                format(self.checkpoint.rootdir, self.cur_iter))
            vdl_file_name = self.checkpoint.meta['vdl_file_name']

        self.log_writer = LogWriter(
            logdir=self.checkpoint.rootdir, file_name=vdl_file_name)
        self.checkpoint.record('vdl_file_name',
                               os.path.basename(self.log_writer.file_name))

    def train(self):
        """
        """
        loss_sum = 0
        timer = Timer(iters=self.iters - self.cur_iter)

        while self.cur_iter < self.iters:

            for sample in self.train_dataloader:
                self.cur_iter += 1
                if self.cur_iter > self.iters:
                    break

                loss = trainning_step(self.model, self.optimizer, sample)
                loss_sum += loss
                
                timer.step()
                status = self.scheduler.step()
                if status.do_log and env.local_rank == 0:
                    lr = self.optimizer.get_lr()
                    loss_sum = float(loss_sum / self.scheduler.log_interval)
                    logger.info(
                        '[TRAIN] iter={}/{}, loss={:.4f}, lr={:.6f} | ETA {}'.
                        format(self.cur_iter, self.iters, loss_sum, lr,
                               timer.eta))

                    self.log_writer.add_scalar(
                        tag='Training/learning_rate',
                        value=lr,
                        step=self.cur_iter)
                    self.log_writer.add_scalar(
                        tag='Training/loss', value=loss_sum, step=self.cur_iter)

                    loss_sum = 0

                if status.do_eval and env.local_rank == 0:
                    # TODO: whether to save a checkpoint based on the metric
                    metrics = self.evaluate()

                if status.save_checkpoint and env.local_rank == 0:
                    dic = {
                        'params': self.model.state_dict(),
                        'opts': self.optimizer.state_dict()
                    }

                    logger.info('Push model to checkpoint {}'.format(
                        self.checkpoint.rootdir))
                    self.checkpoint.push(dic)
                    self.checkpoint.record('iters', self.cur_iter)

    def evaluate(self) -> float:
        """
        """
        results = []
        metrics = None

        if self.val_dataset is None:
            raise RuntimeError
        with logger.processing('evaluate on validate dataset'):
            for step_id, sample in enumerate(self.eval_dataloader):
                # list in list out
                pred_dicts = validation_step(self.model, sample)
                results += self.val_dataset.generate_prediction_dicts(
                        sample, pred_dicts,
                        output_path= None
                    )
                if step_id % 1 == 0:
                    logger.info("Eval iter: {}\n".format(step_id))
            metrics = self.val_dataset.evaluation(results)
            logger.info(metrics)
        return metrics
