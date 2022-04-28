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

import paddle

nranks = paddle.distributed.ParallelEnv().nranks
local_rank = paddle.distributed.ParallelEnv().local_rank


def init_distributed():
    """
    """
    if not is_distributed_inited():
        paddle.distributed.fleet.init(is_collective=True)


def is_distributed_inited():
    """
    """
    return paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
    )
