# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import pickle
import argparse
import numpy as np

import paddle
from paddle import inference
import paddle.nn.functional as F
from paddle.static import InputSpec
from paddle3d.apis.config import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        help='config file',
        type=str)
    parser.add_argument(
        '--model_file',
        help='model file path',
        type=str)
    parser.add_argument(
        '--output_dir',
        help='Output file path',
        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def load_predictor(model_file):
    """load_predictor
    initialize the inference engine
    Args:
        model_file: inference model path (*.pdmodel and *.pdiparams)
    Return:
        predictor: Predictor created using Paddle Inference.
        input_tensor: Input tensor of the predictor.
        output_tensor: Output tensor of the predictor.
    """
    config = inference.Config(model_file + '.pdmodel', model_file + '.pdiparams')
    config.enable_use_gpu(1000, 0)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)

    # get input and output tensor property
    input_names = predictor.get_input_names()
    
    output_names = predictor.get_output_names()

    return predictor, input_names, output_names

from paddle3d.datasets.cadnn_kitti.cadnn_kitti import KittiCadnnDataset
data_p = KittiCadnnDataset("data", mode='val', point_cloud_range=[2, -30.08, -3.0, 46.8, 30.08, 1.0], 
         depth_downsample_factor=4, voxel_size=[0.16, 0.16, 0.16], class_names=['Car', 'Pedestrian', 'Cyclist'], remove_outside_boxes=True)

def main(args):
    cfg = Config(path=args.cfg)
    data_p = cfg.val_dataset
    os.makedirs(args.output_dir, exist_ok=True)

    predictor, input_names, output_names = \
        load_predictor(args.model_file)

    input_tensor1 = predictor.get_input_handle(input_names[0])
    input_tensor2 = predictor.get_input_handle(input_names[1])
    input_tensor3 = predictor.get_input_handle(input_names[2])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    n = len(data_p)   
    for i in range(0, 5): #data, data_tensor in zip(dataset, loader):
        # forward
        data = data_p[i]
        # data['images'] -= mean
        # data['images'] /= std
        data["images"] = np.expand_dims(data["images"], axis=0)
        data["trans_lidar_to_cam"] = np.expand_dims(data["trans_lidar_to_cam"], axis=0)
        data["trans_cam_to_img"] = np.expand_dims(data["trans_cam_to_img"], axis=0)
        input_tensor1.copy_from_cpu(data[input_names[0]])
        input_tensor2.copy_from_cpu(data[input_names[1]])
        input_tensor3.copy_from_cpu(data[input_names[2]])
        predictor.run()
        outs = []
        for name in output_names:
            out = predictor.get_output_handle(name)
            out = out.copy_to_cpu()
            out = paddle.to_tensor(out)
            outs.append(out)
       
        res = {}
        res['pred_boxes'] = outs[0]
        res['pred_labels'] = outs[1]
        res['pred_scores'] = outs[2]
        
        print(res)
        


if __name__ == '__main__':
    args = parse_args()
    main(args)

