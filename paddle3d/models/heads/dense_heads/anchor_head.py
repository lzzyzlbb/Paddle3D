# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Normal
from paddle import ParamAttr
from paddle3d.apis import manager
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .target_assigner.anchor_generator import AnchorGenerator
from paddle3d.utils.box_coder import ResidualCoder

__all__ = ['AnchorHeadSingle']

@manager.HEADS.add_component
class AnchorHeadSingle(nn.Layer):
    def __init__(self, model_cfg, input_channels, class_names, grid_size, point_cloud_range, anchor_target_cfg,
                 predict_boxes_when_training, anchor_generator_cfg, num_dir_bins):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = len(class_names)
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.anchor_generator_cfg = anchor_generator_cfg
        self.num_dir_bins = num_dir_bins
        self.box_coder = ResidualCoder(num_dir_bins=num_dir_bins)
        grid_size = np.array(grid_size)
        anchors, self.num_anchors_per_location = self.generate_anchors(
            grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = paddle.concat(anchors, axis=-3) # [x for x in anchors]

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2D(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1,
            bias_attr=ParamAttr(initializer=Constant(-np.log((1 - 0.01) / 0.01)))
        )
        self.conv_box = nn.Conv2D(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=Normal(mean=0, std=0.001))
        )
        self.target_assigner = AxisAlignedTargetAssigner(anchor_generator_cfg, 
                anchor_target_cfg, class_names=self.class_names, box_coder=self.box_coder
            )
        self.conv_dir_cls = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * num_dir_bins,
            kernel_size=1
        )
        self.forward_ret_dict = {}

        
    def generate_anchors(self, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=self.anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in self.anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
        
        return anchors_list, num_anchors_per_location_list

    
    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        num_anchors = paddle.shape(self.anchors.reshape([-1, paddle.shape(self.anchors)[5]]))[0]
        batch_anchors = self.anchors.reshape([1, -1, paddle.shape(self.anchors)[5]]).tile([batch_size, 1, 1])
        batch_cls_preds = cls_preds.reshape([batch_size, num_anchors, -1]) \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(box_preds, list) \
            else paddle.concat(box_preds, axis=1).reshape([batch_size, num_anchors, -1])
        batch_box_preds = self.box_coder.decode_paddle(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg['dir_offset']
            dir_limit_offset = self.model_cfg['dir_limit_offset']
            dir_cls_preds = dir_cls_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(dir_cls_preds, list) \
                else paddle.concat(dir_cls_preds, axis=1).reshape([batch_size, num_anchors, -1])
            print(dir_cls_preds.shape)
            dir_labels = paddle.argmax(dir_cls_preds, axis=-1)

            period = (2 * np.pi / self.num_dir_bins)
            dir_rot = self.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.cast(batch_box_preds.dtype)

        return batch_cls_preds, batch_box_preds

    def limit_period(self, val, offset=0.5, period=np.pi):
        ans = val - paddle.floor(val / period + offset) * period
        return ans

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        cls_preds = cls_preds.transpose([0, 2, 3, 1])  # [N, H, W, C]
        box_preds = box_preds.transpose([0, 2, 3, 1])  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        dir_cls_preds = dir_cls_preds.transpose([0, 2, 3, 1])
        self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
    

        if self.training:
            targets_dict = self.target_assigner.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=1,
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
    
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives)
        reg_weights = positives
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True)
        reg_weights /= paddle.clamp(pos_normalizer, min=1.0)
        cls_weights /= paddle.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = paddle.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.reshape(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives
        pos_normalizer = positives.sum(1, keepdim=True)
        reg_weights /= paddle.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = paddle.concat(
                    [anchor.transpose([3, 4, 0, 1, 2, 5]).reshape([-1, anchor.shape[-1]]) for anchor in
                     self.anchors], axis=0)
            else:
                anchors = paddle.concat(self.anchors, axis=-3)
        else:
            anchors = self.anchors
        anchors = anchors.reshape([1, -1, anchors.shape[-1]]).tile([batch_size, 1, 1])
        box_preds = box_preds.reshape([batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.reshape([batch_size, -1, self.model_cfg.NUM_DIR_BINS])
            weights = positives.type_as(dir_logits)
            weights /= paddle.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.reshape([batch_size, -1, anchors.shape[-1]])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = paddle.floor(offset_rot / (2 * np.pi / num_bins))
        dir_cls_targets = paddle.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = paddle.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets
