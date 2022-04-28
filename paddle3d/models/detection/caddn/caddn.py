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
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.utils.logger import logger
from paddle3d.models.layers import ConvBNReLU
from .bev import BEV
from .f2v import FrustumToVoxel
import iou3d_nms

@manager.MODELS.add_component
class CADDN(nn.Layer):
    """
    """

    def __init__(self, backbone_3d, bev_cfg, dense_head, 
                 class_head, channel_reduce_cfg, f2v_cfg, map_to_bev_cfg, post_process_cfg):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.class_head = class_head
        self.channel_reduce = ConvBNReLU(**channel_reduce_cfg)
        self.map_to_bev = ConvBNReLU(**map_to_bev_cfg)
        self.backbone_2d = BEV(**bev_cfg)
        self.dense_head = dense_head
        self.f2v = FrustumToVoxel(**f2v_cfg)
        self.post_process_cfg = post_process_cfg
        
    def forward(self, data, targets=None):
        images = data["images"]
        if not self.training:
            b, c, h, w = paddle.shape(images)
            data["batch_size"] = b
            data["image_shape"] = paddle.concat([h, w]).unsqueeze(0)
        # ffe
        image_features = self.backbone_3d(images)
        depth_logits = self.class_head(image_features)
        b, c, h, w = paddle.shape(image_features[0])
        depth_logits = F.interpolate(depth_logits, size=[h, w], mode='bilinear', align_corners=False)
        image_features = self.channel_reduce(image_features[0])
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)
        data["frustum_features"] = frustum_features
                                                    
        #   frustum_to_voxel  
        data = self.f2v(data)
        # map_to_bev
        # voxel_features = voxel_features.reshape([])
        voxel_features = data["voxel_features"]
        bev_features = voxel_features.flatten(start_axis=1, stop_axis=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        b, c, h, w = paddle.shape(bev_features)
        bev_features = bev_features.reshape([b, self.map_to_bev._conv._in_channels, h, w])
        bev_features = self.map_to_bev(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        data["spatial_features"] = bev_features
        
        # backbone_2d
        data = self.backbone_2d(data)
        predictions = self.dense_head(data)
        
        if not self.training:
            return self.post_process(predictions)
        else:
            loss = self.get_loss(predictions, targets)
            return {'loss': loss}
        
    def get_loss(self, predictions, targets):
        disp_dict = {}

        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.ffe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth
        return loss, tb_dict, disp_dict
    
    def post_process(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        batch_size = 1 # batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = F.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [F.sigmoid(x) for x in cls_preds]
            
            label_preds = paddle.argmax(cls_preds, axis=-1) + 1.0
            cls_preds = paddle.max(cls_preds, axis=-1)
            selected_score, selected_label, selected_box  = self.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds, 
                label_preds=label_preds,
                nms_config=self.post_process_cfg['nms_config'],
                score_thresh=self.post_process_cfg['score_thresh']
            ) 
            record_dict = paddle.concat([selected_score.unsqueeze(1), selected_box, selected_label.unsqueeze(1)], axis=1)
            
            pred_dicts.append(record_dict)
        return paddle.concat(pred_dicts)


    def class_agnostic_nms(self, box_scores, box_preds, label_preds, nms_config, score_thresh):
        
        scores_mask = paddle.nonzero(box_scores >= score_thresh)
        
        fake_score = paddle.to_tensor([0.0], dtype='float32')
        fake_label = paddle.to_tensor([-1.0], dtype='float32')
        fake_box = paddle.to_tensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0]], dtype='float32')
        if paddle.shape(scores_mask)[0] == 0:
            return fake_score, fake_label, fake_box
        else:    
            scores_mask = scores_mask
            box_scores = paddle.gather(box_scores, index=scores_mask)
            box_preds = paddle.gather(box_preds, index=scores_mask)
            label_preds = paddle.gather(label_preds, index=scores_mask)
            order = box_scores.argsort(0, descending=True)
            order = order[:nms_config['nms_pre_maxsize']]
            box_preds = paddle.gather(box_preds, index=order)
            box_scores = paddle.gather(box_scores, index=order)
            label_preds = paddle.gather(label_preds, index=order)
            # When order is one-value tensor,
            # boxes[order] loses a dimension, so we add a reshape
            keep, num_out = iou3d_nms.nms_gpu(box_preds, nms_config['nms_thresh'])
            if num_out.cast("int64") == 0:
                return fake_score, fake_label, fake_box
            else:
                selected = keep[0:num_out]
                selected = selected[:nms_config['nms_post_maxsize']]
                selected_score = paddle.gather(box_scores, index = selected)
                selected_box = paddle.gather(box_preds, index = selected)
                selected_label = paddle.gather(label_preds, index = selected)
                return selected_score, selected_label, selected_box


    
    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth classification scores
        Args:
            image_features [torch.Tensor(N, C, H, W)]: Image features
            depth_logits [torch.Tensor(N, D, H, W)]: Depth classification logits
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, axis=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    