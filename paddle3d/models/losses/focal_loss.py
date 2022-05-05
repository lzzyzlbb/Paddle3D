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
from paddle import nn
import paddle.nn.functional as F


class FocalLoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        """forward

        Args:
            prediction (paddle.Tensor): model prediction
            target (paddle.Tensor): ground truth

        Returns:
            paddle.Tensor: focal loss
        """
        positive_index = (target == 1).astype("float32")
        negative_index = (target < 1).astype("float32")

        negative_weights = paddle.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = paddle.log(prediction) \
                        * paddle.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = paddle.log(1 - prediction) \
                        * paddle.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss

class MultiFocalLoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, prediction, target):
        """forward

        Args:
            prediction (paddle.Tensor): model prediction
            target (paddle.Tensor): ground truth

        Returns:
            paddle.Tensor: focal loss
        """
        n = prediction.shape[0]
        
        out_size = [n] + prediction.shape[2:]
        if target.shape[1:] != prediction.shape[2:]:
            raise ValueError(f'Expected target size {out_size}, got {target.shape}')

        # compute softmax over the classes axis
        input_soft = F.softmax(prediction, axis=1)
        log_input_soft = F.log_softmax(prediction, axis=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=prediction.shape[1]).cast(prediction.dtype)
        new_shape = [0, len(target_one_hot.shape)-1] + [i for i in range(1, len(target_one_hot.shape)-1)]
        
        target_one_hot = target_one_hot.transpose(new_shape)

        # compute the actual focal loss
        weight = paddle.pow(-input_soft + 1.0, self.beta)

        focal = -self.alpha * weight * log_input_soft
        loss = paddle.einsum('bc...,bc...->b...', target_one_hot, focal)
        return loss


class SigmoidFocalClassificationLoss(nn.Layer):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma = 2.0, alpha = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def sigmoid_cross_entropy_with_logits(self, prediction, target):
        """ Implementation for sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = paddle.clip(prediction, min=0) - prediction * target + \
            paddle.log1p(paddle.exp(-paddle.abs(prediction)))
        return loss

    def forward(self, prediction, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = F.sigmoid(prediction)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * paddle.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(prediction, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

