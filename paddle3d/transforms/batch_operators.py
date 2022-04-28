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

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from paddle3d.transforms.base import TransformABC
from paddle3d.apis import manager

import numpy as np
from skimage import transform as trans


def encode_label(K, ry, dims, locs):
    """get bbox 3d and 2d by model output
    Args:
        K (np.ndarray): camera intrisic matrix
        ry (np.ndarray): rotation y
        dims (np.ndarray): dimensions
        locs (np.ndarray): locations
    """
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += -np.float32(l) / 2
    y_corners += -np.float32(h)
    z_corners += -np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([
        min(corners_2d[0]),
        min(corners_2d[1]),
        max(corners_2d[0]),
        max(corners_2d[1])
    ])

    return proj_point, box2d, corners_3d


def get_transfrom_matrix(center_scale, output_size):
    """get transform matrix
    """
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32)


def affine_transform(point, matrix):
    """do affine transform to label
    """
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]


def get_3rd_point(point_a, point_b):
    """get 3rd point
    """
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c


def gaussian_radius(h, w, thresh_min=0.7):
    """gaussian radius
    """
    a1 = 1
    b1 = h + w
    c1 = h * w * (1 - thresh_min) / (1 + thresh_min)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - thresh_min) * w * h
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * thresh_min
    b3 = -2 * thresh_min * (h + w)
    c3 = (thresh_min - 1) * w * h
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """get 2D gaussian map
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """draw umich gaussian
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


@manager.TRANSFORMS.add_component
class Gt2SmokeTarget(TransformABC):
    def __init__(self,
                 mode: str,
                 num_classes: int,
                 flip_prob: float = 0.5,
                 aug_prob: float = 0.3,
                 max_objs: int = 50,
                 input_size: Tuple[int, int] = (1280, 384),
                 output_stride: Tuple[int, int] = (4, 4),
                 shift_range: Tuple[float, float, float] = (),
                 scale_range: Tuple[float, float, float] = ()):
        self.max_objs = max_objs
        self.input_width = input_size[0]
        self.input_height = input_size[1]
        self.output_width = self.input_width // output_stride[0]
        self.output_height = self.input_height // output_stride[1]

        self.shift_range = shift_range
        self.scale_range = scale_range
        self.shift_scale = (0.2, 0.4)
        self.flip_prob = flip_prob
        self.aug_prob = aug_prob
        self.is_train = True if mode == 'train' else False
        self.num_classes = num_classes

    def __call__(self, data: dict, label: Optional[dict] = None):
        img = data['image']
        K = label['K']
        anns = label['objects']

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size, [self.input_width, self.input_height])
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size, [self.output_width, self.output_height])

        if not self.is_train:
            # for inference we parametrize with original size
            target = {}
            target["image_size"] = size
            target["is_train"] = self.is_train
            target["trans_mat"] = trans_mat
            target["K"] = K
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return np.array(img), target  #, original_idx

        heat_map = np.zeros(
            [self.num_classes, self.output_height, self.output_width],
            dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        c_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)
        bbox2d_size = np.zeros([self.max_objs, 2], dtype=np.float32)

        for i, a in enumerate(anns):
            if i == self.max_objs:
                break
            # a = a.copy()
            cls = a.label_id

            locs = np.array(a.bbox3d.pt)
            rot_y = np.array(a.bbox3d.theta)
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(
                K, rot_y,
                np.array([a.bbox3d.length, a.bbox3d.height, a.bbox3d.width]),
                locs)
            if np.all(box2d == 0):
                continue
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
            center = np.array([(box2d[0] + box2d[2]) / 2,
                               (box2d[1] + box2d[3]) / 2],
                              dtype=np.float32)

            if (0 < center[0] < self.output_width) and (0 < center[1] <
                                                        self.output_height):
                point_int = center.astype(np.int32)
                p_offset = point - point_int
                c_offset = center - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int,
                                                    radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                c_offsets[i] = c_offset
                dimensions[i] = np.array(
                    [a.bbox3d.length, a.bbox3d.height, a.bbox3d.width])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

                # targets for 2d bbox
                bbox2d_size[i, 0] = w
                bbox2d_size[i, 1] = h

        target = {}
        target["image_size"] = np.array(img.size)
        target["is_train"] = self.is_train
        target["trans_mat"] = trans_mat
        target["K"] = K
        target["hm"] = heat_map
        target["reg"] = regression
        target["cls_ids"] = cls_ids
        target["proj_p"] = proj_points
        target["dimensions"] = dimensions
        target["locations"] = locations
        target["rotys"] = rotys
        target["reg_mask"] = reg_mask
        target["flip_mask"] = flip_mask
        target["bbox_size"] = bbox2d_size
        target["c_offsets"] = c_offsets

        label['target'] = target
        data["image"] = np.array(img).transpose((2, 0, 1))

        return data, target
