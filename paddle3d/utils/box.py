import numpy as np
import paddle

def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = paddle.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = paddle.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = paddle.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = paddle.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = paddle.clip(x_max - x_min, min=0)
    y_len = paddle.clip(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / paddle.clip(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = paddle.abs(boxes3d[:, 6] - paddle.floor(boxes3d[:, 6] / np.pi + 0.5) * np.pi)
    print(rot_angle.shape)
    choose_dims = paddle.where(rot_angle[:, None] < np.pi / 4, paddle.gather(boxes3d, index=paddle.to_tensor([3, 4]), axis=1),
                             paddle.gather(boxes3d, index=paddle.to_tensor([4, 3]), axis=1))
    aligned_bev_boxes = paddle.concat([boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2], axis=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)
