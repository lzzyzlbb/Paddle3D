"""
This file has been modified by Cody Reading to add the `get_pad_params` function
"""

import logging
import os
import pickle
import random
import shutil
import subprocess
import paddle

import numpy as np


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_paddle(val)
    ans = val - paddle.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points = paddle.to_tensor(points)
    angle = paddle.to_tensor(angle)

    cosa = paddle.cos(angle)
    sina = paddle.sin(angle)
    zeros = paddle.zeros(shape=[points.shape[0]])
    ones = paddle.ones(shape=[points.shape[0]])
    rot_matrix = paddle.stack([
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ], axis=1).reshape([-1, 3, 3]).cast("float32")
    points_rot = paddle.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = paddle.concat([points_rot, points[:, :, 3:]], axis=-1)
    return points_rot.numpy()


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
        & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = paddle.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = paddle.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)
    paddle.backends.cudnn.deterministic = True
    paddle.backends.cudnn.benchmark = False


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size [int]: Desired padded output size
        cur_size [int]: Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params [tuple(int)]: Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds
