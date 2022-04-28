import iou3d_nms


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (paddle.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (paddle.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        paddle.Tensor: Indexes after nms.
    """
    order = scores.argsort(0, descending=True)
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = paddle.gather(boxes, index=order)
    # When order is one-value tensor,
    # boxes[order] loses a dimension, so we add a reshape
    boxes = boxes.reshape([-1, boxes.shape[-1]])
    keep, num_out = iou3d_nms.nms_gpu(boxes, thresh)
    keep = order[keep[:num_out]]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep
