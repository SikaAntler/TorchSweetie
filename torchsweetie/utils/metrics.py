import math

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def bbox_iou(boxes1: Tensor, boxes2: Tensor, ciou: bool = True) -> Tensor:
    """Compute the DIoU or CIoU between the predict boxes and target boxes.

    Be careful, this function cannot be used during NMS algorithm.
    The number of boxes in both sets are the same, since the target boxes are built from labels.

    Args:
        boxes1 (N, 4): the predict boxes, where 4 is (cx, cy, w, h)
        boxes2 (N, 4): the target boxes, where 4 is (cx, cy, w, h)

    Returns:
        iou (N, ): IoU between the predict boxes and target boxes

    """

    cx1, cy1, w1, h1 = boxes1.tensor_split((1, 2, 3), dim=1)
    w1_half, h1_half = w1 / 2, h1 / 2
    x1_min, x1_max = cx1 - w1_half, cx1 + w1_half
    y1_min, y1_max = cy1 - h1_half, cy1 + h1_half

    cx2, cy2, w2, h2 = boxes2.tensor_split((1, 2, 3), dim=1)
    w2_half, h2_half = w2 / 2, h2 / 2
    x2_min, x2_max = cx2 - w2_half, cx2 + w2_half
    y2_min, y2_max = cy2 - h2_half, cy2 + h2_half

    # IoU
    left = torch.max(x1_min, x2_min)
    top = torch.max(y1_min, y2_min)
    right = torch.min(x1_max, x2_max)
    bottom = torch.min(y1_max, y2_max)
    inter = (right - left).clamp(0) * (bottom - top).clamp(0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union

    # DIoU
    diagonal_w = torch.max(x1_max, x2_max) - torch.min(x1_min, x2_min)
    diagonal_h = torch.max(y1_max, y2_max) - torch.min(y1_min, y2_min)
    diagonal = diagonal_w**2 + diagonal_h**2
    centers = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    if ciou:
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-8)
        return iou - centers / (diagonal + 1e-8) - alpha * v
    else:
        return iou - centers / diagonal


def ap_per_class(tp: ndarray, conf: ndarray, p_cls: ndarray, l_cls: ndarray) -> ndarray:
    """Compute the average precision for each class and each IoU threshold.

    Args:
        tp (N, S): the flags of all predictions, the value of true positive is "True",
            where S is the number of IoU thresholds.
        conf (N, ): the confidences of all predictions.
        p_cls (N, ): the classes of predictions.
        l_cls (T, ): the classes of labels.

    Returns:
        ap (NC, S): average precision for each class and each IoU threshold.

    """

    # sort by objectness
    i = np.argsort(-conf)  # from big to small
    tp, conf, p_cls = tp[i], conf[i], p_cls[i]

    # find unique classes
    unique_classes, num_each_class = np.unique(l_cls, return_counts=True)
    num_classes = unique_classes.shape[0]

    # compute AP for each class
    ap = np.zeros((num_classes, tp.shape[1]))
    for cls_idx, cls in enumerate(unique_classes):
        i = p_cls == cls
        num_labels = num_each_class[cls_idx]  # tp + fn

        if i.sum() == 0 or num_labels == 0:
            continue

        # accumulate TPs and FPs
        tpc = tp[i].cumsum(0)
        fpc = (1 - tp[i]).cumsum(0)  # flip bool values

        # recall
        recall = tpc / num_labels

        # precision
        precision = tpc / (tpc + fpc)

        # compute AP for each iou threshold
        for j in range(ap.shape[1]):
            ap[cls_idx, j] = compute_ap(recall, precision)

    return ap


def compute_ap(recall: ndarray, precision: ndarray) -> ndarray:
    """Compute the average precision by given recall and precision for one class and one IoU threshold.

    Args:
        recall (C, ): the recall values for one class and one IoU threshold .
        precision (C, ): the precision values for one class and one IoU threshold.

    Returns:
        ap (C, ): the average precision for one class and one IoU threshold.

    """

    # append sentinel values at the begin and end
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # compute the precision envelop
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))

    # integrate area under curve
    i = np.where(precision[1:] != precision[:-1])[0]
    ap = np.sum((precision[i + 1] - precision[i]) * precision[i + 1])

    return ap
