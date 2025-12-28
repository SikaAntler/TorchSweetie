import math

import torch
from torch import Tensor

from .enums import BoxFormat


def bbox_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    cxcywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> Tensor:
    """Calculates IoU, GIoU, DIoU, or CIoU between two boxes.

    Args:
        boxes1 (N, 4): where 4 = [cx, cy, w, h] if cxcywh else [x1, y1, x2, y2]
        boxes2 (N, 4): where 4 = [cx, cy, w, h] if cxcywh else [x1, y1, x2, y2]

    Returns:
        iou (N, ): IoU between two boxes
    """

    if cxcywh:
        cx1, cy1, w1, h1 = boxes1.chunk(4, -1)
        w1_half, h1_half = w1 / 2, h1 / 2
        b1_x1, b1_x2 = cx1 - w1_half, cx1 + w1_half
        b1_y1, b1_y2 = cy1 - h1_half, cy1 + h1_half
        cx2, cy2, w2, h2 = boxes2.chunk(4, -1)
        w2_half, h2_half = w2 / 2, h2 / 2
        b2_x1, b2_x2 = cx2 - w2_half, cx2 + w2_half
        b2_y1, b2_y2 = cy2 - h2_half, cy2 + h2_half
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.chunk(4, -1)
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
        ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = (
                ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4

            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU

            return iou - rho2 / c2  # DIoU

        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    return iou  # IoU


def xyxy2cxcywh(boxes: Tensor, normalized: bool = True) -> Tensor:
    """转换边界框格式

    Args:
        boxes: (B, 4), where 4 = [x1, y1, x2, y2]
        normalized: 是否归一化

    Returns:
        (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.chunk(4, 1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    if not normalized:
        w += 1
        h += 1

    return torch.hstack([cx, cy, w, h])


def cxcywh2xyxy(boxes: Tensor, normalized: bool = True) -> Tensor:
    """转换边界框格式

    Args:
        boxes: (B, 4), where 4 = [cx, cy, w, h]
        normalized: 是否归一化

    Returns:
        (B, 4), where 4 = [x1, y1, x2, y2]
    """
    cx, cy, w, h = boxes.chunk(4, 1)

    if normalized:
        half_w = w / 2
        half_h = h / 2
    else:
        half_w = (w - 1) / 2
        half_h = (h - 1) / 2

    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h

    return torch.hstack([x1, y1, x2, y2])


def denormalize(
    boxes: Tensor, img_w: int, img_h: int, box_format: BoxFormat = BoxFormat.xyxy
) -> None:
    """逆归一化

    Args:
        boxes: (B, 4)
        box_format: 边界框坐标格式

    Notes:
        假设图像和边界框横向均占据 0 1 2 3 4 五个像素
        left' = 0.0, right' = 1.0, center_x' = 0.5, width' = 1.0
        因此在还原时
        left     = left'     * (img_w - 1) = 0
        right    = right'    * (img_w - 1) = 4
        center_x = center_x' * (img_w - 1) = 2
        width    = width'    * img_w       = 5
    """
    boxes[:, 0] *= img_w - 1
    boxes[:, 1] *= img_h - 1

    if box_format == BoxFormat.xyxy:
        boxes[:, 2] *= img_w - 1
        boxes[:, 3] *= img_h - 1
    elif box_format == BoxFormat.xywh:
        boxes[:, 2] *= img_w
        boxes[:, 3] *= img_h
    elif box_format == BoxFormat.cxcywh:
        boxes[:, 2] *= img_w
        boxes[:, 3] *= img_h
