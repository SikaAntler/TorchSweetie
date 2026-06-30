import math

import torch
from torch import Tensor


def bbox_iou(
    boxes1: Tensor,
    boxes2: Tensor,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> Tensor:
    """Calculates IoU, GIoU, DIoU, or CIoU between two boxes.

    Args:
        boxes1 (N, 4): where 4 = [cx, cy, w, h]
        boxes2 (N, 4): where 4 = [cx, cy, w, h]

    Returns:
        iou (N, ): IoU between two boxes
    """

    cx1, cy1, w1, h1 = boxes1.tensor_split((1, 2, 3), dim=1)
    w1_half, h1_half = w1 / 2, h1 / 2
    b1_x1, b1_x2 = cx1 - w1_half, cx1 + w1_half
    b1_y1, b1_y2 = cy1 - h1_half, cy1 + h1_half

    cx2, cy2, w2, h2 = boxes2.tensor_split((1, 2, 3), dim=1)
    w2_half, h2_half = w2 / 2, h2 / 2
    b2_x1, b2_x2 = cx2 - w2_half, cx2 + w2_half
    b2_y1, b2_y2 = cy2 - h2_half, cy2 + h2_half

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
