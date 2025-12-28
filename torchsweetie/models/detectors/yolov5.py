import torch
import torchvision
from torch import Tensor, nn

from ...data import DetDataPack
from ...utils import MODELS
from .backbones import YOLOv5BackBone
from .heads import YOLOv5Head
from .necks import YOLOv5Neck


@MODELS.register(scope="detection")
class YOLOv5(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: list[list[int]],
        in_channels: int = 3,
        stride: list[int] = [8, 16, 32],
    ) -> None:
        super().__init__()

        self.backbone = YOLOv5BackBone(in_channels)
        self.neck = YOLOv5Neck()
        self.head = YOLOv5Head(num_classes, anchors, stride)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3  # ty: ignore
                m.momentum = 0.03  # ty: ignore
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # ty: ignore

    def forward(self, data: DetDataPack) -> Tensor | list:
        x = data.images

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.training:
            return x
        else:
            return self.non_maximum_suppression(x)

    def non_maximum_suppression(
        self, predictions: Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        # predictions: (B, N, 5+C), where 5+C = [cx, cy, w, h, obj, c1, c2, ..., cn]

        results = []

        for pred in predictions:
            cls_scores, cls_idxs = torch.max(pred[:, 5:], 1)
            scores = pred[:, 4] * cls_scores
            keep = scores >= conf_thres
            pred = pred[keep]
            scores = scores[keep]
            cls_idxs = cls_idxs[keep]

            if pred.numel() == 0:
                results.append(
                    (
                        pred.new_zeros((0, 4)),
                        pred.new_zeros((0,)),
                        pred.new_zeros((0,), dtype=torch.long),
                    )
                )
                continue

            # cxcywh -> xyxy
            half_w = pred[:, 2:3] / 2
            half_h = pred[:, 3:4] / 2
            x1 = pred[:, 0:1] - half_w
            y1 = pred[:, 1:2] - half_h
            x2 = pred[:, 0:1] + half_w
            y2 = pred[:, 1:2] + half_h
            boxes = torch.hstack([x1, y1, x2, y2])  # (N, 4)

            indices = torchvision.ops.batched_nms(boxes, scores, cls_idxs, iou_thres)
            boxes = boxes[indices]
            scores = scores[indices]
            cls_idxs = cls_idxs[indices]

            results.append((boxes, scores, cls_idxs))

        return results
