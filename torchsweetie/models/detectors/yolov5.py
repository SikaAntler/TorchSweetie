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
            return self.non_maximum_suppression(x, 0.001, 0.6, True)

    def non_maximum_suppression(
        self, predictions: Tensor, conf_thres: float, iou_thres: float, multi_labels: bool = False
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        # predictions: (B, N, 5+C), where 5+C = [cx, cy, w, h, obj, c1, c2, ..., cn]

        results = []

        for pred in predictions:
            if multi_labels:
                conf = pred[:, 5:] * pred[:, 4:5]
                indices, cls_idxs = (conf > conf_thres).nonzero(as_tuple=False).T
                boxes = pred[indices, :4]
                scores = pred[indices, cls_idxs]
            else:
                cls_scores, cls_idxs = torch.max(pred[:, 5:], 1)
                scores = pred[:, 4] * cls_scores
                keep = scores >= conf_thres
                boxes = pred[:, :4][keep]
                scores = scores[keep]
                cls_idxs = cls_idxs[keep]

            # max_nms
            keep = torch.argsort(scores, descending=True)[:30000]
            boxes = boxes[keep]
            scores = scores[keep]
            cls_idxs = cls_idxs[keep]

            if boxes.numel() == 0:
                results.append(
                    (
                        boxes.new_zeros((0, 4)),
                        scores.new_zeros((0,)),
                        cls_idxs.new_zeros((0,), dtype=torch.long),
                    )
                )
                continue

            # cxcywh -> xyxy
            half_w = boxes[:, 2:3] / 2
            half_h = boxes[:, 3:4] / 2
            x1 = boxes[:, 0:1] - half_w
            y1 = boxes[:, 1:2] - half_h
            x2 = boxes[:, 0:1] + half_w
            y2 = boxes[:, 1:2] + half_h
            boxes = torch.hstack([x1, y1, x2, y2])  # (N, 4)

            indices = torchvision.ops.batched_nms(boxes, scores, cls_idxs, iou_thres)
            indices = indices[:300]
            boxes = boxes[indices]
            scores = scores[indices]
            cls_idxs = cls_idxs[indices]

            results.append((boxes, scores, cls_idxs))

        return results
