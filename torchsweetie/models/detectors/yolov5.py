import torch
from torch import Tensor, nn

from ...data import DetDataPack, DetResult
from ...utils import MODELS, cxcywh2xyxy
from .backbones import YOLOv5BackBone
from .heads import YOLOv5Head
from .necks import YOLOv5Neck


@MODELS.register(scope="detection")
class YOLOv5(nn.Module):
    export: bool = False

    def __init__(
        self,
        num_classes: int,
        anchors: list[list[int]],
        in_channels: int = 3,
        strides: list[int] = [8, 16, 32],
        max_nms: int = 30000,
    ) -> None:
        super().__init__()

        self.backbone = YOLOv5BackBone(in_channels)
        self.neck = YOLOv5Neck()
        self.head = YOLOv5Head(num_classes, anchors, strides)

        self.max_nms = max_nms

        self.multi_labels = True
        self.conf_threshold = 0.001

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3  # ty: ignore
                m.momentum = 0.03  # ty: ignore
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # ty: ignore

    def forward(self, data: DetDataPack) -> Tensor | DetResult | list[DetResult]:
        x = data.images

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.training:
            return x
        elif self.export:
            return self.postprocess_export(x)
        else:
            return self.postprocess(x)

    def postprocess_export(self, predictions: Tensor) -> DetResult:
        batched_boxes = []
        batched_scores = []
        batched_cls_idxs = []

        for pred in predictions:
            cls_scores, cls_idxs = torch.max(pred[:, 5:], 1)
            scores = pred[:, 4] * cls_scores
            _, keep = torch.topk(scores, self.max_nms)
            boxes = pred[:, :4][keep]
            scores = scores[keep]
            cls_idxs = cls_idxs[keep]

            boxes = cxcywh2xyxy(boxes)

            batched_boxes.append(boxes)
            batched_scores.append(scores)
            batched_cls_idxs.append(cls_idxs)

        return DetResult(
            torch.cat(batched_boxes), torch.cat(batched_scores), torch.cat(batched_cls_idxs)
        )

    def postprocess(self, predictions: Tensor) -> list[DetResult]:
        # predictions: (B, N, 5+C), where 5+C = [cx, cy, w, h, obj, c1, c2, ..., cn]

        results: list[DetResult] = []

        for pred in predictions:
            if self.multi_labels:
                conf = pred[:, 5:] * pred[:, 4:5]
                indices, cls_idxs = (conf > self.conf_threshold).nonzero(as_tuple=False).T
                boxes = pred[indices, :4]
                scores = conf[indices, cls_idxs]
            else:
                cls_scores, cls_idxs = torch.max(pred[:, 5:], 1)
                scores = pred[:, 4] * cls_scores
                keep = scores >= self.conf_threshold
                boxes = pred[:, :4][keep]
                scores = scores[keep]
                cls_idxs = cls_idxs[keep]

            if boxes.numel() == 0:
                results.append(
                    DetResult(
                        boxes.new_zeros((0, 4)),
                        scores.new_zeros((0,)),
                        cls_idxs.new_zeros((0,), dtype=torch.long),
                    )
                )
                continue

            # max_nms
            keep = torch.argsort(scores, descending=True)[: self.max_nms]
            boxes = boxes[keep]
            scores = scores[keep]
            cls_idxs = cls_idxs[keep]

            boxes = cxcywh2xyxy(boxes)

            results.append(DetResult(boxes, scores, cls_idxs))

        return results
