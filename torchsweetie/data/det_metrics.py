from torch import Tensor
from torchvision.ops import batched_nms

from ..data import DetDataPack, DetResult
from ..utils import BoxFormat, cxcywh2xyxy, denormalize


def convert_to_preds_and_target(
    data: DetDataPack,
    predictions: list[DetResult],
    iou_threshold: float,
    max_detection: int,
    box_format: BoxFormat,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    preds: list[dict[str, Tensor]] = []
    target: list[dict[str, Tensor]] = []

    for img_id, (boxes, scores, cls_idxs) in enumerate(predictions):
        indices = batched_nms(boxes, scores, cls_idxs, iou_threshold)
        indices = indices[:max_detection]
        preds.append(
            {
                "boxes": boxes[indices],
                "scores": scores[indices],
                "labels": cls_idxs[indices],
            }
        )

        mask = data.img_idxs == img_id
        gt_cls_idxs = data.cls_idxs[mask]
        gt_boxes = data.boxes[mask]

        if box_format == BoxFormat.cxcywh:
            gt_boxes = cxcywh2xyxy(gt_boxes)

        img_h, img_w = data.images.shape[-2:]
        denormalize(gt_boxes, img_w, img_h)

        target.append({"boxes": gt_boxes, "labels": gt_cls_idxs})

    return preds, target
