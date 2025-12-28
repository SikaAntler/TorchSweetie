import numpy as np

from ..data.det_dataset import BBox


def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
    inter_left = max(bbox1.left, bbox2.left)
    inter_top = max(bbox1.top, bbox2.top)
    inter_right = max(bbox1.right, bbox2.right)
    inter_bottom = max(bbox1.bottom, bbox2.bottom)

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    if inter_area <= 0.0:
        return 0.0

    area1 = (bbox1.right - bbox1.left) * (bbox1.bottom - bbox1.top)
    area2 = (bbox2.right - bbox2.left) * (bbox2.bottom - bbox2.top)

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_thresholds = np.linspace(0.0, 1.0, 101)

    ap = 0.0
    for t in recall_thresholds:
        precision_at_recall = precisions[recalls >= t]
        if precision_at_recall.size == 0:
            p = 0.0
        else:
            p = np.max(precision_at_recall)
        ap += p
    ap /= 101.0

    return ap


def evaluate_single_image_single_class(
    ground_truth: list[BBox],
    predictions: list[BBox],
    iou_threshold: float,
) -> list[bool]:
    if len(predictions) == 0:
        return []

    pred_sorted = sorted(predictions, key=lambda x: x.score, reverse=True)

    matched_gt = set()
    tp_flags = []

    for pred in pred_sorted:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_flags.append(True)
            matched_gt.add(best_idx)
        else:
            tp_flags.append(False)

    return tp_flags


def evaluate_dataset(
    all_ground_truths: list[list[BBox]],
    all_predications: list[list[BBox]],
    classes: list[str],
    iou_thresholds: list[float] | None = None,
) -> dict:
    assert len(all_ground_truths) == len(all_predications)

    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
