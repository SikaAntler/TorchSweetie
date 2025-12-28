import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import LOSSES, bbox_iou, cxcywh2xyxy, denormalize
from .tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()

        self.reg_max = reg_max

    def forward(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left

        loss_left = F.cross_entropy(pred_dist, target_left.view(-1), reduction="none").view(
            target_left.shape
        )
        loss_right = F.cross_entropy(pred_dist, target_right.view(-1), reduction="none").view(
            target_left.shape
        )

        return (loss_left * weight_left * loss_right * weight_right).mean(-1, True)


def bbox2dist(anchor_points: Tensor, bboxes: Tensor, reg_max: int | None = None) -> Tensor:
    x1y1, x2y2 = bboxes.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)

    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points - rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(
    feats: list[Tensor], strides: list[int], grid_cell_offset: float = 0.5
) -> tuple[Tensor, Tensor]:
    """将多尺度特征图上的每一个网格位置，转换成一个二维参考点 (x, y)，同时记录这个点所属特征层的 stride

    Args:
        feats: [P3, P4, P5, ...]
               P3/ 8: (B, C, 160, 160)
               P4/16: (B, C,  80,  80)
               P5/32: (B, C,  40,  40)
        strides: [8, 16, 32, ...]

    Returns:
        (33600, 2)
        (33600, 1)
    """
    assert len(feats) == len(strides)

    anchor_points, stride_tensor = [], []

    for i in range(len(feats)):
        stride = strides[i]
        h, w = feats[i].shape[2:]
        dtype, device = feats[i].dtype, feats[i].device
        sx = torch.arange(w, dtype=dtype, device=device) + grid_cell_offset
        sy = torch.arange(h, dtype=dtype, device=device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")  # (160, 160)
        # (160, 160, 2) -> (25600, 2)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # (25600, 1)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        target_bboxes: Tensor,
        target_scores: Tensor,
        target_scores_sum: Tensor,
        fg_mask: Tensor,
        imgsz: Tensor,
        stride: Tensor,
    ) -> tuple[Tensor, Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
                )
                * weight
            )
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
            loss_dfl = loss_dfl.mean(-1, True) * weight

        loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


@LOSSES.register(scope="detection")
class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        strides: list[int] = [8, 16, 32],
        tal_topk: int = 10,
    ):
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + reg_max * 4
        self.strides = strides
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.use_dfl = reg_max > 1

        self.assigner = TaskAlignedAssigner(tal_topk, num_classes, 0.5, 6.0, strides)
        self.bbox_loss = BboxLoss(reg_max)
        self.proj = torch.arange(reg_max, dtype=torch.float)

    # def forward(self, preds: dict[str, Tensor]):
    # loss = torch.zeros(3, device=)

    def preprocess(self, target: Tensor, batch_size: int, img_w: int, img_h: int) -> Tensor:
        nl = len(target)
        device = target.device

        if nl == 0:
            out = torch.zeros(batch_size, 0, 5, device=device)
        else:
            batch_idx = target[:, 0]
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=device) - offsets[batch_idx]
            out[batch_idx, within_idx] = target[:, 1:]
            out[..., 1:5] = denormalize(cxcywh2xyxy(out[..., 1:5]), img_w, img_h)

        return out

    def box_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tensor:
        # anchor_points: (33600, 2)
        # pred_dist: (B, 33600, 4*reg_max)
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )

        return dist2bbox(pred_dist, anchor_points, False)

    def get_assigned_targets_and_loss(
        self, preds: tuple[Tensor, Tensor, list[Tensor]], batch: dict
    ):
        boxes, scores, feats = preds

        pred_distri = boxes.permute(0, 2, 1).contiguous()  # (B, 4R, 33600) -> (B, 33600, 4R)
        pred_scores = scores.permute(0, 2, 1).contiguous()  # (B, C, 33600) -> (B, 33600, C)

        anchor_points, stride_tensor = make_anchors(feats, self.strides)

        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.strides[0]

        # targets = torch.cat()

        loss = torch.zeros(3)
