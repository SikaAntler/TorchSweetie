import torch
from torch import Tensor, nn

from ..utils import bbox_iou, cxcywh2xyxy, xyxy2cxcywh


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        strides: list[int] = [8, 16, 32],
        eps: float = 1e-9,
    ) -> None:
        super().__init__()

        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.strides = strides
        self.strides_val = strides[1] if len(self.strides) > 1 else strides[0]
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # pd_scores: (bs, na, nc)
        # pd_bboxes: (bs, na, 4)
        # anc_points: (na, 2)
        # gt_labels: (bs, nb, 1)
        # gt_bboxes: (bs, nb, 4)

        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        msak_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(msak_pos, overlaps)

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(-1, True)  # (bs, max_num_obj)
        pos_overlaps = (overlaps * mask_pos).amax(-1, True)  # (bs, max_num_obj)
        norm_align_metrics = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metrics

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        anc_points: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(
            align_metric, mask_gt.expand(-1, -1, self.topk).bool()
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_candidates_in_gts(
        self, xy_centers: Tensor, gt_bboxes: Tensor, mask_gt: Tensor
    ) -> Tensor:
        gt_bboxes_cxcywh = xyxy2cxcywh(gt_bboxes)
        wh_mask = gt_bboxes_cxcywh[..., 2:] < self.strides[0]
        gt_bboxes_cxcywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.strides_val, gt_bboxes_cxcywh.dtype, gt_bboxes_cxcywh.device),
            gt_bboxes_cxcywh[..., 2:],
        )
        gt_bboxes = cxcywh2xyxy(gt_bboxes_cxcywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
            bs, n_boxes, n_anchors, -1
        )

        return bbox_deltas.amin(3).gt_(self.eps)

    def get_box_metrics(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        get_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(self.bs).view(-1, -1).expand(-1, self.n_max_boxes)
        ind[1] = get_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes: Tensor, pd_bboxes: Tensor) -> Tensor:
        return bbox_iou(gt_bboxes, pd_bboxes, False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics: Tensor, topk_mask=None) -> Tensor:
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, -1, True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def select_highest_overlaps(
        self, mask_pos: Tensor, overlaps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # (bs, nb, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            # (bs, nb, h*w)
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, self.n_max_boxes, -1)

            max_overlaps_idx = overlaps.argmax(1)  # (bs, h*w)
            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            # (bs, nb, h*w)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()

            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)  # (bs, h*w)

        return target_gt_idx, fg_mask, mask_pos

    def get_targets(
        self, gt_labels: Tensor, gt_bboxes: Tensor, target_gt_idx: Tensor, fg_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # assigned target labels: (bs, 1)
        batch_ind = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (bs, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (bs, h*w)

        # assigned target bboxes: (bs, max_num_obj, 4) -> (bs, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # assigned target scores
        target_labels.clamp_(0)

        # (bs, h*w, nc)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (bs, h*w, nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
