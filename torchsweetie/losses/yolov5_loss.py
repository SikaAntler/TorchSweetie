import torch
from torch import Tensor, nn

from ..data import DetDataPack
from ..utils import LOSSES, bbox_iou


@LOSSES.register(scope="detection")
class YOLOv5Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: list[list[int]],
        stride: list[int],
        box: float,
        cls: float,
        cls_pw: float,
        obj: float,
        obj_pw: float,
        anchor_t: float,
    ) -> None:
        super().__init__()

        self.nc: int = num_classes
        self.na: int = len(anchors[0]) // 2
        self.box: float = box
        self.cls: float = cls
        self.obj: float = obj
        self.anchor_t: float = anchor_t

        self.register_buffer("anchors", torch.FloatTensor(anchors).reshape(3, -1, 2))
        self.anchors /= torch.FloatTensor(stride).view(-1, 1, 1)  # ty: ignore

        self.loss_fn_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw]))
        self.loss_fn_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))

        self.balance: list[float] = [4.0, 1.0, 0.4]

    def forward(self, p: Tensor, data: DetDataPack) -> dict[str, Tensor]:
        # outs[0]: (BS, 3, 80, 80, 85)
        # outs[1]: (BS, 3, 40, 40, 85)
        # outs[2]: (BS, 3, 20, 20, 85)

        # targets: (N, 6), where 6 = [image, class, cx, cy, w, h]
        targets = data.labels

        device = targets.device
        lbox = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        classes, boxes, indices, anchors = self.build_targets(p, targets)

        # the three losses
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]  # image, anchor, grid_j, grid_i
            # (bs, 3, 64, 64)
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            if n := len(b):
                # (T, 9) -> (T, 2), (T, 2), (T, 1), (T, 4)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)

                # regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.concat((pxy, pwh), 1)  # (T, 4)
                iou = bbox_iou(pbox, boxes[i]).squeeze()  # (T, )
                lbox += (1.0 - iou).mean()  # iou loss

                # objectness
                iou = iou.detach().clamp(0)  # used for loss before, could be negative
                tobj[b, a, gj, gi] = iou  # use iou as box confidence

                # classification
                if self.nc > 1:
                    t = torch.full_like(pcls, 0, device=device)  # (T, 4)
                    t[range(n), classes[i]] = 1  # (T, 4)
                    lcls += self.loss_fn_cls(pcls, t)

            obji = self.loss_fn_obj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        # three losses multiple their coefficient
        bs = tobj.shape[0]
        lbox *= self.box * bs
        lcls *= self.cls * bs
        lobj *= self.obj * bs

        return {"box": lbox, "cls": lcls, "obj": lobj}

    def build_targets(self, p: Tensor, targets: Tensor):
        dtype = targets.dtype
        device = targets.device
        nt = len(targets)

        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, dtype=torch.long, device=device)  # (7, )

        # each layer has 3 anchors, the indices are 0, 1, 2
        anchor_idx = torch.arange(3, dtype=dtype, device=device)  # (3,)
        anchor_idx = anchor_idx.view(3, 1).repeat(1, nt)  # (3, 1) -> (3, N)

        # [(N, 6) -> (3, N, 6)] concat [(3, N) -> (3, N, 1)] -> (3, N, 7)
        targets = torch.concat([targets.repeat(3, 1, 1), anchor_idx[..., None]], 2)

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],  # center
                [1, 0],  # left
                [0, 1],  # top
                [-1, 0],  # right
                [0, -1],  # bottom
            ],
            dtype=dtype,
            device=device,
        )
        off *= g  # offsets: (5, 2)

        for i in range(3):
            anchors = self.anchors[i]  # (3, 2)
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xywh gain

            # Match targets to anchors
            t = targets * gain  # (3, N, 7), where 7 = [i, c, cx, cy, w, h, a]
            if nt:
                # filter the objects with large width or height than anchors'
                wh_ratio = t[..., 4:6] / anchors[:, None]  # (3, N, 2)
                f = torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.anchor_t  # (3, N)
                t = t[f]  # (M, 7), where M is the number of remained

                # offsets
                gxy = t[:, 2:4]  # (M, 2), where 2 = (x, y)
                gxy_i = gain[[2, 3]] - gxy  # (M, 2) diagonal inverse
                left, top = ((gxy % 1 < g) & (gxy > 1)).T  # (2, M) -> (M,), (M,)
                right, bottom = ((gxy_i % 1 < g) & (gxy_i > 1)).T  # (2, M) -> (M,), (M,)
                f = torch.stack((torch.ones_like(left), left, top, right, bottom))  # (5, M)
                t = t.repeat(5, 1, 1)[f]  # (M, 7) -> (5, M, 7) -> (T, 7)
                # (1, M, 2) + (5, 1, 2) = (5, M, 2) -> (T, 2)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[f]
            else:
                t = targets[0]  # (N, 7) = (0, 7)
                offsets = 0

            # define
            b, c, gxy, gwh, a = t.tensor_split((1, 2, 4, 6), dim=1)
            b = b.long().view(-1)  # (T, )
            c = c.long().view(-1)  # (T, )
            gij = (gxy - offsets).long()  # (T, 2)
            gi, gj = gij.T  # grid indices:  (T, )
            a = a.long().view(-1)

            # append
            tcls.append(c)  # [(T, )]
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # [(T, 4)]
            indices.append(
                (
                    b,
                    a,
                    gj.clamp_(0, shape[2] - 1),
                    gi.clamp_(0, shape[3] - 1),
                )
            )  # indices of images, anchors, grid_j, grid_i
            anch.append(anchors[a])  # anchors [(T, 2)]

        return tcls, tbox, indices, anch
