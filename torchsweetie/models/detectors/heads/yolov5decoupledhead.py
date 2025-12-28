import math

import torch
from torch import Tensor, nn

from ..layers.yolov5layers import DFL, Conv


class YOLOv5DecoupledHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        strides: list[int] = [8, 16, 32],
        num_channels: list[int] = [256, 512, 1024],
    ) -> None:
        super().__init__()

        self.nc = num_classes
        self.nl = 3
        self.reg_max = reg_max
        self.no = num_classes + reg_max * 4
        self.strides = strides

        c2 = max((16, num_channels[0] // 4, reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in num_channels
        )

        c3 = max(num_channels[0], min(num_classes, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1))
            for x in num_channels
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self.bias_init()

    def bias_init(self) -> None:
        for i, (a, b) in enumerate(zip(self.cv2, self.cv3)):
            a[-1].bias.data[:] = 2.0  # ty: ignore
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.strides[i]) ** 2)  # ty: ignore

    def forward(self, feats: list[Tensor]):
        # outs[0]: (bs,  256, 80, 80)
        # outs[1]: (bs,  512, 40, 40)
        # outs[2]: (bs, 1024, 20, 20)
        boxes = []
        scores = []

        for feat in feats:
            bs = feat.shape[0]
            box = self.cv2(feat)  # (bs, 4, 80, 80)
            boxes.append(box.reshape(bs, 4 * self.reg_max, -1))
            score = self.cv3(feat)  # (bs, 1, 80, 80)
            scores.append(score.reshape(bs, 1, -1))

        boxes = torch.cat(boxes, -1)
        scores = torch.cat(scores, -1)

        if self.training:
            return boxes, scores, feats
        else:
            boxes, scores
