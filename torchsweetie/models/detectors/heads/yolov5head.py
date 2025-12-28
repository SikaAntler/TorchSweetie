import math

import torch
from torch import Tensor, nn


class YOLOv5Head(nn.Module):
    def __init__(self, num_classes: int, anchors: list[list[int]], stride: list[int]) -> None:
        super().__init__()

        self.nc = num_classes
        self.no = 5 + num_classes
        self.na = len(anchors[0]) // 2
        self.stride = stride

        out_channels = self.no * 3
        self.m = nn.ModuleList(
            [
                nn.Conv2d(256, out_channels, 1),
                nn.Conv2d(512, out_channels, 1),
                nn.Conv2d(1024, out_channels, 1),
            ]
        )
        self._initialize_biases()

        self.register_buffer("anchors", torch.FloatTensor(anchors).reshape(3, -1, 2))
        self.anchors /= torch.FloatTensor(stride).view(-1, 1, 1)  # ty: ignore

        self.grid = [torch.empty(0) for _ in range(3)]
        self.anchor_grid = [torch.empty(0) for _ in range(3)]

    def forward(self, outs: list[Tensor]) -> Tensor | list[Tensor]:
        # outs[0]: (BS,  256, 80, 80)
        # outs[1]: (BS,  512, 40, 40)
        # outs[2]: (BS, 1024, 20, 20)
        z = []
        for i in range(3):
            x = outs[i]
            x = self.m[i](x)
            bs, _, ny, nx = x.shape
            # (bs, 255, ny, nx) -> (bs, 3, 85, ny, nx) -> (bs, 3, ny, nx, 85)
            x = x.view(bs, self.na, -1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.training:
                z.append(x)
            else:
                if self.grid[i].shape[2:4] != x.shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                #   xy: (bs, 3, ny, nx,  2)
                #   wh: (bs, 3, ny, nx,  2)
                # conf: (bs, 3, ny, nx, 81)
                xy, wh, conf = x.sigmoid().split((2, 2, self.nc + 1), 4)

                # (bs, 3, ny, nx, 2) * (1, 3, ny, nx, 2) = (bs, 3, ny, nx, 2)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]

                # (bs, 3, ny, nx, no)
                y = torch.concat([xy, wh, conf], 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            return z
        else:
            return torch.concat(z, 1)  # (bs, 3*nl, ny, nx, 85)

    def _make_grid(self, nx: int, ny: int, i: int) -> tuple[Tensor, Tensor]:
        device = self.anchors.device
        dtype = self.anchors.dtype
        shape = (1, self.na, ny, nx, 2)

        y = torch.arange(ny, dtype=dtype, device=device)  # (ny,)
        x = torch.arange(nx, dtype=dtype, device=device)  # (nx,)
        yv, xv = torch.meshgrid(y, x, indexing="ij")  # (ny, nx)

        # (ny, nx, 2) -> (1, 3, ny, nx, 2)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5

        # (3, 2) * 8/16/32 -> (1, 3, 1, 1, 2) -> (1, 3, ny, nx, 2)
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2))
        anchor_grid = anchor_grid.expand(shape)

        return grid, anchor_grid

    def _initialize_biases(self, cf=None):
        """Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + self.nc] += (
                math.log(0.6 / (self.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
