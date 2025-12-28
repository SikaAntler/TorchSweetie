import torch
from torch import Tensor, nn

from ..layers.yolov5layers import C3, Conv


class Focus(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1
    ) -> None:
        super().__init__()

        self.conv = Conv(in_channels * 4, out_channels, kernel_size, stride, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(
            torch.concat(
                (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1
            )
        )


class SPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k=(5, 9, 13)) -> None:
        super().__init__()

        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)

        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class YOLOv5BackBone(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.stem = Focus(in_channels, 64, 3)

        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),
            C3(128, 128, 3),
        )

        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),
            C3(256, 256, 9),
        )

        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2, 1),
            C3(512, 512, 9),
        )

        self.stage4 = nn.Sequential(
            Conv(512, 1024, 3, 2, 1),
            SPP(1024, 1024),
            C3(1024, 1024, 3, False),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        out3 = self.stage2(x)
        out4 = self.stage3(out3)
        out5 = self.stage4(out4)

        return [out3, out4, out5]
