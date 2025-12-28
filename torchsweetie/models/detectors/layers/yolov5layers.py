import torch
from torch import Tensor, nn


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, shortcut: bool = True, expand: float = 0.5
    ) -> None:
        super().__init__()

        c_ = int(out_channels * expand)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_, out_channels, 3, 1, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, e: float = 0.5
    ) -> None:
        super().__init__()

        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, expand=1.0) for _ in range(n)))

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class DFL(nn.Module):
    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        b, _, a = x.shape

        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 4, a)
