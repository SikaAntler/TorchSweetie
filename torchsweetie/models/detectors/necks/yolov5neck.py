import torch
from torch import Tensor, nn

from ..layers.yolov5layers import C3, Conv


class YOLOv5Neck(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.reduce2 = Conv(1024, 512)

        self.upsample2 = nn.Upsample(scale_factor=2)

        self.topdown2 = nn.Sequential(
            C3(1024, 512, 3, False),
            Conv(512, 256),
        )

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.topdown1 = C3(512, 256, 3, False)

        self.downsample0 = Conv(256, 256, 3, 2, 1)

        self.bottomup0 = C3(512, 512, 3, False)

        self.downsample1 = Conv(512, 512, 3, 2, 1)

        self.bottomup1 = C3(1024, 1024, 3, False)

    def forward(self, outs: list[Tensor]) -> list[Tensor]:
        assert len(outs) == 3, "does not support p6 now"
        out3, out4, out5 = outs
        # out3: (BS,  256, 80, 80)
        # out4: (BS,  512, 40, 40)
        # out5: (BS, 1024, 20, 20)

        x5 = self.reduce2(out5)  # (BS, 512, 20, 20)

        x = self.upsample2(x5)  # (BS, 512, 40, 40)
        x = torch.concat([x, out4], 1)  # (BS, 1024, 40, 40)

        x4 = self.topdown2(x)  # (BS, 512, 40, 40)

        x = self.upsample1(x4)  # (BS, 256, 80, 80)
        x = torch.concat([x, out3], 1)  # (BS, 512, 80, 80)

        out3 = self.topdown1(x)  # (BS, 256, 80, 80) OUTPUT

        x = self.downsample0(out3)  # (BS, 256, 40, 40)
        x = torch.concat([x, x4], 1)  # (BS, 512, 40, 40)

        out4 = self.bottomup0(x)  # (BS, 512, 40, 40) OUTPUT

        x = self.downsample1(out4)  # (BS, 512, 20, 20)
        x = torch.concat([x, x5], 1)  # (BS, 1024, 20, 20)

        out5 = self.bottomup1(x)  # (BS, 1024, 20, 20)  OUTPUT

        return [out3, out4, out5]
