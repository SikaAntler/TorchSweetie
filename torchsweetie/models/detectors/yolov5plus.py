from torch import nn

from ...data import DetDataPack
from ...utils import MODELS
from .backbones import YOLOv5BackBone
from .heads import YOLOv5DecoupledHead
from .necks import YOLOv5Neck


@MODELS.register(scope="detection")
class YOLOv5Plus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        in_channels: int = 3,
        strides: list[int] = [8, 16, 32],
    ) -> None:
        super().__init__()

        self.backbone = YOLOv5BackBone(in_channels)
        self.neck = YOLOv5Neck()
        self.head = YOLOv5DecoupledHead(num_classes, reg_max, strides)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3  # ty: ignore
                m.momentum = 0.03  # ty: ignore
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # ty: ignore

    def forward(self, data: DetDataPack):
        x = data.images

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        if self.training:
            return x
        else:
            return x
