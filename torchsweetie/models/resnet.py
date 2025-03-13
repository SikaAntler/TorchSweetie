from pathlib import Path
from typing import Optional

from rich import print
from torch import nn
from torchvision.models import Weights, resnet

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "resnet18",
    "resnet18_fe",
    "resnet34",
    "resnet34_fe",
    "resnet50",
    "resnet50_fe",
    "resnet101",
    "resnet101_fe",
    "resnet152",
    "resnet152_fe",
    "resnext50_32x4d",
    "resnext50_32x4d_fe",
    "resnext101_32x8d",
    "resnext101_32x8d_fe",
    "resnext101_64x4d",
    "resnext101_64x4d_fe",
]

_pretrained_weights = {
    "resnet18": resnet.ResNet18_Weights.DEFAULT,
    "resnet34": resnet.ResNet34_Weights.DEFAULT,
    "resnet50": resnet.ResNet50_Weights.DEFAULT,
    "resnet101": resnet.ResNet101_Weights.DEFAULT,
    "resnet152": resnet.ResNet152_Weights.DEFAULT,
    "resnext50_32x4d": resnet.ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d": resnet.ResNeXt101_32X8D_Weights.DEFAULT,
    "resnext101_64x4d": resnet.ResNeXt101_64X4D_Weights.DEFAULT,
}

_resnet_models = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet101": resnet.resnet101,
    "resnet152": resnet.resnet152,
    "resnext50_32x4d": resnet.resnext50_32x4d,
    "resnext101_32x8d": resnet.resnext101_32x8d,
    "resnext101_64x4d": resnet.resnext101_64x4d,
}

_num_features = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,
    "resnext101_64x4d": 2048,
}


class ResNetFE(nn.Module):
    def __init__(
        self,
        conv1: nn.Module,
        bn1: nn.Module,
        relu: nn.Module,
        maxpool: nn.Module,
        layer1: nn.Module,
        layer2: nn.Module,
        layer3: nn.Module,
        layer4: nn.Module,
        avgpool: nn.Module,
        num_features: int,
        remap: int | None,
    ) -> None:
        super().__init__()

        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.maxpool = maxpool
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.avgpool = avgpool

        if remap is not None:
            self.remap = nn.Linear(num_features, remap)
        else:
            self.remap = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        if self.remap is not None:
            x = self.remap(x)

        return x


def _init_model(
    model_name: str,
    num_classes: int,
    weights: Optional[str | Path] = None,
    fe: bool = False,
    remap: int | None = None,
) -> nn.Module:
    # TODO: 如果用M个类别做的预训练权重给N个类别用，怎么处理？
    if weights == "torchvision":
        _weights: Weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _resnet_models[model_name](weights=_weights)
        if num_classes != 1000:
            model.fc = nn.Linear(_num_features[model_name], num_classes)
    else:
        model = _resnet_models[model_name](num_classes=num_classes)

    if fe:
        model = ResNetFE(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            _num_features[model_name],
            remap,
        )

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def resnet18(num_classes: int, weights: Optional[str | Path] = None) -> nn.Module:
    return _init_model("resnet18", num_classes, weights)


@MODELS.register()
def resnet18_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnet18", num_classes, weights, True, remap)


@MODELS.register()
def resnet34(num_classes: int, weights: Optional[str | Path] = None) -> nn.Module:
    return _init_model("resnet34", num_classes, weights)


@MODELS.register()
def resnet34_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnet34", num_classes, weights, True, remap)


@MODELS.register()
def resnet50(num_classes: int, weights: Optional[str | Path] = None) -> nn.Module:
    return _init_model("resnet50", num_classes, weights)


@MODELS.register()
def resnet50_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnet50", num_classes, weights, True, remap)


@MODELS.register()
def resnet101(num_classes: int, weights=None) -> nn.Module:
    return _init_model("resnet101", num_classes, weights)


@MODELS.register()
def resnet101_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnet101", num_classes, weights, True, remap)


@MODELS.register()
def resnet152(num_classes: int, weights: Optional[str | Path] = None) -> nn.Module:
    return _init_model("resnet152", num_classes, weights)


@MODELS.register()
def resnet152_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnet152e", num_classes, weights, True, remap)


@MODELS.register()
def resnext50_32x4d(
    num_classes: int, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext50_32x4d", num_classes, weights)


@MODELS.register()
def resnext50_32x4d_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext50_32x4d", num_classes, weights, True, remap)


@MODELS.register()
def resnext101_32x8d(
    num_classes: int, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext101_32x8d", num_classes, weights)


@MODELS.register()
def resnext101_32x8d_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext101_32x8d", num_classes, weights, True, remap)


@MODELS.register()
def resnext101_64x4d(
    num_classes: int, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext101_64x4d", num_classes, weights)


@MODELS.register()
def resnext101_64x4d_fe(
    num_classes: int, remap=None, weights: Optional[str | Path] = None
) -> nn.Module:
    return _init_model("resnext101_64x4d", num_classes, weights, True, remap)
