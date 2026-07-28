import torch
from torch import Tensor, nn
from torchvision.models import resnet

from ...utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, print_main
from .cls_model import ClsModel

SCOPE = "classification"

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


class ResNet(nn.Module):
    def __init__(self, model: resnet.ResNet, num_features: int, remap: int | None) -> None:
        super().__init__()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        self.remap = nn.Linear(num_features, remap) if remap else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.remap:
            x = self.remap(x)

        return x


def _init_model(
    model_name: str, num_classes: int, pretrained: bool = False, remap: int | None = None
) -> ClsModel:
    if pretrained:
        pretrained_weights = _pretrained_weights[model_name]
        print_main(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{pretrained_weights.url}{URL_E})",
        )
    else:
        pretrained_weights = None

    _model = _resnet_models[model_name](weights=pretrained_weights)
    backbone = ResNet(_model, _num_features[model_name], remap)

    head = nn.Linear(_num_features[model_name], num_classes)

    return ClsModel(backbone, head)


@MODELS.register(scope=SCOPE)
def resnet18(num_classes: int, pretrained: bool = False, remap: int | None = None) -> ClsModel:
    return _init_model("resnet18", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnet34(num_classes: int, pretrained: bool = False, remap: int | None = None) -> ClsModel:
    return _init_model("resnet34", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnet50(num_classes: int, pretrained: bool = False, remap: int | None = None) -> ClsModel:
    return _init_model("resnet50", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnet101(num_classes: int, pretrained: bool = False, remap: int | None = None) -> ClsModel:
    return _init_model("resnet101", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnet152(num_classes: int, pretrained: bool = False, remap: int | None = None) -> ClsModel:
    return _init_model("resnet152", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnext50_32x4d(
    num_classes: int, pretrained: bool = False, remap: int | None = None
) -> ClsModel:
    return _init_model("resnext50_32x4d", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnext101_32x8d(
    num_classes: int, pretrained: bool = False, remap: int | None = None
) -> ClsModel:
    return _init_model("resnext101_32x8d", num_classes, pretrained, remap)


@MODELS.register(scope=SCOPE)
def resnext101_64x4d(
    num_classes: int, pretrained: bool = False, remap: int | None = None
) -> ClsModel:
    return _init_model("resnext101_64x4d", num_classes, pretrained, remap)
