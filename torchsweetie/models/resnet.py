from typing import Optional

import torch
from rich import print
from torch import Tensor, nn
from torchvision.models import resnet

from ..data import ClsDataPack
from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "ResNet",
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


class ResNet(nn.Module):
    def __init__(self, model: resnet.ResNet, num_features: int, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        if num_classes != 0:
            if num_classes == model.fc.weight.shape[0]:
                self.fc = model.fc
            else:
                self.fc = nn.Linear(num_features, num_classes)

    def forward(self, data: ClsDataPack) -> Tensor:
        x = data.inputs

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

        if self.num_classes != 0:
            x = self.fc(x)

        return x


# def _check_weights_size(weights: dict, model_state_dict: dict, keys: list[str]) -> None:
#     for key in keys:
#         if key not in weights or key not in model_state_dict:
#             continue
#
#         if weights[key].shape != model_state_dict[key].shape:
#             weights.pop(key)


def _init_model(model_name: str, num_classes: int, weights: Optional[str] = None) -> ResNet:
    if weights == "torchvision":  # pretrained weights only for training
        pretrained_weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{pretrained_weights.url}{URL_E})",
        )
        _model = _resnet_models[model_name](weights=pretrained_weights)
        model = ResNet(_model, _num_features[model_name], num_classes)
    else:
        _model = _resnet_models[model_name]()
        model = ResNet(_model, _num_features[model_name], num_classes)
        if weights is not None:
            print(f"Loading weights from: {URL_B}{weights}{URL_E}")
            model.load_state_dict(torch.load(weights, "cpu"), False)

    return model


@MODELS.register()
def resnet18(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnet18", num_classes, weights)


@MODELS.register()
def resnet34(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnet34", num_classes, weights)


@MODELS.register()
def resnet50(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnet50", num_classes, weights)


@MODELS.register()
def resnet101(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnet101", num_classes, weights)


@MODELS.register()
def resnet152(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnet152", num_classes, weights)


@MODELS.register()
def resnext50_32x4d(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnext50_32x4d", num_classes, weights)


@MODELS.register()
def resnext101_32x8d(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnext101_32x8d", num_classes, weights)


@MODELS.register()
def resnext101_64x4d(num_classes: int, weights: Optional[str] = None) -> ResNet:
    return _init_model("resnext101_64x4d", num_classes, weights)
