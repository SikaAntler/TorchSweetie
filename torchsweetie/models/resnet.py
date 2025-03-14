from typing import Optional

from rich import print
from torch import nn
from torchvision.models import ResNet, Weights, resnet

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
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


def _init_model(model_name: str, num_classes: int, weights: Optional[str] = None) -> ResNet:
    if weights == "torchvision":
        _weights: Weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _resnet_models[model_name](weights=_weights)
    else:
        model = _resnet_models[model_name]()

    if num_classes == 0:
        model.fc = nn.Identity()  # pyright: ignore
    else:
        model.fc = nn.Linear(_num_features[model_name], num_classes)

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

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
