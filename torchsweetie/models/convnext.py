from typing import Optional

from rich import print
from torch import nn
from torchvision.models import ConvNeXt, convnext

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]

_pretrained_weights = {
    "tiny": convnext.ConvNeXt_Tiny_Weights.DEFAULT,
    "small": convnext.ConvNeXt_Small_Weights.DEFAULT,
    "base": convnext.ConvNeXt_Base_Weights.DEFAULT,
    "large": convnext.ConvNeXt_Large_Weights.DEFAULT,
}

_convnext_model = {
    "tiny": convnext.convnext_tiny,
    "small": convnext.convnext_small,
    "base": convnext.convnext_base,
    "large": convnext.convnext_large,
}

_num_features = {
    "tiny": 768,
    "small": 768,
    "base": 1024,
    "large": 1536,
}


def _init_model(model_name: str, num_classes: int, weights: Optional[str] = None) -> ConvNeXt:
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _convnext_model[model_name](weights=_weights)
    else:
        model = _convnext_model[model_name]()

    if num_classes == 0:
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1])
    else:
        model.classifier[2] = nn.Linear(_num_features[model_name], num_classes)

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def convnext_tiny(num_classes: int, weights: Optional[str] = None) -> ConvNeXt:
    return _init_model("tiny", num_classes, weights)


@MODELS.register()
def convnext_small(num_classes: int, weights: Optional[str] = None) -> ConvNeXt:
    return _init_model("small", num_classes, weights)


@MODELS.register()
def convnext_base(num_classes: int, weights: Optional[str] = None) -> ConvNeXt:
    return _init_model("base", num_classes, weights)


@MODELS.register()
def convnext_large(num_classes: int, weights: Optional[str] = None) -> ConvNeXt:
    return _init_model("large", num_classes, weights)
