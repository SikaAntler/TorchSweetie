from pathlib import Path
from typing import Optional

from rich import print
from torch import nn
from torchvision.models import EfficientNet, efficientnet

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "efficientnet_v2_s",
    "efficientnet_v2_s_fe",
    "efficientnet_v2_m",
    "efficientnet_v2_m_fe",
    "efficientnet_v2_l",
    "efficientnet_v2_l_fe",
]

_pretrained_weights = {
    "s": efficientnet.EfficientNet_V2_S_Weights.DEFAULT,
    "m": efficientnet.EfficientNet_V2_M_Weights.DEFAULT,
    "l": efficientnet.EfficientNet_V2_L_Weights.DEFAULT,
}

_efficientnet_models = {
    "s": efficientnet.efficientnet_v2_s,
    "m": efficientnet.efficientnet_v2_m,
    "l": efficientnet.efficientnet_v2_l,
}

_num_features = {
    "s": 1280,
    "m": 1280,
    "l": 1280,
}


def _init_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    weights: Optional[str | Path] = None,
    fe: bool = False,
) -> EfficientNet:
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _efficientnet_models[model_name](weights=_weights)
        if num_classes != 1000:
            model.classifier = nn.Sequential(
                nn.Dropout(dropout, True),
                nn.Linear(_num_features[model_name], num_classes),
            )
    else:
        model = _efficientnet_models[model_name](num_classes=num_classes)

    if fe:
        model.classifier = nn.Sequential(
            nn.Dropout(dropout, True),
        )

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def efficientnet_v2_s(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("s", num_classes, 0.2, weights)


@MODELS.register()
def efficientnet_v2_s_fe(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("s", num_classes, 0.2, weights, True)


@MODELS.register()
def efficientnet_v2_m(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("m", num_classes, 0.3, weights)


@MODELS.register()
def efficientnet_v2_m_fe(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("m", num_classes, 0.3, weights, True)


@MODELS.register()
def efficientnet_v2_l(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("l", num_classes, 0.4, weights)


@MODELS.register()
def efficientnet_v2_l_fe(num_classes: int, weights: Optional[str | Path] = None) -> EfficientNet:
    return _init_model("l", num_classes, 0.4, weights, True)
