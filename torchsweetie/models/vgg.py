from typing import Optional

from rich import print
from torch import nn
from torchvision.models import VGG, vgg

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "vgg16",
    "vgg19",
]

_pretrained_weights = {
    "vgg16": vgg.VGG16_Weights.DEFAULT,
    "vgg19": vgg.VGG19_Weights.DEFAULT,
}

_vgg_models = {
    "vgg16": vgg.vgg16,
    "vgg19": vgg.vgg19,
}


def _init_model(model_name: str, num_classes: int, dropout, weights: Optional[str] = None) -> VGG:
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _vgg_models[model_name](weights=_weights)
    else:
        model = _vgg_models[model_name]()

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(dropout),
    )
    if num_classes != 0:
        model.classifier.append(nn.Linear(4096, num_classes))

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def vgg16(num_classes: int, dropout: float = 0.5, weights: Optional[str] = None) -> VGG:
    return _init_model("vgg16", num_classes, dropout, weights)


@MODELS.register()
def vgg19(num_classes: int, dropout: float = 0.5, weights: Optional[str] = None) -> VGG:
    return _init_model("vgg19", num_classes, dropout, weights)
