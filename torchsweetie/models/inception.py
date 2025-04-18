from typing import Optional

from rich import print
from torch import nn
from torchvision.models import Inception3, inception

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "inception_v3",
]

_pretrained_weights = {
    "inception_v3": inception.Inception_V3_Weights.DEFAULT,
}

_inception_models = {
    "inception_v3": inception.inception_v3,
}


def _init_inception(model_name: str, num_classes: int, weights: Optional[str] = None) -> Inception3:
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _inception_models[model_name](weights=_weights)
    else:
        model = _inception_models[model_name]()

    if num_classes == 0:
        model.fc = nn.Identity()  # pyright: ignore
    else:
        model.fc = nn.Linear(2048, num_classes)

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def inception_v3(num_classes: int, weights: Optional[str] = None) -> Inception3:
    return _init_inception("inception_v3", num_classes, weights)
