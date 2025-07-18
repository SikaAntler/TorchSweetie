from typing import Optional

import torch
from rich import print
from torch import nn
from torchvision.models import SwinTransformer, swin_transformer

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E

__all__ = [
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]

_pretrained_weights = {
    "swin_v2_t": swin_transformer.Swin_V2_T_Weights.DEFAULT,
    "swin_v2_s": swin_transformer.Swin_V2_S_Weights.DEFAULT,
    "swin_v2_b": swin_transformer.Swin_V2_B_Weights.DEFAULT,
}

_swin_models = {
    "swin_v2_t": swin_transformer.swin_v2_t,
    "swin_v2_s": swin_transformer.swin_v2_s,
    "swin_v2_b": swin_transformer.swin_v2_b,
}

_num_features = {  # embed_dim * 2 ** (len(depths) - 1)
    "swin_v2_t": 768,  # 96 * 2^3
    "swin_v2_s": 768,  # 96 * 2^3
    "swin_v2_b": 1024,  # 128 * 2^3
}


def _init_model(
    model_name: str, num_classes: int, weights: Optional[str] = None
) -> SwinTransformer:
    if weights is None:
        model = _swin_models[model_name](num_classes=num_classes)
    elif weights == "torchvision":
        pretrained_weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{pretrained_weights.url}{URL_E})",
        )
        model = _swin_models[model_name](weights=pretrained_weights)
        if num_classes not in [0, 1000]:
            model.head = nn.Linear(_num_features[model_name], num_classes)
    else:
        _weights = torch.load(weights, map_location="cpu")
        print(f"Loading weights from: {URL_B}{weights}{URL_E}")
        model = _swin_models[model_name](num_classes=num_classes)

        model.load_state_dict(_weights, False)

    if num_classes == 0:
        model.head == nn.Identity()  # pyright: ignore

    return model


@MODELS.register()
def swin_v2_t(num_classes: int, weights: Optional[str] = None) -> SwinTransformer:
    return _init_model("swin_v2_t", num_classes, weights)


@MODELS.register()
def swin_v2_s(num_classes: int, weights: Optional[str] = None) -> SwinTransformer:
    return _init_model("swin_v2_s", num_classes, weights)


@MODELS.register()
def swin_v2_b(num_classes: int, weights: Optional[str] = None) -> SwinTransformer:
    return _init_model("swin_v2_b", num_classes, weights)
