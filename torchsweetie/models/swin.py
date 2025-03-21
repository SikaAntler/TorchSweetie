from typing import Optional

from rich import print
from torch import nn
from torchvision.models import SwinTransformer, swin_transformer

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

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
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _swin_models[model_name](weights=_weights)
    else:
        model = _swin_models[model_name]()

    if num_classes == 0:
        model.head = nn.Identity()  # pyright: ignore
    else:
        model.head = nn.Linear(_num_features[model_name], num_classes)

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

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
