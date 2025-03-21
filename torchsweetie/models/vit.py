from collections import OrderedDict
from typing import Optional

from rich import print
from torch import nn
from torchvision.models import VisionTransformer, vision_transformer

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights

__all__ = [
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]

_pretrained_weights = {
    "vit_b_16": vision_transformer.ViT_B_16_Weights.DEFAULT,
    "vit_b_32": vision_transformer.ViT_B_32_Weights.DEFAULT,
    "vit_l_16": vision_transformer.ViT_L_16_Weights.DEFAULT,
    "vit_l_32": vision_transformer.ViT_L_32_Weights.DEFAULT,
    "vit_h_14": vision_transformer.ViT_H_14_Weights.DEFAULT,
}

_vit_models = {
    "vit_b_16": vision_transformer.vit_b_16,
    "vit_b_32": vision_transformer.vit_b_32,
    "vit_l_16": vision_transformer.vit_l_16,
    "vit_l_32": vision_transformer.vit_l_32,
    "vit_h_14": vision_transformer.vit_h_14,
}

_hidden_dims = {
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024,
    "vit_h_14": 1280,
}


def _init_model(
    model_name: str, num_classes: int, weights: Optional[str] = None
) -> VisionTransformer:
    if weights == "torchvision":
        _weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{_weights.url}{URL_E})",
        )
        model = _vit_models[model_name](weights=_weights)
    else:
        model = _vit_models[model_name]()

    if num_classes == 0:
        model.heads = nn.Identity()  # pyright: ignore
    else:
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(_hidden_dims[model_name], num_classes)
        model.heads = nn.Sequential(heads_layers)

    if weights != "torchvision" and weights is not None:
        load_weights(model, weights)

    return model


@MODELS.register()
def vit_b_16(num_classes: int, weights: Optional[str] = None) -> VisionTransformer:
    return _init_model("vit_b_16", num_classes, weights)


@MODELS.register()
def vit_b_32(num_classes: int, weights: Optional[str] = None) -> VisionTransformer:
    return _init_model("vit_b_32", num_classes, weights)


@MODELS.register()
def vit_l_16(num_classes: int, weights: Optional[str] = None) -> VisionTransformer:
    return _init_model("vit_l_16", num_classes, weights)


@MODELS.register()
def vit_l_32(num_classes: int, weights: Optional[str] = None) -> VisionTransformer:
    return _init_model("vit_l_32", num_classes, weights)


@MODELS.register()
def vit_h_14(num_classes: int, weights: Optional[str] = None) -> VisionTransformer:
    return _init_model("vit_h_14", num_classes, weights)
