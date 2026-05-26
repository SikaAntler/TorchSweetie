from pathlib import Path

import torch
from rich import print
from torch import nn

from . import URL_B, URL_E


def load_weights(
    module: nn.Module, filename: Path | str, rm_ddp: bool = False, strict: bool = False
) -> None:
    print(f"Loading weights from: {URL_B}{filename}{URL_E}")
    weights = torch.load(filename, map_location="cpu")

    if rm_ddp:
        _weights = {}
        for name, param in weights.items():
            name = name.replace("modules", "")
            _weights[name] = param
        weights = _weights

    module.load_state_dict(weights, strict)


def load_weights_for_model(model: nn.Module, weights: str) -> None:
    print(f"Loading weights from: {URL_B}{weights}{URL_E}")

    model_dict = model.state_dict()
    weights_dict = torch.load(weights, "cpu")

    filtered_dict = {}
    loaded_weights = 0
    total_weights = 0
    loaded_params = 0
    total_params = 0

    for name, param in weights_dict.items():
        if name in model_dict and param.shape == model_dict[name].shape:
            filtered_dict[name] = param
            loaded_weights += 1
            loaded_params += param.numel()
        total_weights += 1
        total_params += param.numel()

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(
        f"Loaded: weights = {loaded_weights}/{total_weights}, params = {loaded_params:,}/{total_params:,}"
    )
