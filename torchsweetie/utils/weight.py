from pathlib import Path
from typing import Union

import torch
from rich import print
from torch import nn

from .color import URL_B, URL_E


def load_weights(module: nn.Module, filename: Union[Path, str], rm_ddp=False, strict=False) -> None:
    print(f"Loading weights from: {URL_B}{filename}{URL_E}")
    weights = torch.load(filename, map_location="cpu")

    if rm_ddp:
        _weights = {}
        for name, param in weights.items():
            name = name.replace("modules", "")
            _weights[name] = param
        weights = _weights

    module.load_state_dict(weights, strict)
