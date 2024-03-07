from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from rich import print


def load_weights(
    module: nn.Module, filename: Union[Path, str], rm_ddp=False, strict=False
) -> None:
    # print(
    #     f"Loading [bold magenta]weights[/bold magenta]",
    #     f"from: [bold cyan]{filename}[/bold cyan]",
    # )
    print(f"Loading weights from: {filename}")
    weights = torch.load(filename, map_location="cpu")

    if rm_ddp:
        _weights = {}
        for name, param in weights.items():
            name = name.replace("modules", "")
            _weights[name] = param
        weights = _weights

    module.load_state_dict(weights, strict)
