from dataclasses import dataclass, field
from pathlib import Path

import torch
from rich import print
from rich.console import Console
from rich.table import Table
from torch import nn

from .color import URL_B, URL_E


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


@dataclass
class ShapeMismatchInfo:
    name: str
    model_shape: tuple
    weight_shape: tuple
    weight_params: int


@dataclass
class LoadStats:
    loaded_count: int = 0
    loaded_params: int = 0

    random_init_count: int = 0
    random_init_params: int = 0

    shape_mismatch_count: int = 0
    shape_mismatch_params: int = 0

    unexpected_count: int = 0
    unexpected_params: int = 0

    shape_mismatches: list[ShapeMismatchInfo] = field(default_factory=list)
    unexpected_names: list[str] = field(default_factory=list)


def load_weights_for_model(
    model: nn.Module, weights: str, verbose: bool = False, topk: int = 10
) -> None:
    if verbose:
        print(f"Loading weights from: {URL_B}{weights}{URL_E}")

    model_dict = model.state_dict()
    weights_dict = torch.load(weights, "cpu")

    stats = LoadStats()
    loadable_dict = {}

    # 统计预训练权重：loaded、shape mismatch、unexpected
    for name, param in weights_dict.items():
        if name not in model_dict:
            stats.unexpected_count += 1
            stats.unexpected_params += param.numel()

            stats.unexpected_names.append(name)

            continue

        if model_dict[name].shape != param.shape:
            stats.shape_mismatch_count += 1
            stats.shape_mismatch_params += param.numel()

            stats.shape_mismatches.append(
                ShapeMismatchInfo(
                    name, tuple(model_dict[name].shape), tuple(param.shape), param.numel()
                )
            )

            continue

        loadable_dict[name] = param
        stats.loaded_count += 1
        stats.loaded_params += param.numel()

    # 统计模型中未成功加载的参数
    loaded_names = set(loadable_dict.keys())
    for name, param in model_dict.items():
        if name not in loaded_names:
            stats.random_init_count += 1
            stats.random_init_params += param.numel()

    # 安全加载
    model_dict.update(loadable_dict)
    model.load_state_dict(model_dict, strict=True)

    if verbose and len(stats.shape_mismatches) != 0:
        console = Console()

        table = Table(title="Weight Loading Report", show_header=True, header_style="bold cyan")

        table.add_column("Category", style="green")
        table.add_column("Count", justify="right")
        table.add_column("Params", justify="right")

        table.add_row("Loaded", f"{stats.loaded_count:,}", f"{stats.loaded_params:,}")
        table.add_row(
            "Random Init", f"{stats.random_init_count:,}", f"{stats.random_init_params:,}"
        )
        table.add_row(
            "Shape Mismatch", f"{stats.shape_mismatch_count:,}", f"{stats.shape_mismatch_params:,}"
        )
        table.add_row("Unexpected", f"{stats.unexpected_count:,}", f"{stats.unexpected_params:,}")

        table.add_section()

        total_model_params = sum(p.numel() for p in model_dict.values())
        load_ratio = stats.loaded_params / total_model_params * 100

        table.add_row("Load Ratio", "-", f"{load_ratio:.2f}%", style="bold yellow")

        console.print(table)

        mismatch_table = Table(
            title=f"Top {min(topk, len(stats.shape_mismatches))} Shape Mismatched",
            header_style="bold red",
        )

        mismatch_table.add_column("Layer")
        mismatch_table.add_column("Weight Shape")
        mismatch_table.add_column("Model Shape")
        mismatch_table.add_column("Params", justify="right")

        mismatches = sorted(stats.shape_mismatches, key=lambda x: x.weight_params, reverse=True)

        for item in mismatches[:topk]:
            mismatch_table.add_row(
                item.name, str(item.weight_shape), str(item.model_shape), f"{item.weight_params:,}"
            )

        console.print(mismatch_table)
