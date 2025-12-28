from argparse import ArgumentParser
from pathlib import Path

import torch

weights_map = {
    "model.0.": "backbone.stem.",
    "model.1.": "backbone.stage1.0.",
    "model.2.": "backbone.stage1.1.",
    "model.3.": "backbone.stage2.0.",
    "model.4.": "backbone.stage2.1.",
    "model.5.": "backbone.stage3.0.",
    "model.6.": "backbone.stage3.1.",
    "model.7.": "backbone.stage4.0.",
    "model.8.": "backbone.stage4.1.",
    "model.9.": "backbone.stage4.2.",
    "model.10.": "neck.reduce2.",
    "model.13.": "neck.topdown2.0.",
    "model.14.": "neck.topdown2.1.",
    "model.17.": "neck.topdown1.",
    "model.18.": "neck.downsample0.",
    "model.20.": "neck.bottomup0.",
    "model.21.": "neck.downsample1.",
    "model.23.": "neck.bottomup1.",
    "model.24.": "head.",
}

ignore = ["model.24.anchor_grid"]


def main(cfg) -> None:
    weights_file = Path(cfg.weights_file)
    weights = torch.load(weights_file, "cpu", weights_only=True)

    ts_weights = {}
    for name, param in weights.items():
        name: str
        if name in ignore:
            continue
        for n in weights_map:
            if name.startswith(n):
                ts_name = name.replace(n, weights_map[n])
                ts_weights[ts_name] = param
                break

    assert len(weights) - len(ignore) == len(ts_weights)

    torch.save(ts_weights, weights_file.with_name(f"{weights_file.stem}_ts.pth"))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("weights_file", type=str)

    main(parser.parse_args())
