from argparse import ArgumentParser
from pathlib import Path

from torchsweetie.exporter import ClsExporter


def main(cfg) -> None:
    root_dir = Path.cwd()
    cfg_file = root_dir / cfg.cfg_file
    exp_dir = root_dir / cfg.run_dir / cfg_file.stem / cfg.exp_dir
    assert exp_dir.exists()

    if cfg.best:
        weights = "best-*[0-9].pth"
    elif cfg.last:
        weights = "last-*[0-9].pth"
    else:
        weights = f"epoch-{cfg.epoch}.pth"
    weights = list(exp_dir.glob(weights))
    assert len(weights) == 1
    weights = weights[0].name

    exporter = ClsExporter(cfg.cfg_file, cfg.run_dir, cfg.exp_dir, weights)

    exporter.export_onnx(
        tuple(cfg.input_size),
        cfg.half,
        cfg.device if cfg.device == "cpu" else int(cfg.device),
        cfg.onnx_file,
        cfg.dynamic_batch_size,
        cfg.simplify,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--cfg-file",
        "--cfg",
        type=str,
        required=True,
        help="path of the config file (relative)",
    )
    parser.add_argument(
        "--run-dir",
        "--run",
        default="runs",
        type=str,
        help="path of the running directory (relative)",
    )
    parser.add_argument(
        "--exp-dir",
        "--exp",
        type=str,
        required=True,
        help="path of the experimental directory (relative e.g. YYYYmmdd-HHMMSS)",
    )

    group_weights = parser.add_mutually_exclusive_group()
    group_weights.add_argument(
        "--best", action="store_true", help="whether to load the best weights"
    )
    group_weights.add_argument(
        "--last", action="store_true", help="whether to load the last weights"
    )
    group_weights.add_argument(
        "--epoch", type=int, help="which epoch of weights want to load"
    )

    parser.add_argument(
        "--input-size",
        "--size",
        nargs="+",
        type=int,
        required=True,
        help="the input size for dummy input (e.g. 32 3 320 320)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="the dtype for the input and model",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="running device, e.g. cpu or integer for cuda",
    )
    parser.add_argument(
        "--onnx-file",
        default=None,
        type=str,
        help="the output onnx file (relative)",
    )
    parser.add_argument(
        "--dynamic-batch-size",
        "--dynamic",
        action="store_true",
        help="whether using dynamic batch size",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="whether to simplify the onnx model",
    )

    main(parser.parse_args())
