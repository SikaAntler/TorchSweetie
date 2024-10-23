from argparse import ArgumentParser

from torchsweetie.exporter import ClsExporter


def main(cfg) -> None:
    exporter = ClsExporter(
        cfg.root_dir,
        cfg.cfg_file,
        cfg.run_dir,
        cfg.exp_dir,
        cfg.weights,
    )
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
        "--root-dir",
        "--root",
        type=str,
        required=True,
        help="path of the roor directory",
    )
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
        help="path of the experiment directory (relative e.g. YYYYmmdd-HHMMSS)",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="path of the weights (relative)"
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
        required=True,
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
