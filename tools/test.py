from argparse import ArgumentParser

import torch

from torchsweetie.tester import ClsTester


def main(cfg) -> None:
    tester = ClsTester(
        cfg.root_dir,
        cfg.cfg_file,
        cfg.run_dir,
        cfg.exp_dir,
        cfg.weights,
    )

    tester.test()

    tester.report(cfg.digits, cfg.export)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root-dir",
        "--root",
        type=str,
        required=True,
        help="path of the root directory",
    )
    parser.add_argument(
        "--cfg-file",
        "--cfg",
        type=str,
        required=True,
        help="paht of the config file (relative)",
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
        "--digits", default=3, type=int, help="digits remain for accuracy"
    )
    parser.add_argument(
        "--export", action="store_true", help="whether to export the report"
    )

    main(parser.parse_args())
