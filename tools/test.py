from argparse import ArgumentParser
from pathlib import Path

from torchsweetie.tester import ClsTester


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

    tester = ClsTester(cfg.cfg_file, cfg.run_dir, cfg.exp_dir, weights)
    tester.test()
    tester.report(cfg.digits, cfg.export)


if __name__ == "__main__":
    parser = ArgumentParser()

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
        help="path of the experimental directory (relative e.g. YYYYmmdd-HHMMSS)",
    )

    group_weights = parser.add_mutually_exclusive_group()
    group_weights.add_argument(
        "--best", action="store_true", help="whether to load the best weights"
    )
    group_weights.add_argument(
        "--last", action="store_true", help="whether to load the last weights"
    )
    group_weights.add_argument("--epoch", type=int, help="which epoch of weights want to load")

    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")
    parser.add_argument("--export", action="store_true", help="whether to export the report")

    main(parser.parse_args())
