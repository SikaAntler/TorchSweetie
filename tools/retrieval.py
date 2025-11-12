from argparse import ArgumentParser
from pathlib import Path

from torchsweetie.exporter import RetrievalExporter
from torchsweetie.tester import RetrievalTester


def main(cfg) -> None:
    exp_dir = Path(cfg.exp_dir)

    cfg_file: Path = exp_dir.parent / "config.yaml"
    if not cfg_file.exists():
        cfg_file = Path(cfg.cfg_file)

    if cfg.best:
        weights = "best-*[0-9].pth"
    elif cfg.last:
        weights = "last-*[0-9].pth"
    else:
        weights = f"epoch-{cfg.epoch}.pth"

    weights = list(exp_dir.glob(weights))
    assert len(weights) == 1
    weights = weights[0].name

    exporter = RetrievalExporter(cfg_file, exp_dir, weights)
    exporter.export()

    tester = RetrievalTester(cfg_file, exp_dir, weights)
    tester.test()

    tester.report(exporter.embeddings, exporter.labels, cfg.topk_list, cfg.digits)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--exp-dir",
        "--exp",
        type=str,
        required=True,
        help="path of the experimental directory (relative e.g. YYYYmmdd-HHMMSS)",
    )
    parser.add_argument(
        "--cfg-file",
        "--cfg",
        type=str,
        help="paht of the config file (relative)",
    )

    group_weights = parser.add_mutually_exclusive_group(required=True)
    group_weights.add_argument(
        "--best", action="store_true", help="whether to load the best weights"
    )
    group_weights.add_argument(
        "--last", action="store_true", help="whether to load the last weights"
    )
    group_weights.add_argument("--epoch", type=int, help="which epoch of weights want to load")

    parser.add_argument("--topk-list", "--topk", nargs="+", type=int, help="the list of k in topk")
    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")
    parser.add_argument("--export", action="store_true", help="whether to export the report")

    main(parser.parse_args())
