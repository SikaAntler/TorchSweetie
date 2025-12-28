from argparse import ArgumentParser
from pathlib import Path

from torchsweetie.utils import print_cls_report, print_det_report, print_report_old


def main(cfg) -> None:
    filename = Path(cfg.exp_dir).absolute() / "report.csv"

    match cfg.task:
        case "classification":
            if cfg.old:
                print_report_old(filename, cfg.digits)
            else:
                print_cls_report(filename, cfg.digits, cfg.interval)
        case "detection":
            print_det_report(filename, cfg.digits)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["classification", "detection"],
        default="classification",
        help="type of the training task",
    )

    parser.add_argument(
        "--exp-dir",
        "--exp",
        type=str,
        required=True,
        help="path of the experimental directory (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )

    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")

    parser.add_argument("--interval", default=0, type=int, help="interval for adding an empty line")

    parser.add_argument(
        "--old", action="store_true", help="whether to print in old format (classification only)"
    )

    main(parser.parse_args())
