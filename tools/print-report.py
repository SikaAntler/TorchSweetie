from argparse import ArgumentParser
from pathlib import Path

from torchsweetie.utils import print_report, print_report_old


def main(cfg) -> None:
    filename = Path(cfg.exp_name).absolute() / "report.csv"

    if cfg.old:
        print_report_old(filename, cfg.digits)
    else:
        print_report(filename, cfg.digits, cfg.interval)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "exp_name",
        type=str,
        help="path of the experimental directory (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )
    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")
    parser.add_argument("--interval", default=0, type=int, help="interval for adding an empty line")
    parser.add_argument("--old", action="store_true", help="whether to print in old format")

    main(parser.parse_args())
