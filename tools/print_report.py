from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from _utils import display_len, format_string
from rich import print


def main(cfg) -> None:
    ROOT = Path().cwd()

    report = pd.read_csv(ROOT / cfg.exp_name / "report.csv", index_col=0)
    report = report.to_dict()

    # 计算最长类名
    W = 0
    for key in report.keys():
        length = display_len(key)
        W = max(W, length)

    print(
        f"\n{'':>{W}}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}\n\n"
    )

    D = cfg.digits

    for key, value in report.items():
        class_name = format_string(key, W)

        if key == "accuracy":
            f1_score = value["f1-score"]  # pandas解析问题
            print(f"\n{class_name}{'':>12}{'':>12}{f1_score:>12.{D}f}{'':>12}")
        else:
            precision = value["precision"]
            recall = value["recall"]
            f1_score = value["f1-score"]
            support = int(value["support"])
            print(
                f"{class_name}{precision:>12.{D}f}{recall:>12.{D}f}{f1_score:12.{D}f}{support:>12}"
            )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--exp-name",
        "--exp",
        type=str,
        required=True,
        help="path of the experimental directory (relative e.g. YYYYmmdd-HHMMSS)",
    )
    parser.add_argument(
        "--digits", default=3, type=int, help="digits remain for accuracy"
    )

    main(parser.parse_args())
