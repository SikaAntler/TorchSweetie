from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from _utils import display_len, format_string
from rich import print


def main(cfg) -> None:
    ROOT = Path().cwd()

    report = pd.read_csv(ROOT / cfg.exp_name / "report.csv", index_col=0)
    report = report.to_dict()

    N = 12

    # 计算最长类名
    W = 0
    for key in report.keys():
        length = display_len(key)  # pyright: ignore
        W = max(W, length)

    print(f"\n{'':>{W}}{'precision':>{N}}{'recall':>{N}}{'f1-score':>{N}}{'support':>{N}}\n\n")

    D = cfg.digits

    for key, value in report.items():
        class_name = format_string(key, W)  # pyright: ignore

        if key == "accuracy":
            f1_score = round(value["f1-score"], D)  # pandas解析问题
            print(f"\n{class_name}{'':>{N}}{'':>{N}}{f1_score:>{N}.{D}f}{'':>{N}}")
        else:
            precision = round(value["precision"], D)
            recall = round(value["recall"], D)
            f1_score = round(value["f1-score"], D)
            support = int(value["support"])
            print(
                f"{class_name}{precision:>{N}.{D}f}{recall:>{N}.{D}f}{f1_score:{N}.{D}f}{support:>{N}}"
            )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "exp_name",
        type=str,
        help="path of the experimental directory (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )
    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")

    main(parser.parse_args())
