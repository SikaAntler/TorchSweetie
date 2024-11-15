from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rich import print


def is_chinese(c: str) -> bool:
    assert len(c) == 1

    if "\u4e00" <= c <= "\u9fa5":
        return True
    else:
        return False


def main(cfg) -> None:
    ROOT = Path().cwd()

    report = pd.read_csv(ROOT / cfg.exp_name / "report.csv", index_col=0)
    report = report.to_dict()

    # 计算最长类名
    W = 0
    for key in report.keys():
        length = 0
        for c in key:
            if is_chinese(c):
                length += 2
            else:
                length += 1
        W = max(W, length)

    print(
        f"\n{'':>{W}}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}\n\n"
    )

    for key, value in report.items():
        length = 0
        for c in key:
            if is_chinese(c):
                length += 2
            else:
                length += 1

        format_key = key
        while length < W:
            format_key = " " + format_key
            length += 1

        if key == "accuracy":
            f1_score = value["f1-score"]  # pandas解析问题
            print(f"\n{format_key}{'':>12}{'':>12}{f1_score:>12.3f}{'':>12}")
        else:
            precision = value["precision"]
            recall = value["recall"]
            f1_score = value["f1-score"]
            support = value["support"]
            print(
                f"{format_key}{precision:>12.3f}{recall:>12.3f}{f1_score:12.3f}{support:>12}"
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

    main(parser.parse_args())
