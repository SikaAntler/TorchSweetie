from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rich.console import Console


def is_chinese(c: str) -> bool:
    assert len(c) == 1

    if "\u4e00" <= c <= "\u9fa5":
        return True
    else:
        return False


def main(cfg) -> None:
    ROOT = Path().cwd()

    console = Console()

    # 计算最长类名
    report = pd.read_csv(ROOT / cfg.exp_list[0] / "report.csv", index_col=0)
    report = report.to_dict()
    W = 0
    for key in report.keys():
        length = 0
        for c in key:
            if is_chinese(c):
                length += 2
            else:
                length += 1
        W = max(W, length)

    header = f"\n{'':>{W}}"
    classes = []
    f1_score_list = []
    for i, exp_dir in enumerate(cfg.exp_list):
        exp_dir = ROOT / exp_dir

        exp_name = exp_dir.name.split("-")[-1]
        header += f"{exp_name:>12}"

        report = pd.read_csv(exp_dir / "report.csv", index_col=0).T

        indices = report.index.to_list()
        if classes == []:
            classes = indices
        else:
            assert len(classes) == len(indices)

        f1_score = report["f1-score"].to_list()

        f1_score_list.append(f1_score)
    header += "\n"
    console.print(header, highlight=False)

    for i in range(1, len(classes)):
        length = 0
        for c in classes[i]:
            if is_chinese(c):
                length += 2
            else:
                length += 1

        format_cls = classes[i]
        while length < W:
            format_cls = " " + format_cls
            length += 1

        if classes[i] == "accuracy":
            format_cls = "\n" + format_cls

        # find the maximum f1 score
        f1_scores = []
        for j in range(len(cfg.exp_list)):
            f1_scores.append(float(f1_score_list[j][i]))
        max_f1_score = max(f1_scores)

        string = f"{format_cls}"
        for f1_score in f1_scores:
            if f1_score == max_f1_score:
                string += f"[red]{f1_score:>12.3f}[/red]"
            else:
                string += f"{f1_score:>12.3f}"
        console.print(string, highlight=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--exp-list",
        "--exp",
        nargs="+",
        type=str,
        required=True,
        help="list of some experimental directories (relative e.g. YYYYmmdd-HHMMSS)",
    )

    main(parser.parse_args())
