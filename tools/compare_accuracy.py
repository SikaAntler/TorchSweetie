from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from _utils import display_len, format_string
from rich.console import Console


def main(cfg) -> None:
    # 计算最长别名
    N = 12
    if cfg.aliases is not None:
        assert len(cfg.exp_list) == len(cfg.aliases), "别名必须与实验对应"
        for alias in cfg.aliases:
            length = display_len(alias)
            N = max(N, length)
        N = N + 2 if N >= 12 else N

    ROOT = Path().cwd()

    console = Console()

    # 计算最长类名
    report = pd.read_csv(ROOT / cfg.exp_list[0] / "report.csv", index_col=0)
    report = report.to_dict()
    W = 0
    for key in report.keys():
        length = display_len(key)  # pyright: ignore
        W = max(W, length)

    header = f"\n{'':>{W}}"
    classes = []
    f1_score_list = []
    for i, exp_dir in enumerate(cfg.exp_list):
        exp_dir = ROOT / exp_dir

        if cfg.aliases is None:
            exp_name = exp_dir.name.split("-")[-1]
        else:
            exp_name = cfg.aliases[i]

        header += format_string(exp_name, N)

        report = pd.read_csv(exp_dir / "report.csv", index_col=0).T

        indices = report.index.to_list()
        if classes == []:
            classes = indices
        else:
            assert len(classes) == len(indices), "实验类名必须相同！"

        f1_score = report["f1-score"].to_list()

        f1_score_list.append(f1_score)
    header += "\n"
    console.print(header, highlight=False)

    D = cfg.digits

    for i in range(1, len(classes)):
        class_name = format_string(classes[i], W)

        if classes[i] == "accuracy":
            class_name = "\n" + class_name

        # find the maximum f1 score
        f1_scores = []
        for j in range(len(cfg.exp_list)):
            f1_score = float(f1_score_list[j][i])
            f1_scores.append(round(f1_score, D))
        max_f1_score = max(f1_scores)

        string = f"{class_name}"
        for f1_score in f1_scores:
            if f1_score == max_f1_score:
                string += f"[red]{f1_score:>{N}.{D}f}[/red]"
            else:
                string += f"{f1_score:>{N}.{D}f}"
        console.print(string, highlight=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "exp_list",
        nargs="+",
        type=str,
        help="list of some experimental directories (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )
    parser.add_argument(
        "--aliases",
        nargs="+",
        type=str,
        help="list of some aliases for experimental directories",
    )
    parser.add_argument("--digits", default=3, type=int, help="digits remain for accuracy")

    main(parser.parse_args())
