from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def get_report(exp_name: str) -> pd.DataFrame:
    report = pd.read_csv(Path(exp_name).absolute() / "report.csv", index_col=0)
    if len(report.columns.to_list()) != 4:
        report = report.T

    return report


def compare_accuracy_old(cfg) -> None:
    from _utils import display_len, format_string

    # 计算最长别名
    N = 12
    if cfg.aliases is not None:
        assert len(cfg.exp_list) == len(cfg.aliases), "别名必须与实验对应"
        for alias in cfg.aliases:
            length = display_len(alias)
            N = max(N, length)
        N = N + 2 if N >= 12 else N

    # 计算最长类名
    W = 0
    report = get_report(cfg.exp_list[0])
    for name in report.index:
        length = display_len(name)
        W = max(W, length)

    header = f"\n{'':>{W}}"
    classes = []
    f1_score_list = []
    for i, exp_name in enumerate(cfg.exp_list):
        if cfg.aliases is None:
            name = exp_name.split("-")[-1]
        else:
            name = cfg.aliases[i]

        header += format_string(name, N)

        report = get_report(exp_name)

        indices = report.index.to_list()
        if classes == []:
            classes = indices
        else:
            assert len(classes) == len(indices), "实验类名必须相同！"

        f1_score = report["f1-score"].to_list()

        f1_score_list.append(f1_score)
    header += "\n"

    console = Console()
    console.print(header, highlight=False)

    D = cfg.digits

    for i in range(len(classes)):
        class_name = format_string(classes[i], W)

        if classes[i] == "accuracy":
            class_name = "\n" + class_name

        # find the maximum f1 score
        f1_scores = []
        for j in range(len(cfg.exp_list)):
            f1_score = float(f1_score_list[j][i])
            f1_scores.append(round(f1_score, D))
        max_score = max(f1_scores)

        string = f"{class_name}"
        for f1_score in f1_scores:
            if f1_score == max_score:
                string += f"[red]{f1_score:>{N}.{D}f}[/red]"
            else:
                string += f"{f1_score:>{N}.{D}f}"
        console.print(string, highlight=False)


def compare_accuracy(cfg) -> None:
    if cfg.aliases is None:
        N = 6  # len(HHMMSS)
    else:
        assert len(cfg.exp_list) == len(cfg.aliases), "别名必须与实验对应"
        N = max([len(a) for a in cfg.aliases])

    D = cfg.digits
    N = max(N, 2 + D)

    table = Table(title="Compare Accuracy", show_footer=True)
    table.add_column("", "", justify="right", style="cyan")

    data = {}
    for i, exp_name in enumerate(cfg.exp_list):
        if cfg.aliases is None:
            name = Path(exp_name).name.split("-")[-1]
        else:
            name = cfg.aliases[i]
        table.add_column(name, name, justify="right", width=N)

        report = get_report(exp_name)

        if i == 0:
            for idx in report.index:
                data[idx] = []

        for idx in report.index:
            data[idx].append(report["f1-score"][idx])

    for i, (idx, f1_scores) in enumerate(data.items()):
        if cfg.interval != 0 and i <= len(data) - 3 and i != 0 and i % cfg.interval == 0:
            table.add_row()

        format_scores = []
        max_score = max(f1_scores)
        for f1_score in f1_scores:
            if f1_score == max_score:
                f1_score = round(f1_score, D)
                format_scores.append(f"[bold red]{f1_score:.{D}f}[/bold red]")
            else:
                f1_score = round(f1_score, D)
                format_scores.append(f"{f1_score:.{D}f}")

        if idx == "accuracy":
            table.add_row()

        table.add_row(idx, *format_scores)

    console = Console()
    console.print(table)


def main(cfg) -> None:
    if cfg.old:
        compare_accuracy_old(cfg)
    else:
        compare_accuracy(cfg)


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
    parser.add_argument("--interval", default=0, type=int, help="interval for adding an empty line")
    parser.add_argument("--old", action="store_true", help="whether to print in old format")

    main(parser.parse_args())
