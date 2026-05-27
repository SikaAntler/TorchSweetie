from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def get_report(filename: Path) -> pd.DataFrame:
    report = pd.read_csv(filename, index_col=0)
    if len(report.columns.to_list()) != 4:
        report = report.T

    return report


def print_report_old(filename: Path, digits: int) -> None:
    from .string_utils import display_len, format_string

    # 计算最长类名
    W = 0
    report = get_report(filename)
    for idx in report.index:
        length = display_len(idx)
        W = max(W, length)

    N = 12

    print(f"\n{'':>{W}}{'precision':>{N}}{'recall':>{N}}{'f1-score':>{N}}{'support':>{N}}\n\n")

    D = digits

    for idx, precision, recall, f1_score, support in report.itertuples():
        class_name = format_string(idx, W)

        f1_score = round(f1_score, D)
        precision = round(precision, D)
        recall = round(recall, D)
        support = int(support)

        if idx == "accuracy":
            print(f"\n{class_name}{'':>{N}}{'':>{N}}{f1_score:>{N}.{D}f}{'':>{N}}")
        else:
            print(
                f"{class_name}{precision:>{N}.{D}f}{recall:>{N}.{D}f}{f1_score:{N}.{D}f}{support:>{N}}"
            )


def print_report(filename: Path, digits: int = 3, interval: int = 0) -> None:
    report = get_report(filename)

    N = 9  # len(precision)

    table = Table(title="Classification Report")
    table.add_column("", justify="right", style="cyan")
    table.add_column("precision", justify="right", width=N)
    table.add_column("recall", justify="right", width=N)
    table.add_column("f1-score", justify="right", width=N)
    table.add_column("support", justify="right", width=N)

    D = digits

    for i, (idx, precision, recall, f1_score, support) in enumerate(report.itertuples()):
        if interval != 0 and i <= len(report) - 3 and i != 0 and i % interval == 0:
            table.add_row()

        f1_score = round(f1_score, D)
        precision = round(precision, D)
        recall = round(recall, D)
        support = int(support)

        if idx == "accuracy":
            precision = ""
            recall = ""
            f1_score = f"[bold red]{f1_score:.{D}f}[/bold red]"
            support = ""
            table.add_row()
        else:
            precision = f"{precision:.{D}f}"
            recall = f"{recall:.{D}f}"
            f1_score = f"{f1_score:.{D}f}"
            support = str(int(support))

        table.add_row(idx, precision, recall, f1_score, support)

    console = Console()
    console.print(table)
