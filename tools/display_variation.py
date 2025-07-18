from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def main(cfg) -> None:
    ROOT = Path.cwd()
    console = Console()

    base_report = pd.read_csv(ROOT / cfg.baseline / "report.csv", index_col=0).T
    support = base_report["support"].to_list()
    base_report = base_report["f1-score"].to_dict()

    exp_report = pd.read_csv(ROOT / cfg.exp_name / "report.csv", index_col=0).T
    exp_report = exp_report["f1-score"].to_dict()

    table = Table(title="F1-Score Variation")
    table.add_column("Class", justify="right", style="cyan")
    table.add_column("Base", justify="right", width=8)
    table.add_column("Exp", justify="right", width=8)
    table.add_column("Var(%)", justify="right", width=8)
    table.add_column("Support", justify="right", width=8)

    DA = cfg.digits_acc
    DV = cfg.digits_var

    data = []
    for (key, value), num in zip(base_report.items(), support):
        exp_value = exp_report[key]
        variation = (exp_value / (value + 1e-6) - 1) * 100
        exp_value = round(exp_value, DA)
        variation = round(variation, DV)
        data.append((key, value, exp_value, variation, num))

    # sort by variation
    sorted_data = sorted(data, key=lambda d: d[3])
    sorted_keys = [d[0] for d in sorted_data]

    for key, value, exp_value, variation, num in data:
        if key in sorted_keys[: cfg.topn]:
            variation = f"[bold red]{variation:.{DV}f}[/bold red]"
        elif key in sorted_keys[-cfg.topn :]:
            variation = f"[bold blue]{variation:.{DV}f}[/bold blue]"
        else:
            variation = f"{variation:.2f}"

        if key == "accuracy":
            table.add_row("", "", "", "", "")
            num = ""
        else:
            num = str(int(num))

        table.add_row(key, f"{value:.{DA}f}", f"{exp_value:.{DA}f}", variation, num)

    console.print(table)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="the baseline experimental directory (e.g. path/to/baseline/YYYYmmdd-HHMMSS)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="the experimental dieectory (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )
    parser.add_argument("--digits-acc", default=3, type=int, help="digits remain for accuracy")
    parser.add_argument("--digits-var", default=2, type=int, help="digits remain for variation")
    parser.add_argument(
        "--topn",
        default=5,
        type=int,
        help="highlight variations for top n class of both increase and decrease",
    )

    main(parser.parse_args())
