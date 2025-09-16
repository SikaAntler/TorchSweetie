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


def main(cfg) -> None:
    base_report = get_report(cfg.baseline)
    support = base_report["support"].to_list()
    base_f1_score = base_report["f1-score"].to_dict()

    exp_report = get_report(cfg.exp_name)
    exp_f1_score = exp_report["f1-score"].to_dict()

    N = 8

    table = Table(title="F1-Score Variation")
    table.add_column("", justify="right", style="cyan")
    table.add_column("baseline", justify="right", width=N)
    table.add_column("exp", justify="right", width=N)
    table.add_column("var(%)", justify="right", width=N)
    table.add_column("support", justify="right", width=N)

    DA = cfg.digits_acc
    DV = cfg.digits_var

    data = []
    num_var = {"Improve": 0, "Stable": 0, "Decline": 0}
    avg_imp, avg_dec = 0, 0
    for (idx, base_score), num in zip(base_f1_score.items(), support):
        exp_score = exp_f1_score[idx]
        variation = (exp_score / (base_score + 1e-6) - 1) * 100

        if idx not in ["accuracy", "macro avg", "weighted avg"]:
            if variation > 0:
                num_var["Improve"] += 1
                avg_imp += variation
            elif variation == 0:
                num_var["Stable"] += 1
            elif variation < 0:
                num_var["Decline"] += 1
                avg_dec += variation

        base_score = round(base_score, DA)
        exp_score = round(exp_score, DA)
        variation = round(variation, DV)
        data.append((idx, base_score, exp_score, variation, num))
    avg_imp /= num_var["Improve"] + 1e-6
    avg_dec /= num_var["Decline"] + 1e-6

    # sort by variation
    sorted_data = sorted(data, key=lambda d: d[3])
    sorted_keys = [d[0] for d in sorted_data]

    for i, (idx, base_score, exp_score, variation, num) in enumerate(data):
        if cfg.interval != 0 and i <= len(data) - 3 and i != 0 and i % cfg.interval == 0:
            table.add_row()

        if idx in sorted_keys[: cfg.topn]:
            variation = f"[bold red]{variation:.{DV}f}[/bold red]"
        elif idx in sorted_keys[-cfg.topn :]:
            variation = f"[bold blue]{variation:.{DV}f}[/bold blue]"
        else:
            variation = f"{variation:.2f}"

        if idx == "accuracy":
            table.add_row()
            num = ""
        else:
            num = str(int(num))

        table.add_row(idx, f"{base_score:.{DA}f}", f"{exp_score:.{DA}f}", variation, num)

    console = Console()
    console.print(table)

    for idx, base_score in num_var.items():
        console.print(f"{idx:>7}: {base_score:>2} classes")
    console.print(f"Avg Imp: {avg_imp:.2f}%")
    console.print(f"Avg Dec: {avg_dec:.2f}%")


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
    parser.add_argument("--interval", default=0, type=int, help="interval for adding an empty line")

    main(parser.parse_args())
