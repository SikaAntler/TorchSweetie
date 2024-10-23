from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from rich.console import Console


def main(cfg) -> None:
    ROOT = Path(cfg.root_dir)

    console = Console()

    header = f"\n{'':>12}"
    classes = []
    f1_scores_list = []
    for i, exp_dir in enumerate(cfg.exp_list):
        exp_dir = ROOT / exp_dir
        report_file = exp_dir / "report.csv"
        report = pd.read_csv(report_file, header=None)

        sub_name = exp_dir.name.split("-")[-1]
        header += f"{sub_name:>12}"

        if i == 0:
            classes = report.iloc[
                0
            ].tolist()  # NaN, ..., accuracy, macro avg, weighted avg
        else:
            assert classes == report.iloc[0].tolist()
        f1_scores = report.iloc[3].tolist()
        f1_scores_list.append(f1_scores)
    header += "\n"
    console.print(header, highlight=False)

    for i in range(1, len(classes)):
        length = 0
        for c in classes[i]:
            if "\u4e00" <= c <= "\u9fa5":
                length += 2
            else:
                length += 1

        format_cls = classes[i]
        while length < 12:
            format_cls = " " + format_cls
            length += 1

        if classes[i] == "accuracy":
            format_cls = "\n" + format_cls

        # find the maximum f1 score
        f1_scores = []
        for j in range(len(cfg.exp_list)):
            f1_scores.append(float(f1_scores_list[j][i]))
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
        "--root-dir",
        "--root",
        type=str,
        required=True,
        help="path of the root directory",
    )
    parser.add_argument(
        "--exp-list",
        "--exp",
        nargs="+",
        type=str,
        required=True,
        help="list of some experiment directories (relative e.g. YYYYmmdd-HHMMSS)",
    )

    main(parser.parse_args())
