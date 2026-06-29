from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from cleanlab.rank import get_label_quality_scores

from torchsweetie.data import ClsDataset
from torchsweetie.engine import ClsKFoldCrossValidator


def main(cfg) -> None:
    all_dataset = []
    classes = []
    all_labels = []
    all_pred_probs = []

    for _, exp_name in enumerate(cfg.exp_list):
        exp_dir = Path(exp_name).absolute()
        run_dir = exp_dir.parent
        cfg_file = run_dir / "config.yaml"

        if cfg.best:
            weights = "best-*[0-9].pth"
        elif cfg.last:
            weights = "last-*[0-9].pth"
        else:
            weights = f"epoch-{cfg.epoch}.pth"

        weights = list(exp_dir.glob(weights))
        assert len(weights) == 1
        weights = weights[0].name

        validator = ClsKFoldCrossValidator(cfg_file, exp_dir, weights)

        assert validator.dataloader
        assert isinstance(validator.dataloader.dataset, ClsDataset)
        dataset = validator.dataloader.dataset.dataset
        all_dataset.extend(dataset)
        classes = validator.dataloader.dataset.classes

        all_labels.append(np.array([classes.index(n) for _, n in dataset]))

        validator.run()
        all_pred_probs.append(validator.softmax().numpy())

    labels = np.hstack(all_labels)
    pred_probs = np.vstack(all_pred_probs)

    quality_scores = get_label_quality_scores(labels, pred_probs, method="normalized_margin")
    num = int(cfg.low_rate / 100 * len(quality_scores))
    low_quality_indices = np.argsort(quality_scores)[:num]

    save_file = Path(cfg.save_file).absolute()
    with open(save_file, "w", encoding="utf-8") as fw:
        for idx in low_quality_indices:
            pred = classes[np.argmax(pred_probs[idx])]
            fw.write(f"{all_dataset[idx][0]},{pred},{quality_scores[idx]}\n")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "exp_list",
        nargs="+",
        type=str,
        help="list of some experimental directories (e.g. path/to/exp_name/YYYYmmdd-HHMMSS)",
    )

    group_weights = parser.add_mutually_exclusive_group(required=True)
    group_weights.add_argument(
        "--best", action="store_true", help="whether to load the best weights"
    )
    group_weights.add_argument(
        "--last", action="store_true", help="whether to load the last weights"
    )
    group_weights.add_argument("--epoch", type=int, help="which epoch of weights want to load")

    parser.add_argument(
        "--low-rate", "--low", type=int, required=True, help="percentage of low quality data"
    )

    parser.add_argument(
        "--save-file",
        "--save",
        type=str,
        required=True,
        help="path to save the low quality data in csv format",
    )

    main(parser.parse_args())
