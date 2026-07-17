from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from rich import print
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision

from torchsweetie.utils import URL_B, URL_E, cxcywh2xyxy, denormalize


def convert_to(label_file: Path, img_w: int, img_h: int) -> dict[str, Tensor]:
    with open(label_file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    boxes, scores, labels = [], [], []

    for line in lines:
        items = line.rstrip().split(" ")
        assert len(items) in [5, 6]

        idx = int(items[0])
        cx = float(items[1])
        cy = float(items[2])
        w = float(items[3])
        h = float(items[4])
        boxes.append(torch.tensor([cx, cy, w, h]))
        if len(items) == 6:
            scores.append(float(items[5]))
        labels.append(idx)

    boxes = cxcywh2xyxy(torch.vstack(boxes))
    denormalize(boxes, img_w, img_h)

    data = {"boxes": boxes, "labels": torch.LongTensor(labels)}
    if len(scores) != 0:
        data["scores"] = torch.tensor(scores)

    return data


def main(cfg) -> None:
    data_dir = Path(cfg.data_dir)
    val_dir = Path(cfg.val_dir)
    run_dir = Path(cfg.run_dir)

    exp_dir = run_dir / val_dir.name
    if not exp_dir.exists():
        exp_dir.mkdir()

    target_labels_dir = data_dir / "labels/val/"
    preds_labels_dir = Path(cfg.val_dir) / "labels/"

    target_label_files = list(f.name for f in target_labels_dir.iterdir())
    preds_filenames = list(f.name for f in preds_labels_dir.iterdir())
    print(len(target_label_files), len(preds_filenames))

    target = []
    preds = []

    metric = MeanAveragePrecision(
        max_detection_thresholds=[1, 17, 300], class_metrics=True, backend="faster_coco_eval"
    )

    for target_name in target_label_files:
        target_ = convert_to(target_labels_dir / target_name, cfg.img_w, cfg.img_h)
        target.append(target_)

        if target_name in preds_filenames:
            pred = convert_to(preds_labels_dir / target_name, cfg.img_w, cfg.img_h)
            preds.append(pred)
        else:
            preds.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0, 4)),
                    "labels": torch.zeros((0,)),
                }
            )

    metric.update(preds, target)  # ty: ignore
    result = metric.compute()  # ty: ignore

    classes_file = data_dir / "classes.csv"
    classes = pd.read_csv(classes_file, header=None)[0].to_list()

    report_file = exp_dir / "report.csv"

    with open(report_file, "w", encoding="utf-8") as fw:
        for name, value in zip(classes, result["map_per_class"]):
            value = value.item()
            fw.write(f"{name},{value}\n")
        fw.write(f"mAP,{result['map'].item()}\n")
        fw.write(f"mAP50,{result['map_50'].item()}\n")
        fw.write(f"mAP50,{result['map_75'].item()}\n")

    print(f"Saved the report: {URL_B}{report_file}{URL_E}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str)

    parser.add_argument("--val-dir", type=str, required=True)

    parser.add_argument("--img-w", default=1280, type=int)
    parser.add_argument("--img-h", default=1280, type=int)

    parser.add_argument(
        "--run-dir",
        "--run",
        default="runs",
        type=str,
        help="path of the running directory (relative)",
    )

    main(parser.parse_args())
