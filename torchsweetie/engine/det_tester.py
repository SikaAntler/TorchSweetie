import math
from pathlib import Path
from typing import override

import pandas as pd
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from ..data import DetDataPack, DetResult, convert_to_preds_and_target, create_det_dataloader
from ..utils import MODELS, URL_B, URL_E, BoxFormat, load_weights_for_model, print_det_report
from .runner import RunnerBase


class DetTester(RunnerBase):
    SCOPE = "detection"

    def __init__(
        self,
        cfg_file: Path,
        exp_dir: Path,
        weights: str,
        iou_threshold: float = 0.6,
        max_detection: int = 300,
    ) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        if self.cfg.train.get("mixed_precision", "no") == "fp16":
            self.model.half()
        self.model.cuda()

        self.iou_threshold = iou_threshold

        self.max_detection = max_detection

        self.metric = MeanAveragePrecision(
            max_detection_thresholds=[1, round(math.sqrt(max_detection)), max_detection],
            class_metrics=True,
            backend="faster_coco_eval",
        )
        self.metric.cuda()

    @override
    def build_model(self) -> nn.Module:
        if "scope" not in self.cfg.model:
            self.cfg.model.scope = self.SCOPE

        self.cfg.model.pop("_weights_", None)
        model = MODELS.create(self.cfg.model)
        load_weights_for_model(model, str(self.weights), True)

        return model

    @override
    def build_dataloader(self) -> DataLoader:
        dataloader_cfg = self.cfg.test_dataloader
        if "box_format" not in dataloader_cfg.dataset:
            dataloader_cfg.dataset.box_format = BoxFormat.cxcywh
        return create_det_dataloader(dataloader_cfg)

    @override
    @torch.inference_mode()
    def run(self) -> None:
        assert self.dataloader

        self.model.eval()

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Test", total=len(self.dataloader))

            for data in self.dataloader:
                data: DetDataPack
                data.img_idxs = data.img_idxs.cuda()
                data.images = data.images.cuda()
                data.cls_idxs = data.cls_idxs.cuda()
                data.boxes = data.boxes.cuda()

                with torch.autocast("cuda"):
                    predictions: list[DetResult] = self.model(data)

                preds, target = convert_to_preds_and_target(
                    data,
                    predictions,
                    self.iou_threshold,
                    self.max_detection,
                    self.cfg.test_dataloader.dataset.box_format,
                )

                self.metric.update(preds, target)  # ty: ignore

                progress.update(task, advance=1)

        self.result = self.metric.compute()  # ty: ignore

    def report(self, digits: int) -> None:
        classes_file = self.cfg.test_dataloader.dataset.classes_file
        classes = pd.read_csv(classes_file, header=None)[0].to_list()

        report_file = self.exp_dir / "report.csv"

        with open(report_file, "w", encoding="utf-8") as fw:
            for name, value in zip(classes, self.result["map_per_class"]):
                value = value.item()
                fw.write(f"{name},{value}\n")
            fw.write(f"mAP,{self.result['map'].item()}\n")
            fw.write(f"mAP50,{self.result['map_50'].item()}\n")
            fw.write(f"mAP50,{self.result['map_75'].item()}\n")

        self.console.print(f"Saved the report: {URL_B}{report_file}{URL_E}")

        print_det_report(report_file, digits)
