import math
from pathlib import Path
from typing import Generator, Literal, override

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from ..data import DetDataPack, DetResult, convert_to_preds_and_target, create_det_dataloader
from ..utils import URL_B, URL_E, BoxFormat
from .trainer import IterBasedTrainer


class DetTrainer(IterBasedTrainer):
    SCOPE = "detection"

    def __init__(self, cfg_file: Path, run_dir: Path) -> None:
        super().__init__(cfg_file, run_dir)

        # train
        self.num_iters: int = self.cfg.train.num_iters

        self.train_iter = self.infinite_dataloader(self.train_dataloader)

        self.iter: int = 0

        # val
        self.val_interval = self.cfg.val.interval
        self.save_best = self.cfg.val.get("save_best", False)
        self.save_last = self.cfg.val.get("save_last", True)
        self.iou_threshold = self.cfg.val.get("iou_threshold", 0.6)
        self.max_detection = self.cfg.val.get("max_detection", 300)

        # val.metric
        if "metric" not in self.cfg.val:
            self.cfg.val.metric = {}
        self.metric = MeanAveragePrecision(
            iou_thresholds=self.cfg.val.metric.get("iou_thresholds"),
            max_detection_thresholds=[1, round(math.sqrt(self.max_detection)), self.max_detection],
            backend="faster_coco_eval",
        )
        self.metric.to(self.device)

        if self.accelerator.is_main_process:
            self.results: list[tuple[int, float, float, float]] = []
            self.best_map = 0.0

    @override
    def build_train_dataloader(self) -> DataLoader:
        dataloader_cfg = self.cfg.train_dataloader
        if "box_format" not in dataloader_cfg.dataset:
            dataloader_cfg.dataset.box_format = BoxFormat.cxcywh
        return create_det_dataloader(dataloader_cfg)

    @override
    def build_val_dataloader(self) -> DataLoader | None:
        if self.cfg.get("val_dataloader") is None:
            val_dataloader = None
        else:
            dataloader_cfg = self.cfg.val_dataloader
            if "box_format" not in dataloader_cfg.dataset:
                dataloader_cfg.dataset.box_format = BoxFormat.cxcywh
            val_dataloader = create_det_dataloader(dataloader_cfg)
            val_dataloader = self.accelerator.prepare(val_dataloader)

        return val_dataloader

    @override
    def before_train(self) -> None:
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[losses]}"),
            console=self.console if self.accelerator.is_main_process else None,
            transient=True,
            disable=not self.accelerator.is_main_process,
        )
        self.progress.start()

        self.task = self.progress.add_task("Train", total=self.num_iters, losses="")

        super().before_train()

    @override
    def run_iter(self) -> None:
        self.model.train()
        self.loss_fn.train()

        data: DetDataPack = next(self.train_iter)
        data.img_idxs = data.img_idxs.to(self.device)
        data.images = data.images.to(self.device)
        data.cls_idxs = data.cls_idxs.to(self.device)
        data.boxes = data.boxes.to(self.device)

        losses: list[str] = []

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(data)
                loss_dict: Tensor | dict[str, Tensor] = self.loss_fn(outputs, data)

            if isinstance(loss_dict, Tensor):
                loss = loss_dict
                loss_item = loss_dict.detach().cpu().item()
                losses.append(f"loss={loss_item:.4f}")
            else:
                loss = sum(loss_dict.values())
                for key, value in loss_dict.items():
                    loss_item = value.detach().cpu().item()
                    losses.append(f"{key}={loss_item:.4f}")

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.clip_grad:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_norm, self.norm_type
                )

            self.optimizer.step()

            if self.accelerator.sync_gradients:
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                if self.momentum_scheduler:
                    self.momentum_scheduler.step()
                if self.ema:
                    self.ema.update(self.accelerator.unwrap_model(self.model))

            self.optimizer.zero_grad()

        self.progress.update(self.task, advance=1, losses=" ".join(losses))

    @override
    def after_iter(self) -> None:
        if (self.iter + 1) % self.val_interval == 0:
            self._val()

        super().after_iter()

    @override
    def after_train(self) -> None:
        self.progress.stop()

        if self.accelerator.is_main_process and self.save_last:
            self._save("last")

        super().after_train()

    @torch.inference_mode()
    def _val(self) -> None:
        if self.val_dataloader is None:
            return

        model = self.ema.ema if self.ema else self.accelerator.unwrap_model(self.model)
        model.eval()

        self.metric.reset()

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console if self.accelerator.is_main_process else None,
            transient=True,
            disable=not self.accelerator.is_main_process,
        ) as progress:
            task = progress.add_task("Val", total=len(self.val_dataloader))

            for data in self.val_dataloader:
                data: DetDataPack
                data.img_idxs = data.img_idxs.to(self.device)
                data.images = data.images.to(self.device)
                data.cls_idxs = data.cls_idxs.to(self.device)
                data.boxes = data.boxes.to(self.device)

                with self.accelerator.autocast():
                    predictions: list[DetResult] = model(data)

                preds, target = convert_to_preds_and_target(
                    data,
                    predictions,
                    self.iou_threshold,
                    self.max_detection,
                    self.cfg.val_dataloader.dataset.box_format,
                )

                self.metric.update(preds, target)  # ty: ignore

                progress.update(task, advance=1)

        result = self.metric.compute()  # ty: ignore
        self.print(
            f"Iteration {self.iter}: mAP = {result['map']:.4f}"
            f" | mAP50 = {result['map_50']:.4f}"
            f" | mAP75 = {result['map_75']:.4f}"
        )

        if self.accelerator.is_main_process:
            self.results.append(
                (self.iter, result["map"].item(), result["map_50"].item(), result["map_75"].item())
            )
            self._record()
            current_map = result["map"].item()
            if current_map >= self.best_map:
                self.best_map = current_map
                if self.save_best:
                    self._save("best")

    @staticmethod
    def infinite_dataloader(dataloader: DataLoader) -> Generator:
        while True:
            for batch in dataloader:
                yield batch

    def _record(self) -> None:
        with open(self.exp_dir / "record.csv", "w", encoding="utf-8") as fw:
            fw.write("Iteration,mAP,mAP50,mAP75\n")
            for result in self.results:
                record = ",".join([str(v) for v in result])
                fw.write(f"{record}\n")

    def _save(self, prefix: Literal["last", "best"]) -> None:
        if self.ema:
            model = self.ema.ema
        else:
            model = self.accelerator.unwrap_model(self.model)

        model_file = self.exp_dir / f"{prefix}-{self.iter}.pth"
        torch.save(model.state_dict(), model_file)
        self.print(f"Saved the {prefix} model: {URL_B}{model_file}{URL_E}")
