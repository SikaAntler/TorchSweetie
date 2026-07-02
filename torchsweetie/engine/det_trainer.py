from pathlib import Path
from typing import override

import torch
from rich import print
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

from ..data import DetDataPack, create_det_dataloader
from .trainer import IterBasedTrainer


class DetTrainer(IterBasedTrainer):
    SCOPE = "detection"

    def __init__(self, cfg_file: Path, run_dir: Path) -> None:
        super().__init__(cfg_file, run_dir)

        # train
        self.num_iters: int = self.cfg.train.num_iters

        # val
        self.val_interval = self.cfg.val.interval

        self.train_iter = self.infinite_dataloader(self.train_dataloader)

        self.iter: int = 0

        self.losses: list[str] = []

        if self.accelerator.is_main_process:
            self.metric = MeanAveragePrecision(
                max_detection_thresholds=[1, 17, 300],
                class_metrics=True,
                backend="faster_coco_eval",
            )

    @override
    def build_train_dataloader(self) -> DataLoader:
        return create_det_dataloader(self.cfg.train_dataloader)

    @override
    def build_val_dataloader(self) -> DataLoader | None:
        if self.cfg.get("val_dataloader") is None:
            val_dataloader = None
        else:
            val_dataloader = create_det_dataloader(self.cfg.val_dataloader)
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

        self.losses.clear()

        data: DetDataPack = next(self.train_iter)
        data.images = data.images.to(self.device)
        data.labels = data.labels.to(self.device)

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(data)
                loss_dict: dict[str, Tensor] = self.loss_fn(outputs, data)

            if isinstance(loss_dict, Tensor):
                loss = loss_dict
                loss_item = loss_dict.detach().cpu().item()
                self.losses.append(f"loss={loss_item:.4f}")
            else:
                loss = sum(v for v in loss_dict.values())
                for key, value in loss_dict.items():
                    loss_item = value.detach().cpu().item()
                    self.losses.append(f"{key}={loss_item:.4f}")

            self.accelerator.backward(loss * self.accelerator.gradient_accumulation_steps)

            if self.accelerator.sync_gradients and self.clip_grad:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_norm, self.norm_type
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients and self.lr_scheduler:
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients and self.ema:
                self.ema.update(self.accelerator.unwrap_model(self.model))

        self.progress.update(self.task, advance=1, losses=" ".join(self.losses))

    @override
    def after_iter(self) -> None:
        if (self.iter + 1) % self.val_interval == 0:
            self._val()

        super().after_iter()

    @override
    def after_train(self) -> None:
        self.progress.stop()

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
            transient=True,
            disable=not self.accelerator.is_main_process,
        ) as progress:
            task = progress.add_task("Val", total=len(self.val_dataloader))

            for data in self.val_dataloader:
                data: DetDataPack
                data.images = data.images.to(self.device)
                data.labels = data.labels.to(self.device)

                with self.accelerator.autocast():
                    predictions = model(data)

                preds: list[dict[str, Tensor]] = []
                target: list[dict[str, Tensor]] = []

                for img_id, (boxes, scores, cls_idxs) in enumerate(predictions):
                    preds.append(
                        {
                            "boxes": boxes,
                            "scores": scores,
                            "labels": cls_idxs,
                        }
                    )

                    mask = data.labels[:, 0] == img_id
                    gt = data.labels[mask]  # (M, 6), where 6 = [img_id, cls_idx, cx, cy, w, h]

                    # cxcywh (normalized) -> xyxy (absolute pixels)
                    img_h, img_w = data.images.shape[-2:]
                    half_w = gt[:, 4] * img_w / 2
                    half_h = gt[:, 5] * img_h / 2
                    cx = gt[:, 2] * img_w
                    cy = gt[:, 3] * img_h
                    gt_boxes = torch.stack(
                        [
                            cx - half_w,
                            cy - half_h,
                            cx + half_w,
                            cy + half_h,
                        ],
                        dim=1,
                    )

                    target.append({"boxes": gt_boxes, "labels": gt[:, 1].long()})

                self.accelerator.wait_for_everyone()

                all_preds = self.accelerator.gather_for_metrics(preds)
                all_target = self.accelerator.gather_for_metrics(target)

                if self.accelerator.is_main_process:
                    self.metric.update(all_preds, all_target)

                progress.update(task, advance=1)

        if self.accelerator.is_main_process:
            result = self.metric.compute()  # ty: ignore
            print(result)

    @staticmethod
    def infinite_dataloader(dataloader: DataLoader):
        while True:
            for batch in dataloader:
                yield batch
