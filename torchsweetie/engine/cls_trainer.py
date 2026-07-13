from pathlib import Path
from typing import Literal, override

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

from ..data import ClsDataPack, create_cls_dataloader
from ..utils import KEY_B, KEY_E, URL_B, URL_E
from .trainer import EpochBasedTrainer


class ClsTrainer(EpochBasedTrainer):
    SCOPE = "classification"

    def __init__(self, cfg_file: Path, run_dir: Path) -> None:
        super().__init__(cfg_file, run_dir)

        assert self.accelerator.gradient_accumulation_steps == 1

        # val
        if self.cfg.get("val") is None:
            self.val_interval = 1
        else:
            self.val_interval = self.cfg.val.get("interval", 1)

        # Useful Parameters
        if self.accelerator.is_main_process:
            self.avg_loss: dict[str, float] = {}
            self.accuracy = 0.0
            self.best_acc = 0.0
            self.best_epoch = -1
            self.results: list[tuple[int, dict[str, float], float]] = []

        # Save
        if self.cfg.get("save") is None:
            self.save_interval = 999
            self.save_last = True
            self.save_best = False
        else:
            self.save_interval = self.cfg.save.get("interval", 999)
            self.save_last = self.cfg.save.get("last", True)
            self.save_best = self.cfg.save.get("best", False)

    @override
    def build_train_dataloader(self) -> DataLoader:
        return create_cls_dataloader(self.cfg.train_dataloader)

    @override
    def build_val_dataloader(self) -> DataLoader | None:
        if self.cfg.get("val_dataloader") is None:
            val_dataloader = None
        else:
            val_dataloader = create_cls_dataloader(self.cfg.val_dataloader)
            val_dataloader = self.accelerator.prepare(val_dataloader)

        return val_dataloader

    @override
    def run_epoch(self) -> None:
        self.model.train()
        self.loss_fn.train()

        total_loss: dict[str, list[float]] = {}

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[losses]}"),
            transient=True,
            disable=not self.accelerator.is_main_process,
        ) as progress:
            task = progress.add_task(
                f"Epoch {self.epoch}/{self.num_epochs - 1} train",
                total=len(self.train_dataloader),
                losses="",
            )

            for data in self.train_dataloader:
                data: ClsDataPack
                data.inputs = data.inputs.to(self.device)
                data.targets = data.targets.to(self.device)
                data.ori_sizes = data.ori_sizes.to(self.device)

                losses: list[str] = []

                with self.accelerator.autocast():
                    outputs = self.model(data)
                    loss_dict: Tensor | dict[str, Tensor] = self.loss_fn(outputs, data)

                if isinstance(loss_dict, Tensor):
                    loss = loss_dict
                    loss_item = loss_dict.detach().cpu().item()
                    losses.append(f"loss={loss_item:.4f}")
                    if "loss" not in total_loss:
                        total_loss["loss"] = []
                    total_loss["loss"].append(loss_item)
                else:
                    loss = sum(loss_dict.values())
                    for key, value in loss_dict.items():
                        loss_item = value.detach().cpu().item()
                        losses.append(f"{key}={loss_item:.4f}")
                        if key not in total_loss:
                            total_loss[key] = []
                        total_loss[key].append(loss_item)

                self.accelerator.backward(loss)

                if self.clip_grad:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_norm, self.norm_type
                    )

                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update(self.accelerator.unwrap_model(self.model))

                # TODO total_loss reduce

                self.optimizer.zero_grad()

                progress.update(task, advance=1, losses=" ".join(losses))

        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.momentum_scheduler:
            self.momentum_scheduler.step()

        if self.accelerator.is_main_process:
            self.avg_loss.clear()
            for key, value in total_loss.items():
                self.avg_loss[key] = sum(value) / len(value)

    @override
    def after_epoch(self) -> None:
        if self.accelerator.is_main_process:
            msg = f"Epoch {self.epoch}: "
            msg += " | ".join([f"(avg){k}={v:.3f}" for k, v in self.avg_loss.items()])

        if (self.epoch + 1) % self.val_interval == 0:
            self._val()
            if self.accelerator.is_main_process:
                msg += f" | accuracy={KEY_B}{self.accuracy:.3f}{KEY_E}"

        self.print(msg)

        if self.accelerator.is_main_process:
            self._record()

        super().after_epoch()

    @torch.inference_mode()
    def _val(self) -> None:
        if self.val_dataloader is None:
            return

        if self.ema:
            model = self.ema.ema
        else:
            model = self.model
            model.eval()
        self.loss_fn.eval()

        corrects = 0

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            disable=not self.accelerator.is_main_process,
        ) as progress:
            task = progress.add_task(f"Epoch {self.epoch} val", total=len(self.val_dataloader))

            for data in self.val_dataloader:
                data: ClsDataPack
                data.inputs = data.inputs.to(self.device)
                data.targets = data.targets.to(self.device)
                data.ori_sizes = data.ori_sizes.to(self.device)

                with self.accelerator.autocast():
                    outputs = model(data)
                    if self.loss_params:
                        outputs = self.loss_fn(outputs, data)

                predicts = torch.argmax(outputs, dim=1)
                metrics = self.accelerator.gather_for_metrics(predicts == data.targets)
                corrects += metrics.sum().cpu().item()

                progress.update(task, advance=1)

        if self.accelerator.is_main_process:
            self.accuracy = corrects / len(self.val_dataloader.dataset)  # ty: ignore

    def _record(self) -> None:
        self.results.append((self.epoch, self.avg_loss, self.accuracy))
        with open(self.exp_dir / "record.csv", "w", encoding="utf-8") as fw:
            loss_header = ",".join(self.avg_loss.keys())
            fw.write(f"epoch,{loss_header},accuracy\n")
            for epoch, avg_loss, accuracy in self.results:
                losses = ",".join([str(v) for v in avg_loss.values()])
                fw.write(f"{epoch},{losses},{accuracy}\n")

        # Save the interval(epoch)
        if (self.epoch + 1) % self.save_interval == 0:
            self._save("epoch")

        # Save the last epoch
        if self.save_last and (self.epoch == self.num_epochs - 1):
            self._save("last")

        # Save the best epoch
        if self.save_best and (self.accuracy >= self.best_acc):
            (self.exp_dir / f"best-{self.best_epoch}.pth").unlink(missing_ok=True)
            (self.exp_dir / f"best-{self.best_epoch}-loss.pth").unlink(missing_ok=True)
            self.best_acc = self.accuracy
            self.best_epoch = self.epoch
            self._save("best")

    def _save(self, prefix: Literal["epoch", "last", "best"]) -> None:
        # Model
        if self.ema:
            model = self.ema.ema
        else:
            model = self.accelerator.unwrap_model(self.model)
        model_file = self.exp_dir / f"{prefix}-{self.epoch}.pth"
        torch.save(model.state_dict(), model_file)
        self.print(f"Saved the {prefix} model: {URL_B}{model_file}{URL_E}")

        # Loss
        if not self.loss_params:
            return
        loss_fn = self.accelerator.unwrap_model(self.loss_fn)
        loss_file = self.exp_dir / f"{prefix}-{self.epoch}-loss.pth"
        torch.save(loss_fn.state_dict(), loss_file)
        self.print(f"Saved the {prefix} loss fn: {URL_B}{loss_file}{URL_E}")
