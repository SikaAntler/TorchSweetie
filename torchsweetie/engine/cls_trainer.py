from pathlib import Path
from typing import Literal, override

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
from torch.utils.data import DataLoader

from ..data import ClsDataPack, create_cls_dataloader
from ..utils import KEY_B, KEY_E, URL_B, URL_E
from .engine import TrainerBase


class ClsTrainer(TrainerBase):
    SCOPE = "classification"

    def __init__(self, cfg_file: Path, run_dir: Path) -> None:
        super().__init__(cfg_file, run_dir, model_scope=self.SCOPE, loss_scope=self.SCOPE)

        # Useful Parameters
        if self.accelerator.is_main_process:
            self.avg_loss = 0.0
            self.accuracy = 0.0
            self.best_acc = 0.0
            self.best_epoch = -1
            self.results = []

        self.register_hooks([])

    @override
    def build_train_dataloader(self) -> DataLoader:
        return create_cls_dataloader(self.cfg.train_dataloader)

    @override
    def build_val_dataloader(self) -> DataLoader:
        return create_cls_dataloader(self.cfg.val_dataloader)

    @override
    def run_step(self) -> None:
        self.model.train()
        self.loss_fn.train()
        total_loss = 0.0

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            disable=not self.accelerator.is_main_process,
        ) as progress:
            task = progress.add_task(f"Epoch {self.epoch} train", total=len(self.train_dataloader))

            for data in self.train_dataloader:
                data: ClsDataPack
                data.inputs = data.inputs.to(self.device)
                data.targets = data.targets.to(self.device)
                data.ori_sizes = data.ori_sizes.to(self.device)

                with self.accelerator.autocast():
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, data)

                self.accelerator.backward(loss)

                if self.clip_grad is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_norm, self.norm_type
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()

                loss_reduce = self.accelerator.reduce(loss, "mean").cpu().item()
                total_loss += loss_reduce

                progress.update(task, advance=1)

        # TODO: if not accelerator.optimizer_step_was_skipped:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.avg_loss = total_loss / len(self.train_dataloader)

    @override
    def after_step(self) -> None:
        self._val()

        print(
            f"Epoch {self.epoch}: avg_loss={KEY_B}{self.avg_loss:.5f}{KEY_E} | accuracy={KEY_B}{self.accuracy:.3f}{KEY_E}"
        )

        if self.accelerator.is_main_process:
            self._record()

        super().after_step()

    @torch.inference_mode()
    def _val(self) -> None:
        if self.val_dataloader is None:
            return

        self.model.eval()
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

                outputs = self.model(data)
                if self.loss_params:
                    outputs = self.loss_fn(outputs, data)

                predicts = torch.argmax(outputs, dim=1)
                metrics = self.accelerator.gather_for_metrics(predicts == data.targets)
                corrects += metrics.sum().cpu().item()

                progress.update(task, advance=1)

        if self.accelerator.is_main_process:
            self.accuracy = corrects / len(self.val_dataloader.dataset)

    def _record(self) -> None:
        self.results.append((self.epoch, self.avg_loss, self.accuracy))
        with open(self.exp_dir / "record.csv", "w", encoding="utf-8") as fw:
            for epoch, avg_loss, accurary in self.results:
                fw.write(f"{epoch},{avg_loss},{accurary}\n")

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
        model = self.accelerator.unwrap_model(self.model)
        model_file = self.exp_dir / f"{prefix}-{self.epoch}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Saved the {prefix} model: {URL_B}{model_file}{URL_E}")

        # Loss
        if not self.loss_params:
            return
        loss_fn = self.accelerator.unwrap_model(self.loss_fn)
        loss_file = self.exp_dir / f"{prefix}-{self.epoch}-loss.pth"
        torch.save(loss_fn.state_dict(), loss_file)
        print(f"Saved the {prefix} loss fn: {URL_B}{loss_file}{URL_E}")
