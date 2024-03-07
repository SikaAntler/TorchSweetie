import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from accelerate import Accelerator
from rich import print
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import (
    LOSSES,
    LR_SCHEDULERS,
    MODELS,
    OPTIMIZERS,
    UTILS,
    get_config,
)


class ClsTrainer:
    NCOLS = 100

    def __init__(self, cfg_file: Union[Path, str]) -> None:
        # Get the root path (project path)
        ROOT = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / cfg_file
        self.cfg = get_config(self.cfg_file)

        # Accelerator
        split_batch = False
        # TODO: split_batch取值问题：
        # 如果是默认的False，在多卡时总batch_size会和配置文件不一致
        # 如果是True，有batch_sampler的情况下需要重设为False
        # 考虑到DDP模式下训练结果有别于单卡，batch_size可以不一致
        mixed_precision = self.cfg.train.get("mixed_precision")
        if mixed_precision is None:
            mixed_precision = "no"
        else:
            print(f"Using mixed precision: {mixed_precision}")
        self.accelerator = Accelerator(
            split_batches=split_batch, mixed_precision=mixed_precision
        )
        self.device = self.accelerator.device

        # Running directory, used to record results and models
        # Only executed by the main process
        if self.accelerator.is_main_process:
            self.run_dir = ROOT / "runs" / self.cfg_file.stem
            times = 0
            while self.run_dir.exists():
                times += 1
                self.run_dir = ROOT / "runs" / f"{self.cfg_file.stem}-{times}"
            self.run_dir.mkdir(parents=True)
            print(f"Running directory: {self.run_dir}:new:")

        # Model
        model = MODELS.create(self.cfg.model)

        # TODO: 需要freeze功能，初步设想是提供一个UTILS函数注册器
        if self.cfg.train.get("freeze"):
            UTILS.create(self.cfg.train.freeze, model)

        # Synchronize BN for DDP mode
        if self.cfg.model.get("sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Loss Function
        loss_fn = LOSSES.create(self.cfg.loss)

        # Optimizer & LR Scheduler
        optimizer = OPTIMIZERS.create(self.cfg.optimizer, model)
        if self.cfg.get("lr_scheduler"):
            lr_scheduler = LR_SCHEDULERS.create(self.cfg.lr_scheduler, optimizer)
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        # Train & Val DataLoaders
        dataloader_train = create_cls_dataloader(
            self.cfg.dataloader, "train", True, True
        )
        dataloader_val = create_cls_dataloader(self.cfg.dataloader, "val", False, False)

        # Prepare all
        (
            self.model,
            self.loss_fn,
            self.optimizer,
            self.dataloader_train,
            self.dataloader_val,
        ) = self.accelerator.prepare(
            model, loss_fn, optimizer, dataloader_train, dataloader_val
        )

        # Useful parameters
        self.num_epochs = self.cfg.train.num_epochs
        self.best_or_last = self.cfg.train.best_or_last
        # TODO: best_or_last应该有best、last和both三种
        if self.best_or_last == "last":
            self.record_last_n_epochs = 1
        else:
            self.record_last_n_epochs = self.cfg.record_last_n_epochs

        if self.accelerator.is_main_process:
            self.accuracy = None
            self.avg_loss = None
            self.best_acc = 0
            self.best_epoch = -1
            self.results = []
            # self.loss_iters = []

    def train(self) -> None:
        # self.progress = Progress(
        #     TextColumn("[progress.description]{task.description}"),
        #     "[{task.completed}/{task.total}]",
        #     BarColumn(),
        #     TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        #     TimeRemainingColumn(),
        #     TimeElapsedColumn(),
        #     disable=not self.accelerator.is_main_process,
        # )

        # with self.progress:
        #     self.pbar_epoch = self.progress.add_task(
        #         f"[green]Epoch", total=self.num_epochs
        #     )
        #     self.pbar_batch = self.progress.add_task(
        #         "[cyan]Batch", total=len(self.dataloader_train)
        #     )
        #     self.pbar_val = self.progress.add_task(
        #         "[cyan]Validation", total=len(self.dataloader_val)
        #     )

        # Compute the appropriate number of columns for the terminal
        self.ncols = min(os.get_terminal_size().columns, self.NCOLS)

        pbar_epochs = tqdm(
            desc="Epoch",
            total=self.num_epochs,
            ncols=self.ncols,
            disable=not self.accelerator.is_main_process,
        )

        for self.epoch in range(self.num_epochs):
            self._train_one_epoch()
            self._val()

            pbar_epochs.update()
            pbar_epochs.set_postfix(
                {"loss": f"{self.avg_loss:.5f}", "acc": f"{self.accuracy:.3f}"}
            )

            if self.accelerator.is_main_process:
                self._record()

        pbar_epochs.close()

        #     self.progress.update(
        #         self.pbar_epoch,
        #         advance=1,
        #         description=f"[green]Epoch {self.accuracy:.3f}",
        #     )
        #     self.progress.reset(self.pbar_batch)
        #     self.progress.reset(self.pbar_val)

        # self.progress.remove_task(self.pbar_batch)
        # self.progress.remove_task(self.pbar_val)

    def _train_one_epoch(self) -> None:
        # Setting before training
        self.model.train()
        self.loss_fn.train()
        total_loss = 0.0

        pbar_train = tqdm(
            desc="Train",
            total=len(self.dataloader_train),
            leave=False,
            ncols=self.ncols,
            disable=not self.accelerator.is_main_process,
        )

        for images, labels in self.dataloader_train:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)

            with self.accelerator.autocast():
                loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            # self.loss_iters.append(loss.item())
            # TODO: gather or reduce for loss

            self.accelerator.backward(loss)

            # TODO: clip_grad_norm

            self.accelerator.wait_for_everyone()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()

            # self.progress.update(
            #     self.pbar_batch,
            #     advance=1,
            #     description=f"[cyan]Batch {self.avg_loss:.3f}",
            # )

            pbar_train.update()
            pbar_train.set_postfix({"loss": f"{loss.item():.5f}"})

        pbar_train.close()

        # TODO: if not accelerator.optimizer_step_was_skipped:
        if self.cfg.get("lr_scheduler"):
            self.lr_scheduler.step()

        self.avg_loss = total_loss / len(self.dataloader_train)

    @torch.no_grad()
    def _val(self):
        # Setting before validation
        self.model.eval()
        self.loss_fn.eval()

        corrects = 0  # for compute accuracy

        pbar_val = tqdm(
            desc="Val",
            total=len(self.dataloader_val),
            leave=False,
            ncols=self.ncols,
            disable=not self.accelerator.is_main_process,
        )

        for images, labels in self.dataloader_val:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            if self.cfg.loss.get("weights", False):
                outputs = self.loss_fn(outputs, labels)

            predicts = torch.argmax(outputs, dim=1)
            corrects += (predicts == labels).sum()

            # TODO: reduce or gather the metrics
            # all_outputs, all_labels = self.accelerator.gather_for_metrics(
            #     outputs, labels
            # )

            # self.progress.update(self.pbar_val, advance=1)

            pbar_val.update()

        pbar_val.close()

        if self.accelerator.is_main_process:
            self.accuracy = corrects.item() / len(self.dataloader_val.dataset)

    def _record(self) -> None:
        self.results.append((self.epoch, self.avg_loss, self.accuracy))
        df = pd.DataFrame(self.results, columns=["Epoch", "Loss", "Accuracy"])
        df.to_csv(self.run_dir / "record.csv", index=False)

        # Save the last epoch
        if self.epoch == self.num_epochs - 1:
            self._save()

        # if self.skip_val:
        #     return

        # Save the best epoch (optional)
        better_acc = self.accuracy > self.best_acc
        is_save = self.epoch >= (self.num_epochs - self.record_last_n_epochs)
        is_save = is_save and self.record_last_n_epochs != 1  # last one = last
        if better_acc and is_save:
            (self.run_dir / f"best-{self.best_epoch}.pth").unlink(missing_ok=True)
            self._save()
            self.best_acc = self.accuracy
            self.best_epoch = self.epoch

    def _save(self) -> None:
        # Model
        model = self.accelerator.unwrap_model(self.model)
        model_file = self.run_dir / f"{self.best_or_last}-{self.epoch}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Saved the {self.best_or_last} model: {model_file}")

        # Loss
        if not self.cfg.loss.get("weights", False):
            return
        loss_fn = self.accelerator.unwrap_model(self.loss_fn)
        loss_file = self.run_dir / f"{self.best_or_last}-{self.epoch}-loss.pth"
        torch.save(loss_fn.state_dict(), loss_file)
        print(f"Saved the {self.best_or_last} loss fn: {loss_file}")
