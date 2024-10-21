import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Union

import pandas as pd
import torch
from accelerate import Accelerator
from rich import print
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import (
    DIR_B,
    DIR_E,
    KEY_B,
    KEY_E,
    LOSSES,
    LR_SCHEDULERS,
    MODELS,
    OPTIMIZERS,
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
            print(f"Using mixed precision: {KEY_B}{mixed_precision}{KEY_E}")
        self.accelerator = Accelerator(
            split_batches=split_batch, mixed_precision=mixed_precision
        )
        self.device = self.accelerator.device

        # Running directory, used to record results and models
        # Only executed by the main process
        if self.accelerator.is_main_process:
            date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = ROOT / "runs" / self.cfg_file.stem / date_time
            self.run_dir.mkdir(parents=True)
            print(f"Running directory: {DIR_B}{self.run_dir}{DIR_E}")

        # Model
        model = MODELS.create(self.cfg.model)

        # TODO: 需要freeze功能，初步设想是提供一个UTILS函数注册器
        # if self.cfg.train.get("freeze"):
        #     UTILS.create(self.cfg.train.freeze, model)

        # Synchronize BN for DDP mode
        if self.cfg.train.get("sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Loss Function
        loss_cfg = self.cfg.loss
        if loss_cfg.get("weights", False):
            self.loss_weights = loss_cfg.pop("weights")
        else:
            self.loss_weights = False
        loss_fn = LOSSES.create(loss_cfg)

        # Optimizer & LR Scheduler
        optimizer = OPTIMIZERS.create(model, self.cfg.optimizer)
        lr_scheduler_cfg = self.cfg.get("lr_scheduler")
        if lr_scheduler_cfg is None:
            self.lr_scheduler = None
        else:
            lr_scheduler = LR_SCHEDULERS.create(optimizer, lr_scheduler_cfg)
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        # Train & Val DataLoaders
        train_dataloader = create_cls_dataloader(self.cfg.train_dataloader)
        val_dataloader = create_cls_dataloader(self.cfg.val_dataloader)

        # Prepare all
        (
            self.model,
            self.loss_fn,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
        ) = self.accelerator.prepare(
            model, loss_fn, optimizer, train_dataloader, val_dataloader
        )

        # Train
        self.num_epochs = self.cfg.train.num_epochs
        self.clip_grad = self.cfg.train.get("clip_grad")
        if self.clip_grad is not None:
            self.max_norm = self.clip_grad.max_norm
            self.norm_type = self.clip_grad.get("norm_type", 2)
        # Val
        val_cfg = self.cfg.get("val")
        if val_cfg is None:
            self.val_interval = 1
            self.val_skip = False
        else:
            self.val_interval = self.cfg.val.get("interval", 1)
            self.val_skip = self.cfg.val.get("skip", False)
        # Save
        save_cfg = self.cfg.get("save")
        if save_cfg is None:
            self.save_interval = 999
            self.save_last = True
            self.save_best = False
        else:
            self.save_interval = self.cfg.save.get("interval", 999)
            self.save_last = self.cfg.save.get("last", True)
            self.save_best = self.cfg.save.get("best", False)

        # Useful parameters
        if self.accelerator.is_main_process:
            self.accuracy = 0
            self.avg_loss = None
            self.best_acc = 0
            self.best_epoch = -1
            self.results = []
            # self.loss_iters = []

    def train(self) -> None:
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

            if (not self.val_skip) and ((self.epoch + 1) % self.val_interval == 0):
                self._val()

            if self.accelerator.is_main_process:
                pbar_epochs.update()
                pbar_epochs.set_postfix(
                    {"loss": f"{self.avg_loss:.5f}", "acc": f"{self.accuracy:.3f}"}
                )
                self._record()

        pbar_epochs.close()

    def _train_one_epoch(self) -> None:
        # Setting before training
        self.model.train()
        self.loss_fn.train()
        total_loss = 0.0

        pbar_train = tqdm(
            desc="Train",
            total=len(self.train_dataloader),
            leave=False,
            ncols=self.ncols,
            disable=not self.accelerator.is_main_process,
        )

        for images, labels in self.train_dataloader:
            self.optimizer.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)

            with self.accelerator.autocast():
                loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            # self.loss_iters.append(loss.item())
            # TODO: gather or reduce for loss

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_norm,
                        self.norm_type,
                    )

            self.accelerator.wait_for_everyone()

            self.optimizer.step()

            self.accelerator.wait_for_everyone()

            pbar_train.update()
            pbar_train.set_postfix({"loss": f"{loss.item():.5f}"})

        pbar_train.close()

        # TODO: if not accelerator.optimizer_step_was_skipped:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.avg_loss = total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _val(self) -> None:
        # Setting before validation
        self.model.eval()
        self.loss_fn.eval()

        pbar_val = tqdm(
            desc="Val",
            total=len(self.val_dataloader),
            leave=False,
            ncols=self.ncols,
            disable=not self.accelerator.is_main_process,
        )

        corrects = 0

        for images, labels in self.val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            if self.loss_weights:
                outputs = self.loss_fn(outputs, labels)

            predicts = torch.argmax(outputs, dim=1)
            corrects += (predicts == labels).sum().cpu().item()

            # TODO: reduce or gather the metrics
            # all_outputs, all_labels = self.accelerator.gather_for_metrics(
            #     outputs, labels
            # )

            pbar_val.update()

        pbar_val.close()

        if self.accelerator.is_main_process:
            self.accuracy = corrects / len(self.val_dataloader.dataset)

    def _record(self) -> None:
        self.results.append((self.epoch, self.avg_loss, self.accuracy))
        df = pd.DataFrame(self.results, columns=["Epoch", "Loss", "Accuracy"])  # pyright: ignore
        df.to_csv(self.run_dir / "record.csv", index=False)

        # Save the interval(epoch)
        if (self.epoch + 1) % self.save_interval == 0:
            self._save("epoch")

        # Save the last epoch
        if self.save_last and (self.epoch == self.num_epochs - 1):
            self._save("last")

        # Save the best epoch
        if self.save_best and (self.accuracy > self.best_acc):
            (self.run_dir / f"best-{self.best_epoch}.pth").unlink(missing_ok=True)
            (self.run_dir / f"best-{self.best_epoch}-loss.pth").unlink(missing_ok=True)
            self.best_acc = self.accuracy
            self.best_epoch = self.epoch
            self._save("best")

    def _save(self, prefix: Literal["epoch", "last", "best"]) -> None:
        # Model
        model = self.accelerator.unwrap_model(self.model)
        model_file = self.run_dir / f"{prefix}-{self.epoch}.pth"
        torch.save(model.state_dict(), model_file)
        tqdm.write(f"Saved the {prefix} model: {model_file}")

        # Loss
        if not self.loss_weights:
            return
        loss_fn = self.accelerator.unwrap_model(self.loss_fn)
        loss_file = self.run_dir / f"{prefix}-{self.epoch}-loss.pth"
        torch.save(loss_fn.state_dict(), loss_file)
        tqdm.write(f"Saved the {prefix} loss fn: {loss_file}")
