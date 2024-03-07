import os
from datetime import datetime
from pathlib import Path
from argparse import Namespace

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..config import Config
from ..data import ClsDataset, create_sampler
from ..models import create_loss, create_model
from ..optim import create_lr_scheduler, create_optimizer
from ..utils import print_cost_time


class ClsTrainer:
    def __init__(self, args: Namespace, dataset: ClsDataset) -> None:
        # Store used attributes from args
        self.num_epochs = args.num_epochs
        self.record_last_n_epochs = args.record_last_n_epochs
        self.skip_val = args.skip_val
        self.num_workers = args.num_workers
        self.ram = args.ram

        # Get the list of devices, if there are more than 1 gpu, use ddp
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(devices) == 1:
            self.ddp = False
        else:
            self.ddp = True

        # Get the root path (project path)
        ROOT = Path().cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / args.cfg_file
        self.cfg = Config.fromfile(self.cfg_file)

        # Set the local_rank=0 and world_size=1 in default,
        # which means use single gpu
        self.local_rank = 0
        self.world_size = 1

        # PyTorch distributed settings (optional)
        if self.ddp:
            print(f"Using DDP")
            dist.init_process_group("nccl")

            # Use torchrun, so you are not required to use os.environ
            # self.local_rank = int(os.environ["LOCAL_RANK"])
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            # This is necessary, or there would be error:
            # Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 5e000
            torch.cuda.set_device(self.local_rank)

        # Create model, synchronize BN (optional)
        self.model = create_model(self.cfg.model)
        if self.ddp and self.cfg.model.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Create loss function
        self.loss_fn = create_loss(self.cfg.loss)

        # Set cuda, DDP (optional)
        self.model.cuda()
        if self.ddp:
            self.model = DistributedDataParallel(self.model)
        # if self.cfg.loss.weights:
        #     self.loss_fn.cuda()  # Loss function might have parameters
        #     if self.ddp:
        #         self.loss_fn = DistributedDataParallel(self.loss_fn)
        # self.loss_fn.cuda()
        # if self.ddp:
        #     self.loss_fn = DistributedDataParallel(self.loss_fn)
        self.loss_fn.cuda()
        if self.cfg.loss.weights and self.ddp:
            self.loss_fn = DistributedDataParallel(self.loss_fn)

        # Create optimizer
        if self.cfg.loss.weights:
            self.optimizer = create_optimizer(
                [self.model, self.loss_fn], self.cfg.optimizer
            )
        else:
            self.optimizer = create_optimizer(self.model, self.cfg.optimizer)

        # Create learning rate scheduler (optional)
        if self.cfg.lr_scheduler:
            self.lr_scheduler = create_lr_scheduler(
                self.optimizer, self.cfg.lr_scheduler
            )

        # Record results and weights
        # Only executed by local rank 0
        if self.local_rank == 0:
            # Check skip_val is whether conflict with record_last_n_epochs
            if args.skip_val and args.record_last_n_epochs > 1:
                raise ValueError(
                    "record_last_n_epochs should be less than or equal to 1 when skip_val is true!"
                )

            # If train.record_last_n_epochs is not specified,
            # record the best weights during the whole training phase
            # self.record_last_n_epochs = self.cfg.train.record_last_n_epochs
            # if self.record_last_n_epochs is None:
            #     self.record_last_n_epochs = self.cfg.train.num_epochs

            self.best_acc = 0
            self.best_epoch = -1
            self.total_loss = None
            self.accuracy = None
            self.results = []

        # Create dataloader
        if self.ddp and (self.cfg.dataloader.sampler is not None):
            raise ValueError("DDP is not compatible with customized sampler!")
        if self.ddp:
            self._create_dataloader_ddp(dataset)
        else:
            self._create_dataloader(dataset)

        # Running directory, used to record results and models
        # Only executed by local rank 0
        if self.local_rank == 0:
            self.run_dir = ROOT / "runs" / self.cfg_file.stem
            times = 0
            while self.run_dir.exists():
                times += 1
                self.run_dir = ROOT / "runs" / f"{self.cfg_file.stem}-{times}"
            self.run_dir.mkdir(parents=True)
            print(f"Running directory: {self.run_dir}🆕")

    def train(self):
        if self.local_rank == 0:
            begin = datetime.now()

        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch)

            if not self.skip_val:
                self._val(epoch)

            if self.local_rank == 0:
                self._record(epoch)

        if self.local_rank == 0:
            end = datetime.now()
            print_cost_time(begin, end)

    def _create_dataloader(self, dataset: ClsDataset) -> None:
        train_set = dataset(
            self.cfg.dataloader.train, "train", self.cfg.dataloader.dataset, self.ram
        )
        if self.cfg.dataloader.batch_sampler is None:
            self.train_loader = DataLoader(
                train_set,
                self.cfg.dataloader.batch_size,
                True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            batch_sampler = create_sampler(
                train_set.labels, self.cfg.dataloader.batch_sampler
            )
            self.train_loader = DataLoader(
                train_set,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        if self.skip_val:
            return
        val_set = dataset(
            self.cfg.dataloader.val, "val", self.cfg.dataloader.dataset, self.ram
        )
        self.val_loader = DataLoader(
            val_set,
            self.cfg.dataloader.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _create_dataloader_ddp(self, dataset: ClsDataset) -> None:
        train_set = dataset(
            self.cfg.dataloader.train, "train", self.cfg.dataloader.dataset, self.ram
        )
        # Only the train phase requires sampler
        self.train_sampler = DistributedSampler(train_set, drop_last=True)
        self.train_loader = DataLoader(
            train_set,
            self.cfg.dataloader.batch_size // self.world_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        if self.skip_val:
            return
        val_set = dataset(
            self.cfg.dataloader.val, "val", self.cfg.dataloader.dataset, self.ram
        )
        val_sampler = DistributedSampler(val_set, shuffle=False)
        self.val_loader = DataLoader(
            val_set,
            self.cfg.dataloader.batch_size // self.world_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _record(self, epoch) -> None:
        self.results.append((epoch, self.total_loss, self.accuracy))
        df = pd.DataFrame(self.results, columns=["Epoch", "Loss", "Accuracy"])
        df.to_csv(self.run_dir / "record.csv", index=False)

        # Save the last epoch
        if epoch == self.num_epochs - 1:
            self._save(epoch, "last")

        # Save the best epoch
        if self.skip_val:
            return
        better_acc = self.accuracy > self.best_acc
        is_save = epoch >= (self.num_epochs - self.record_last_n_epochs)
        is_save = is_save and self.record_last_n_epochs != 1  # last one = last
        if better_acc and is_save:
            (self.run_dir / f"best-{self.best_epoch}.pth").unlink(missing_ok=True)
            self._save(epoch, "best")
            self.best_acc = self.accuracy
            self.best_epoch = epoch

    def _save(self, epoch, best_or_last) -> None:
        if self.ddp:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        filename = self.run_dir / f"{best_or_last}-{epoch}.pth"
        torch.save(model, filename)
        tqdm.write(f"Saved the {best_or_last} model: {filename}")

        if self.cfg.loss.weights:
            if self.ddp:
                loss_fn = self.loss_fn.module.state_dict()
            else:
                loss_fn = self.loss_fn.state_dict()
            filename = self.run_dir / f"{best_or_last}_loss.pth"
            torch.save(loss_fn, filename)
            tqdm.write(f"Saved the {best_or_last} loss function: {filename}")

    def _train_one_epoch(self, epoch) -> None:
        if self.ddp:
            self.train_sampler.set_epoch(epoch)

        if self.local_rank == 0:
            train_pbar = tqdm(
                desc=f"Train Epoch [{epoch}/{self.num_epochs-1}]",
                total=len(self.train_loader),
                ncols=100,
            )
            total_loss = 0

        self.model.train()
        if self.cfg.loss.weights:
            self.loss_fn.train()
        for images, labels in self.train_loader:
            images, labels = images.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if self.ddp:
                dist.reduce(loss, 0, dist.ReduceOp.SUM)

            if self.local_rank == 0:
                with torch.no_grad():
                    loss /= self.world_size
                    total_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.5f}"})
                train_pbar.update()

        if self.cfg.lr_scheduler:
            self.lr_scheduler.step()

        if self.local_rank == 0:
            self.total_loss = total_loss / len(self.train_loader)
            train_pbar.set_postfix({"loss": f"{self.total_loss:.5f}"})
            train_pbar.close()

    @torch.no_grad()
    def _val(self, epoch) -> None:
        if self.local_rank == 0:
            val_pbar = tqdm(
                desc=f"val Epoch [{epoch}/{self.num_epochs-1}]",
                total=len(self.val_loader),
                ncols=100,
            )

        self.model.eval()
        if self.cfg.loss.weights:
            self.loss_fn.eval()
        corrects = 0
        for images, labels in self.val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            if self.cfg.loss.weights:
                outputs = self.loss_fn(outputs)
            predicts = torch.argmax(outputs, dim=1)
            corrects += (predicts == labels).sum()

            if self.local_rank == 0:
                val_pbar.update()

        if self.ddp:
            dist.reduce(corrects, 0, dist.ReduceOp.SUM)

        if self.local_rank == 0:
            self.accuracy = corrects.item() / len(self.val_loader.dataset)
            val_pbar.set_postfix({"acc": f"{self.accuracy:.3f}"})
            val_pbar.close()
