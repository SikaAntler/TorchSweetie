import weakref
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from accelerate import Accelerator
from rich import print
from torch import distributed, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ..utils import (
    DIR_B,
    DIR_E,
    KEY_B,
    KEY_E,
    LOSSES,
    LR_SCHEDULERS,
    MODELS,
    OPTIMIZERS,
    URL_B,
    URL_E,
    ModelEMA,
    load_config,
    load_weights_for_model,
    save_config,
    seed_all_rng,
)


class HookBase:
    trainer: TrainerBase

    def before_train(self) -> None:
        pass

    def after_train(self) -> None:
        pass

    def before_step(self) -> None:
        pass

    def after_backward(self) -> None:
        pass

    def after_step(self) -> None:
        pass


class TrainerBase(ABC):
    SCOPE: str | None = None

    def __init__(self, cfg_file: Path, run_dir: Path) -> None:
        super().__init__()

        # config
        self.cfg_file = cfg_file.absolute()
        self.cfg = load_config(self.cfg_file)

        # train
        self.clip_grad = self.cfg.train.get("clip_grad")
        if self.clip_grad is not None:
            self.max_norm = self.clip_grad.max_norm
            self.norm_type = self.clip_grad.get("norm_type", 2)

        seed_all_rng(self.cfg.train.get("seed", 1997), self.cfg.train.get("deterministic", False))

        # TODO: split_batch取值问题：
        # 如果是默认的False，在多卡时总batch_size会和配置文件不一致
        # 如果是True，有batch_sampler的情况下需要重设为False
        # 考虑到DDP模式下训练结果有别于单卡，batch_size可以不一致
        split_batch = False
        mixed_precision = self.cfg.train.get("mixed_precision", "no")
        self.accelerator = Accelerator(
            split_batches=split_batch,
            mixed_precision=mixed_precision,
            step_scheduler_with_optimizer=False,  # 否则多卡时一次step等于num_processes次
        )
        if self.accelerator.is_main_process and mixed_precision != "no":
            print(f"Using mixed precision: {KEY_B}{mixed_precision}{KEY_E}")
        self.device = self.accelerator.device

        # Running directory, used to record results and models
        # Only executed by the main process
        if self.accelerator.is_main_process:
            print(f"Configuration file: {URL_B}{self.cfg_file}{URL_E}")

            date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.exp_dir = run_dir.absolute() / self.cfg_file.stem / date_time
            self.exp_dir.mkdir(parents=True)
            print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")
            # Save the config to the parent of experiment directory
            save_config(self.cfg, self.exp_dir.parent / "config.yaml")

        self.model = self.build_model()

        self.loss_fn = self.build_loss_fn()

        self.optimizer = self.build_optimizer()

        self.lr_scheduler = self.build_lr_scheduler()

        self.train_dataloader = self.build_train_dataloader()

        self.val_dataloader = self.build_val_dataloader()

        self.prepare()

        self.ema = self.build_ema()

        self._hooks: list[HookBase] = []

    def build_model(self) -> nn.Module:
        if "scope" not in self.cfg.model:
            self.cfg.model.scope = self.SCOPE

        weights = self.cfg.model.pop("_weights_", None)
        model = MODELS.create(self.cfg.model)
        if weights is not None:
            load_weights_for_model(model, weights, self.accelerator.is_main_process)

        if self.cfg.train.get("sync_bn", False):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model

    def build_loss_fn(self) -> nn.Module:
        if "scope" not in self.cfg.loss:
            self.cfg.loss.scope = self.SCOPE

        loss_fn = LOSSES.create(self.cfg.loss)

        if list(loss_fn.parameters()) != []:
            self.loss_params = True
        else:
            self.loss_params = False

        return loss_fn

    def build_optimizer(self) -> Optimizer:
        if "scope" not in self.cfg.optimizer:
            self.cfg.optimizer.scope = self.SCOPE

        return OPTIMIZERS.create(self.cfg.optimizer, self.model)

    def build_lr_scheduler(self) -> LRScheduler:
        self.lr_scheduler = self.cfg.get("lr_scheduler")

        if "scope" not in self.cfg.lr_scheduler:
            self.cfg.lr_scheduler.scope = self.SCOPE

        return LR_SCHEDULERS.create(self.cfg.lr_scheduler, self.optimizer)

    @abstractmethod
    def build_train_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def build_val_dataloader(self) -> DataLoader | None: ...

    def prepare(self) -> None:
        self.model, self.loss_fn, self.optimizer, self.lr_scheduler, self.train_dataloader = (
            self.accelerator.prepare(
                self.model, self.loss_fn, self.optimizer, self.lr_scheduler, self.train_dataloader
            )
        )

    def build_ema(self) -> ModelEMA | None:
        if "ema" in self.cfg:
            return ModelEMA(
                self.accelerator.unwrap_model(self.model),
                self.cfg.ema.decay,
                self.cfg.ema.tau,
                updates=0,
            )
        else:
            return None

    def register_hooks(self, hooks: list[HookBase]) -> None:
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    @abstractmethod
    def train(self) -> None: ...

    def before_train(self) -> None:
        for h in self._hooks:
            h.before_train()

    def after_train(self) -> None:
        for h in self._hooks:
            h.after_train()

        self.accelerator.wait_for_everyone()

        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()

    def before_step(self) -> None:
        for h in self._hooks:
            h.before_step()

    @abstractmethod
    def run_step(self) -> None: ...

    def after_backward(self) -> None:
        for h in self._hooks:
            h.after_backward()

    def after_step(self) -> None:
        for h in self._hooks:
            h.after_step()
