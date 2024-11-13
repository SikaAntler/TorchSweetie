from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from ..utils import LR_SCHEDULERS

__all__ = [
    "CosineAnnealingLRWarmUp",
]


@LR_SCHEDULERS.register()
class CosineAnnealingLRWarmUp(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        warmup: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup = warmup
        self.eta_min = eta_min
        self.scheduler = CosineAnnealingLR(
            optimizer, num_epochs - warmup, eta_min, last_epoch
        )
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch < self.warmup:
            for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                group["lr"] = lr
        elif self.last_epoch == self.warmup:
            for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = lr
        else:
            self.scheduler.step()

    def get_lr(self) -> list[float]:
        lrs = []
        for base_lr in self.base_lrs:
            k = (base_lr - self.eta_min) / (self.warmup + 1)
            b = k
            lr = k * self.last_epoch + b
            lrs.append(lr)

        return lrs
