from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler

from ..utils import LR_SCHEDULERS


@LR_SCHEDULERS.register()
class CosineAnnealingLRWarmUp(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        warmup: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch)

        self.warmup = warmup
        self.eta_min = eta_min

        T_max = num_steps - warmup
        self.scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)

    def step(self, epoch: int | None = None) -> None:
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

    def get_lr(self) -> list[float | Tensor]:
        lrs = []
        for base_lr in self.base_lrs:
            k = (base_lr - self.eta_min) / (self.warmup + 1)
            b = k
            lr = k * self.last_epoch + b
            lrs.append(lr)

        return lrs


@LR_SCHEDULERS.register()
def LinearLRWarmUp(
    optimizer: Optimizer, num_steps: int, end_factor: float = 0.0, warmup: int = 0
) -> LRScheduler:
    def linear_lr_warm_up(step: int) -> float:
        if step < warmup:
            return (1 - end_factor) * step / warmup + end_factor

        return (1 - (step - warmup) / (num_steps - warmup)) * (1.0 - end_factor) + end_factor

    return LambdaLR(optimizer, linear_lr_warm_up)


# class LinearLRWarmUp(LRScheduler):
#     def __init__(
#         self,
#         optimizer: Optimizer,
#         start_factor: float,
#         end_factor: float,
#         num_steps: int,
#         warmup: int = 0,
#         last_epoch: int = -1,
#     ) -> None:
#         super().__init__(optimizer, last_epoch)
#
#         self.warmup = warmup
#         self.start_factor = start_factor
#         self.end_factor = end_factor
#
#         total_iters = num_steps - warmup
#         self.scheduler = LinearLR(optimizer, start_factor, end_factor, total_iters, last_epoch)
#
#     def step(self, epoch=None) -> None:
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch
#         if self.last_epoch < self.warmup:
#             for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#                 group["lr"] = lr
#         elif self.last_epoch == self.warmup:
#             for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
#                 group["lr"] = lr
#         else:
#             self.scheduler.step()
#
#     def get_lr(self) -> list[float | Tensor]:
#         lrs = []
#         for base_lr in self.base_lrs:
#             k = (base_lr - self.end_factor) / (self.warmup + 1)
#             b = k
#             lr = k * self.last_epoch + b
#             lrs.append(lr)
#
#         return lrs
