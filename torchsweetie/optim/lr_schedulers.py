import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ..utils import SCHEDULERS


@SCHEDULERS.register()
def LinearWarmUpLR(
    optimizer: Optimizer,
    num_steps: int,
    start_factor: float = 0.01,
    end_factor: float = 0.0,
    warmup: int = 0,
) -> LRScheduler:
    assert num_steps > 0
    assert warmup >= 0
    assert warmup <= num_steps
    assert 0.0 < start_factor <= 1.0
    assert 0.0 <= end_factor <= 1.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (1 - start_factor) * step / warmup + start_factor

        k = (end_factor - 1.0) / (num_steps - warmup)
        return k * (step - warmup) + 1.0

    return LambdaLR(optimizer, lr_lambda)


@SCHEDULERS.register()
def CosineAnnealingWarmUpLR(
    optimizer: Optimizer,
    num_steps: int,
    start_factor: float = 0.01,
    end_factor: float = 0.0,
    warmup: int = 0,
) -> LRScheduler:
    assert num_steps > 0
    assert warmup >= 0
    assert warmup <= num_steps
    assert 0.0 < start_factor <= 1.0
    assert 0.0 <= end_factor <= 1.0

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (1 - start_factor) * step / warmup + start_factor

        factor = 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (num_steps - warmup)))
        return (1.0 - end_factor) * factor + end_factor

    return LambdaLR(optimizer, lr_lambda)


@SCHEDULERS.register()
def YOLOv5LR(
    optimizer: Optimizer, num_steps: int, end_factor: float, warmup: int, warmup_bias_lr: float
) -> LRScheduler:
    def decay(step: int) -> float:
        return (1 - step / num_steps) * (1.0 - end_factor) + end_factor

    def normal(step: int) -> float:
        if step < warmup:
            return decay(step) * step / warmup

        return decay(step)

    def bias(step: int) -> float:
        if step < warmup:
            initial_lr = optimizer.param_groups[2]["initial_lr"]
            target_lr = initial_lr * decay(step)
            abs_lr = (target_lr - warmup_bias_lr) / warmup * step + warmup_bias_lr
            return abs_lr / initial_lr

        return decay(step)

    return LambdaLR(optimizer, [normal, normal, bias])
