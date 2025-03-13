from torch import nn
from torch.optim import SGD, AdamW

from ..utils import OPTIMIZERS

__all__ = [
    "adamW",
    "sgd",
]

_NO_WEIGHT_DECAY = ["bias", "bn", "ln", "norm"]


@OPTIMIZERS.register("AdamW")
def adamW(model: nn.Module | list[nn.Module], lr: float, weight_decay: float):
    params = _set_weight_decay(model, weight_decay)
    return AdamW(params, lr)


@OPTIMIZERS.register("SGD")
def sgd(model: nn.Module | list[nn.Module], lr: float, momentum: float, weight_decay: float):
    params = _set_weight_decay(model, weight_decay)

    # if loss_fn is not None:
    #     loss_group = []
    #     for name, param in loss_fn.named_parameters():
    #         loss_group.append(param)
    #     params.append(loss_fn)

    return SGD(params, lr, momentum)


def _loop_params(params: list, module: nn.Module):
    for name, param in module.named_parameters():
        # Ignore the params which are freezed, equal to filter(lambda ...)
        if not param.requires_grad:
            continue

        no_wd_flag = False
        for item in _NO_WEIGHT_DECAY:
            if item in name:
                no_wd_flag = True
                break

        if no_wd_flag:
            params[1]["params"].append(param)
        else:
            params[0]["params"].append(param)


def _set_weight_decay(model: nn.Module | list[nn.Module], weight_decay: float):
    params = [
        {"params": [], "weight_decay": weight_decay},
        {"params": []},
    ]
    if isinstance(model, nn.Module):
        _loop_params(params, model)
    elif isinstance(model, list):
        for module in model:
            _loop_params(params, module)

    return params
