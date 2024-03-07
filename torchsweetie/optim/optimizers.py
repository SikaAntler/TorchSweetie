from omegaconf import DictConfig
from torch.nn import Module
from torch.optim import SGD, AdamW

from ..utils import OPTIMIZERS

__all__ = [
    "adamW",
    "sgd",
]

_NO_WEIGHT_DECAY = ["bias", "bn", "ln", "norm"]


@OPTIMIZERS.register
def adamW(cfg: DictConfig, model: Module | list[Module]):
    params = _set_weight_decay(model, cfg.weight_decay)
    return AdamW(params, cfg.lr)


@OPTIMIZERS.register
def sgd(cfg: DictConfig, model: Module | list[Module]):
    params = _set_weight_decay(model, cfg.weight_decay)

    # if loss_fn is not None:
    #     loss_group = []
    #     for name, param in loss_fn.named_parameters():
    #         loss_group.append(param)
    #     params.append(loss_fn)

    return SGD(params, cfg.lr, cfg.momentum)


def _loop_params(params: dict, module: Module):
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


def _set_weight_decay(model: Module | list[Module], weight_decay: float):
    params = [
        {"params": [], "weight_decay": weight_decay},
        {"params": []},
    ]
    if isinstance(model, Module):
        _loop_params(params, model)
    elif isinstance(model, list):
        for module in model:
            _loop_params(params, module)

    return params
