from torch import nn
from torch.optim import SGD, AdamW

from ..utils import KEY_B, KEY_E, OPTIMIZERS, print_main


@OPTIMIZERS.register("AdamW")
def adamW(
    model: nn.Module | list[nn.Module],
    lr: float,
    betas: list[float] = [0.9, 0.999],
    weight_decay: float = 1e-2,
):
    assert len(betas) == 2

    params = _set_weight_decay(model, weight_decay)

    p0 = params[0]["params"]
    p1 = params[1]["params"]
    p2 = params[2]["params"]
    print_main(
        f"{KEY_B}AdamW{KEY_E} with parameter: "
        f"{len(p0)} weight(decay={weight_decay})"
        f" | {len(p1)} weight(decay=0.0)"
        f" | {len(p2)} bias"
    )

    return AdamW(params, lr, (betas[0], betas[1]), weight_decay=0.0)


@OPTIMIZERS.register("SGD")
def sgd(model: nn.Module | list[nn.Module], lr: float, momentum: float, weight_decay: float):
    params = _set_weight_decay(model, weight_decay)

    # if loss_fn is not None:
    #     loss_group = []
    #     for name, param in loss_fn.named_parameters():
    #         loss_group.append(param)
    #     params.append(loss_fn)

    return SGD(params, lr, momentum)


def _loop_params(params: list[dict], module: nn.Module):
    assert len(params) == 3

    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    for m in module.modules():
        for name, param in m.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            if name == "bias":
                params[2]["params"].append(param)
            elif name == "weight" and isinstance(m, bn):
                params[1]["params"].append(param)
            else:
                params[0]["params"].append(param)


def _set_weight_decay(model: nn.Module | list[nn.Module], weight_decay: float):
    # 将参数分为三组
    #   0) weights with decay
    #   1) norm weights no decay
    #   2) biases no decay
    params = [
        {"params": [], "weight_decay": weight_decay},
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": 0.0},
    ]

    if isinstance(model, nn.Module):
        _loop_params(params, model)
    elif isinstance(model, list):
        for module in model:
            _loop_params(params, module)

    return params
