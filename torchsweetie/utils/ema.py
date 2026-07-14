import math
from copy import deepcopy
from typing import Any

import torch
from torch import distributed, nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float, tau: float, updates) -> None:
        self.ema = deepcopy(model).eval()
        self.base_decay = decay
        self.tau = tau
        self.updates = updates

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @property
    def decay(self) -> float:
        return self.base_decay * (1 - math.exp(-self.updates / self.tau))

    @torch.inference_mode()
    def update(self, model: nn.Module) -> None:
        self.updates += 1

        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, v in ema_state.items():
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(model_state[k], alpha=1.0 - self.decay)
            else:
                v.copy_(model_state[k])

        self.synchronize()

    @torch.inference_mode()
    def synchronize(self, src: int = 0) -> None:
        if not distributed.is_available() or not distributed.is_initialized():
            return

        for tensor in self.ema.state_dict().values():
            distributed.broadcast(tensor, src)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema": self.ema.state_dict(),
            "updates": self.updates,
            "base_decay": self.base_decay,
            "tau": self.tau,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema.load_state_dict(state["ema"])
        self.updates = state["updates"]
        self.base_decay = state["base_decay"]
        self.tau = state["tau"]
