from typing import Callable, Sequence

from accelerate.optimizer import AcceleratedOptimizer


class LambdaMomentum:
    def __init__(
        self,
        optimizer: AcceleratedOptimizer,
        momentum_lambda: Callable[[int], float] | Sequence[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        if isinstance(momentum_lambda, Sequence):
            if len(momentum_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} momentum lambdas, "
                    f"but got {len(momentum_lambda)} instead"
                )
            self.momentum_lambdas = list(momentum_lambda)
        else:
            self.momentum_lambdas = [momentum_lambda] * len(optimizer.param_groups)

        for group in optimizer.param_groups:
            if "momentum" not in group:
                raise ValueError("LambdaMomentum only support optimizers with a 'momentum' field")
            group.setdefault("initial_momentum", group["momentum"])

        self.base_momentums = [group["initial_momentum"] for group in optimizer.param_groups]

        self.step()

    def get_momentum(self) -> list[float]:
        return [
            base_momentum * momentum_lambda(self.last_epoch)  # ty: ignore
            for base_momentum, momentum_lambda in zip(self.base_momentums, self.momentum_lambdas)
        ]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        values = self.get_momentum()

        for group, momentum in zip(self.optimizer.param_groups, values):
            group["momentum"] = momentum

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_momentums": self.base_momentums}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict["last_epoch"]
        self.base_momentums = state_dict["base_momentums"]
