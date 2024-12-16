from typing import Callable

from omegaconf import DictConfig
from rich import print


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __getitem__(self, key: str) -> Callable:
        return self._module_dict[key]

    def __len__(self) -> int:
        return len(self._module_dict)

    @property
    def name(self) -> str:
        return self._name

    def register(self, name=None) -> Callable:

        def _register(cls: Callable) -> Callable:
            key = cls.__name__ if name is None else name
            if key in self._module_dict:
                print(f"WARNING: `{key}` has already existed!")
            self._module_dict[key] = cls
            return cls

        return _register

    def create(self, cfg: DictConfig):
        _cfg = cfg.copy()
        name = _cfg.pop("name")

        return self[name](**_cfg)  # pyright: ignore

    def get(self, key: str, default=None):
        return self._module_dict.get(key, default)

    def items(self):
        return self._module_dict.items()

    def keys(self):
        return self._module_dict.keys()

    def values(self):
        return self._module_dict.values()


BATCH_SAMPLERS = Registry("sampler")
LOSSES = Registry("loss")
MODELS = Registry("model")
SIMILARITY = Registry("similarity")
TRANSFORMS = Registry("transform")
UTILS = Registry("utils")


class OptimizerRegistry(Registry):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def create(self, model, cfg: DictConfig):
        _cfg = cfg.copy()
        name = _cfg.pop("name")

        return self[name](model, **_cfg)  # pyright: ignore


OPTIMIZERS = OptimizerRegistry("optimizer")


class LRSchedulerRegistry(Registry):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def create(self, optimizer, cfg: DictConfig):
        _cfg = cfg.copy()
        name = _cfg.pop("name")

        return self[name](optimizer, **_cfg)  # pyright: ignore


LR_SCHEDULERS = LRSchedulerRegistry("lr_scheduler")
