from typing import Any, Callable

from omegaconf import DictConfig
from rich import print


class Registry:
    def __init__(self, name) -> None:
        self._name = name
        self._module_dict = {}

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._module_dict

    def __getitem__(self, key: str):
        return self._module_dict[key.lower()]

    def __len__(self) -> int:
        return len(self._module_dict)

    def create(self, cfg: DictConfig, *args, **kwargs) -> Any:
        module = self[cfg.name]
        print(f"Using the {self._name}: {cfg.name}")

        return module(cfg, *args, **kwargs)

    def get(self, key: str, default=None):
        return self._module_dict.get(key.lower(), default)

    def items(self):
        return self._module_dict.items()

    def keys(self):
        return self._module_dict.keys()

    @property
    def name(self) -> str:
        return self._name

    def register(self, item: Callable) -> Callable:
        key = item.__name__.lower()
        if key in self._module_dict:
            print(f"WARNING: `{key}` has already existed!")
        self._module_dict[key] = item

        return item


DATASETS = Registry("dataset")
LOSSES = Registry("loss")
LR_SCHEDULERS = Registry("lr_scheduler")
MODELS = Registry("model")
OPTIMIZERS = Registry("optimizer")
SAMPLERS = Registry("sampler")
UTILS = Registry("utils")
