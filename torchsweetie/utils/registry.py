from collections.abc import Callable, ItemsView, KeysView, ValuesView
from typing import Any, override

from omegaconf import DictConfig
from rich import print

type Factory[T] = Callable[..., T]


class Registry[T]:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: dict[str, Factory[T]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __getitem__(self, key: str) -> Factory[T]:
        return self._module_dict[key]

    def __len__(self) -> int:
        return len(self._module_dict)

    @property
    def name(self) -> str:
        return self._name

    def register(
        self, name: str | None = None, scope: str | None = None
    ) -> Callable[[Factory[T]], Factory[T]]:

        def _register(factory: Factory[T]) -> Factory[T]:
            key = name or getattr(factory, "__name__", factory.__class__.__name__)
            if scope:
                key = f"{scope}/{key}"
            if key in self._module_dict:
                print(f"WARNING: `{key}` has already existed!")
            self._module_dict[key] = factory
            return factory

        return _register

    def create(self, cfg: DictConfig, *args: Any, **kwargs: Any) -> T:
        _cfg = cfg.copy()
        name = _cfg.pop("name")
        scope = _cfg.pop("scope", None)
        key = f"{scope}/{name}" if scope else name
        if key not in self:
            key = name

        return self[key](*args, **_cfg, **kwargs)  # ty: ignore

    def get(self, key: str, default: Factory[T] | None = None) -> Factory[T] | None:
        return self._module_dict.get(key, default)

    def items(self) -> ItemsView[str, Factory[T]]:
        return self._module_dict.items()

    def keys(self) -> KeysView[str]:
        return self._module_dict.keys()

    def values(self) -> ValuesView[Factory[T]]:
        return self._module_dict.values()


BATCH_SAMPLERS = Registry("batch_sampler")
LOSSES = Registry("loss")
MODELS = Registry("model")
SAMPLERS = Registry("sampler")
SIMILARITY = Registry("similarity")
TRANSFORMS = Registry("transform")
UTILS = Registry("utils")


class OptimizerRegistry[T](Registry[T]):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def create(self, cfg: DictConfig, model) -> T:
        _cfg = cfg.copy()
        name = _cfg.pop("name")
        scope = _cfg.pop("scope", None)
        key = f"{scope}/{name}" if scope else name
        if key not in self:
            key = name

        return self[key](model, **_cfg)  # ty: ignore


OPTIMIZERS = OptimizerRegistry("optimizer")


class LRSchedulerRegistry[T](Registry[T]):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def create(self, cfg: DictConfig, optimizer) -> T:
        _cfg = cfg.copy()
        name = _cfg.pop("name")
        scope = _cfg.pop("scope", None)
        key = f"{scope}/{name}" if scope else name
        if key not in self:
            key = name

        return self[key](optimizer, **_cfg)  # ty: ignore


SCHEDULERS = LRSchedulerRegistry("scheduler")
