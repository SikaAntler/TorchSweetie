from abc import ABC, abstractmethod
from pathlib import Path

import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader

from ..utils import DIR_B, DIR_E, URL_B, URL_E, load_config


class RunnerBase(ABC):
    SCOPE: str | None = None

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__()

        self.console = Console(highlight=False)

        # config
        self.cfg_file = cfg_file.absolute()
        self.cfg = load_config(self.cfg_file)
        self.console.print(f"Configuration file: {URL_B}{self.cfg_file}{URL_E}")

        # experimental directory
        self.exp_dir = exp_dir.absolute()
        assert exp_dir.exists()
        self.console.print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")

        self.weights = self.exp_dir / weights

        self.model = self.build_model()

        self.dataloader = self.build_dataloader()

    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def build_dataloader(self) -> DataLoader | None: ...

    @abstractmethod
    @torch.inference_mode()
    def run(self) -> None: ...
