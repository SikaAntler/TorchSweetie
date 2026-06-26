from pathlib import Path
from typing import override

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import ClsDataPack, create_cls_dataloader
from ..utils import LOSSES, MODELS, load_weights, load_weights_for_model
from .runner import RunnerBase


class ClsKFoldCrossValidator(RunnerBase):
    SCOPE = "classification"
    NCOLS = 100

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        self.outputs: torch.Tensor  # softmax

    @override
    def build_model(self) -> nn.Module:
        if "scope" not in self.cfg.model:
            self.cfg.model.scope = self.SCOPE
            self.cfg.loss.scope = self.SCOPE

        self.cfg.model.pop("_weights_", None)
        model = MODELS.create(self.cfg.model)
        load_weights_for_model(model, str(self.weights), True)

        # optional
        loss_fn: nn.Module = LOSSES.create(self.cfg.loss)
        if any(True for _ in loss_fn.parameters()):
            loss_weights = self.exp_dir / f"{self.weights.stem}-loss{self.weights.suffix}"
            load_weights(loss_fn, loss_weights)
            model = nn.Sequential(model, loss_fn)

        model.cuda()

        return model

    @override
    def build_dataloader(self) -> DataLoader:
        return create_cls_dataloader(self.cfg.test_dataloader)

    @override
    @torch.inference_mode()
    def run(self) -> None:
        assert self.dataloader

        pbar = tqdm(desc="Testing", total=len(self.dataloader), ncols=self.NCOLS)

        self.model.eval()

        outputs = []

        for data in self.dataloader:
            data: ClsDataPack
            data.inputs = data.inputs.cuda()
            data.targets = data.targets.cuda()
            data.ori_sizes = data.ori_sizes.cuda()

            out = self.model(data)
            outputs.append(out)

            pbar.update()

        self.outputs = torch.vstack(outputs)

        pbar.close()

    def softmax(self) -> torch.Tensor:
        return torch.softmax(self.outputs, 1).cpu()
