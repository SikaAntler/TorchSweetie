from pathlib import Path
from typing import override

import pandas as pd
import torch
from rich import print
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import ClsDataPack, create_cls_dataloader
from ..utils import (
    LOSSES,
    MODELS,
    URL_B,
    URL_E,
    load_weights,
    load_weights_for_model,
    print_cls_report,
)
from .runner import RunnerBase


class ClsTester(RunnerBase):
    SCOPE = "classification"
    NCOLS = 100

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        # store the labels and predictions
        self.y_true = []
        self.y_pred = []

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

        if len(self.y_true) + len(self.y_pred) != 0:
            tqdm.write("You may run the test twice, since y_true and y_pred are not empty.")

        self.model.eval()

        for data in self.dataloader:
            data: ClsDataPack
            self.y_true.extend(data.targets.tolist())
            data.inputs = data.inputs.cuda()
            data.targets = data.targets.cuda()
            data.ori_sizes = data.ori_sizes.cuda()

            outputs = self.model(data)

            predicts = torch.argmax(outputs, dim=1)
            self.y_pred.extend(predicts.tolist())

            pbar.update()

        pbar.close()

    def report(self, digits: int) -> None:
        classes_file = self.cfg.test_dataloader.dataset.classes_file
        classes = pd.read_csv(classes_file, header=None)[0].to_list()
        labels = list(range(len(classes)))

        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=labels,
            target_names=classes,
            output_dict=True,
            zero_division=0.0,
        )

        report = pd.DataFrame(report).T

        filename = self.exp_dir / "report.csv"
        report.to_csv(filename)
        print(f"Saved the report: {URL_B}{filename}{URL_E}")

        print_cls_report(filename, digits)
