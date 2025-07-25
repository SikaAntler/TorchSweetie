from pathlib import Path

import pandas as pd
import torch
from rich import print
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import DIR_B, DIR_E, LOSSES, MODELS, URL_B, URL_E, get_config, load_weights


class ClsTester:
    NCOLS = 100

    def __init__(self, cfg_file: str, exp_dir: str, weights: str) -> None:
        # Get the root path (project path)
        self.root_dir = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = self.root_dir / cfg_file
        self.cfg = get_config(self.cfg_file)

        # Running directory, used to record results and models
        self.exp_dir = self.root_dir / exp_dir
        assert self.exp_dir.exists()
        print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")

        # Model
        model_weights = self.exp_dir / weights
        self.cfg.model.weights = model_weights
        self.model = MODELS.create(self.cfg.model)
        self.model.cuda()

        # Loss Function (Optional)
        loss_fn: nn.Module = LOSSES.create(self.cfg.loss)
        if list(loss_fn.parameters()) != []:
            loss_weights = self.exp_dir / f"{model_weights.stem}-loss{model_weights.suffix}"
            load_weights(loss_fn, loss_weights)
            loss_fn.cuda()
            self.loss_fn = loss_fn
        else:
            self.loss_fn = None

        # Dataloader
        dataloader_cfg = self.cfg.test_dataloader
        self.dataloader = create_cls_dataloader(dataloader_cfg)

        # Target names
        target_names = dataloader_cfg.dataset.target_names
        self.target_names = pd.read_csv(target_names, header=None)[0].to_list()

        # Store the labels and predictions
        self.y_true = []
        self.y_pred = []

    @torch.no_grad()
    def test(self) -> None:
        pbar = tqdm(desc=f"Testing", total=len(self.dataloader), ncols=self.NCOLS)

        if len(self.y_true) + len(self.y_pred) != 0:
            tqdm.write(f"You may run the test twice, since y_true and y_pred are not empty.")

        self.model.eval()
        if self.loss_fn is not None:
            self.loss_fn.eval()

        for images, labels in self.dataloader:
            self.y_true.extend(labels.tolist())
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)

            if self.loss_fn is not None:
                outputs = self.loss_fn(outputs, labels)  # pyright: ignore

            predicts = torch.argmax(outputs, dim=1)
            self.y_pred.extend(predicts.tolist())

            pbar.update()

        pbar.close()

    def report(self, digits: int = 3, export: bool = False) -> None:
        report: dict = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.target_names,
            digits=digits,
            output_dict=True,
            zero_division=0.0,  # pyright: ignore
        )

        self._print_report(report, digits)

        if export:
            df_report = pd.DataFrame(report)
            filename = self.exp_dir / "report.csv"
            df_report.to_csv(filename)
            print(f"Saved the report: {URL_B}{filename}{URL_E}")

    def _print_report(self, report: dict, digits: int) -> None:
        N = 12

        # 计算最长类名
        W = 0
        for key in report.keys():
            length = self._display_len(key)
            W = max(W, length)

        print(f"\n{'':>{W}}{'precision':>{N}}{'recall':>{N}}{'f1-score':>{N}}{'support':>{N}}\n\n")

        D = digits

        for key, value in report.items():
            class_name = self._format_string(key, W)

            if key == "accuracy":
                value = round(value, D)
                print(f"\n{class_name}{'':>{N}}{'':>{N}}{value:>{N}.{D}f}{'':>{N}}")
            else:
                precision = round(value["precision"], D)
                recall = round(value["recall"], D)
                f1_score = round(value["f1-score"], D)
                support = value["support"]
                print(
                    f"{class_name}{precision:>{N}.{D}f}{recall:>{N}.{D}f}{f1_score:{N}.{D}f}{support:>{N}}"
                )

    @staticmethod
    def _is_chinese(c: str) -> bool:
        assert len(c) == 1

        if "\u4e00" <= c <= "\u9fa5":
            return True
        else:
            return False

    def _display_len(self, string: str) -> int:
        length = 0
        for c in string:
            if self._is_chinese(c):
                length += 2
            else:
                length += 1

        return length

    def _format_string(self, string: str, max_length: int) -> str:
        length = self._display_len(string)

        while length < max_length:
            string = " " + string
            length += 1

        return string
