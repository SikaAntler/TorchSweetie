from pathlib import Path
from typing import Literal, Union

import pandas as pd
import torch
from rich import print
from sklearn.metrics import classification_report
from torch.nn import Module
from tqdm import tqdm

from torchsweetie.data import create_cls_dataloader
from torchsweetie.utils import LOSSES, MODELS, get_config, load_weights


class ClsTester:
    def __init__(
        self,
        cfg_file: Union[Path, str],
        times: int,
        best_or_last: str,
    ) -> None:
        # Get the root path (project path)
        ROOT = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / cfg_file
        self.cfg = get_config(self.cfg_file)

        # Running directory, used to record results and models
        if times == 0:
            self.run_dir = ROOT / "runs" / self.cfg_file.stem
        else:
            self.run_dir = ROOT / "runs" / f"{self.cfg_file.stem}-{times}"
        assert self.run_dir.exists()
        print(f"Running directory: {self.run_dir}:white_heavy_check_mark:")

        self.best_or_last = best_or_last

        # Model
        self.cfg.model.weights = None
        # Always, no matter whether used pretrained weights or not
        self.model = MODELS.create(self.cfg.model)
        self._load_weights(self.model, "model")

        # Loss function (optional)
        if self.cfg.loss.get("weights"):
            self.loss_fn = LOSSES.create(self.cfg.loss)
            self._load_weights(self.loss_fn, "loss")

        # Dataloader
        self.dataloader = create_cls_dataloader(
            self.cfg.dataloader, "test", False, False
        )

        # Target names
        self.target_names = pd.read_csv(
            self.cfg.dataloader.dataset.target_names, header=None
        )[0].to_list()

        # Store the labels and predictions
        self.y_true = []
        self.y_pred = []

        # Store the embeddings output by model
        self.embeddings = []

    def report(
        self, digits: int = 3, detailed: bool = True, export: bool = False
    ) -> None:
        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.target_names,
            digits=digits,
            output_dict=True,
            zero_division=0.0,
        )

        self._print_report(report, detailed)

        if export:
            report = pd.DataFrame(report)
            filename = self.run_dir / "report.csv"
            print(f"Saving the report: {filename}")
            report.to_csv(filename)

    @torch.no_grad()
    def test(self, store_embeddings: bool = False) -> None:
        pbar = tqdm(
            desc=f"Testing the {self.best_or_last} model",
            total=len(self.dataloader),
            ncols=80,
        )

        if len(self.y_true) + len(self.y_pred) != 0:
            print(f"You may run the test twice, since y_true and y_pred are not empty.")

        self.model.eval()
        if self.cfg.loss.get("weights"):
            self.loss_fn.eval()

        for images, labels in self.dataloader:
            self.y_true.extend(labels.tolist())
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)

            if self.cfg.loss.get("weights"):
                outputs = self.loss_fn(outputs, labels)

            if store_embeddings:
                self.embeddings.append(outputs)

            predicts = torch.argmax(outputs, dim=1)
            self.y_pred.extend(predicts.tolist())

            pbar.update()

        if store_embeddings:
            self.embeddings = torch.concat(self.embeddings)

        pbar.close()

    def _load_weights(self, module: Module, model_or_loss: Literal["model", "loss"]):
        # Find the weight file, load and send to gpu
        if model_or_loss == "model":
            weights = f"{self.best_or_last}-*[0-9].pth"
        elif model_or_loss == "loss":
            weights = f"{self.best_or_last}-*[0-9]-loss.pth"
        else:
            raise ValueError
        weights = list(self.run_dir.glob(weights))
        if len(weights) != 1:
            print(f"{len(weights)} have been found.")
        load_weights(module, weights[0])
        module.cuda()

    def _print_report(self, report: dict, detailed: bool) -> None:
        print(
            f"\n{'':>12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}\n\n"
        )

        for key, value in report.items():
            length = 0
            for c in key:
                if "\u4e00" <= c <= "\u9fa5":
                    length += 2
                else:
                    length += 1

            format_key = key
            while length < 12:
                format_key = " " + format_key
                length += 1

            if key == "accuracy":
                print(f"\n{format_key}{'':>12}{'':>12}{value:>12.3f}{'':>12}")
            else:
                precision = value["precision"]
                recall = value["recall"]
                f1_score = value["f1-score"]
                support = value["support"]
                print(
                    f"{format_key}{precision:>12.3f}{recall:>12.3f}{f1_score:12.3f}{support:>12}"
                )
