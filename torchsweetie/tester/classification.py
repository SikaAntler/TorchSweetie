from pathlib import Path

import pandas as pd
import torch
from rich import print
from sklearn.metrics import classification_report
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import DIR_B, DIR_E, LOSSES, MODELS, get_config, load_weights


class ClsTester:
    def __init__(self, cfg_file: Path | str, exp_name: str, weights: str) -> None:
        # Get the root path (project path)
        ROOT = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / cfg_file
        self.cfg = get_config(self.cfg_file)

        # Running directory, used to record results and models
        self.run_dir = ROOT / "runs" / self.cfg_file.stem / exp_name
        assert self.run_dir.exists()
        print(
            f"Running directory: {DIR_B}{self.run_dir}{DIR_E}:white_heavy_check_mark:"
        )

        # self.best_or_last = best_or_last

        # Model
        model_weights = self.run_dir / weights
        self.cfg.model.weights = model_weights
        self.model = MODELS.create(self.cfg.model)
        self.model.cuda()
        # self._load_weights(self.model, "model")

        # Loss function (optional)
        loss_cfg = self.cfg.loss
        if loss_cfg.get("weights", False):
            loss_cfg.pop("weights")
            self.loss_fn = LOSSES.create(self.cfg.loss)
            loss_weights = (
                self.run_dir / f"{model_weights.stem}-loss{model_weights.suffix}"
            )
            load_weights(self.loss_fn, loss_weights)
            self.loss_fn.cuda()
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
            zero_division=0.0,  # pyright: ignore
        )

        self._print_report(report, detailed)

        if export:
            report = pd.DataFrame(report)
            filename = self.run_dir / "report.csv"
            report.to_csv(filename)
            print(f"Saved the report: {filename}")

    @torch.no_grad()
    def test(self, store_embeddings: bool = False) -> None:
        pbar = tqdm(desc=f"Testing", total=len(self.dataloader), ncols=80)

        if len(self.y_true) + len(self.y_pred) != 0:
            tqdm.write(
                f"You may run the test twice, since y_true and y_pred are not empty."
            )

        self.model.eval()
        if self.loss_fn is not None:
            self.loss_fn.eval()

        embeddings = []
        for images, labels in self.dataloader:
            self.y_true.extend(labels.tolist())
            images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)

            if self.loss_fn is not None:
                outputs = self.loss_fn(outputs, labels)

            if store_embeddings:
                embeddings.append(outputs)

            predicts = torch.argmax(outputs, dim=1)
            self.y_pred.extend(predicts.tolist())

            pbar.update()

        if store_embeddings:
            self.embeddings = torch.concat(embeddings)

        pbar.close()

    # def _load_weights(self, module: Module, model_or_loss: Literal["model", "loss"]):
    #     # Find the weight file, load and send to gpu
    #     if model_or_loss == "model":
    #         weights = f"{self.best_or_last}-*[0-9].pth"
    #     elif model_or_loss == "loss":
    #         weights = f"{self.best_or_last}-*[0-9]-loss.pth"
    #     else:
    #         raise ValueError
    #     weights = list(self.run_dir.glob(weights))
    #     if len(weights) != 1:
    #         print(f"{len(weights)} weights have been found, {weights[0]} has been used")
    #     weights = weights[0]
    #     load_weights(module, weights)
    #     module.cuda()

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
