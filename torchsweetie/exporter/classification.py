from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
import torch.nn as nn
from rich import print

from torchsweetie.utils import LOSSES, MODELS, get_config, load_weights


class ClsExporter:
    def __init__(self, cfg_file: Path | str, exp_name: str, weights: str) -> None:
        # Get the root path (project path)
        ROOT = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / cfg_file
        self.cfg = get_config(self.cfg_file)

        # Running directory, used to record results and models
        self.run_dir = ROOT / "runs" / self.cfg_file.stem / exp_name
        assert self.run_dir.exists()
        print(f"Running directory: {self.run_dir}:white_heavy_check_mark:")

        # Model
        model_weights = self.run_dir / weights
        self.cfg.model.weights = model_weights
        self.model = MODELS.create(self.cfg.model)

        loss_cfg = self.cfg.loss
        if loss_cfg.get("weights", False):
            loss_cfg.pop("weights")
            self.loss_fn = LOSSES.create(self.cfg.loss)
            loss_weights = (
                self.run_dir / f"{model_weights.stem}-loss{model_weights.suffix}"
            )
            load_weights(self.loss_fn, loss_weights)
            self.model = nn.Sequential(self.model, self.loss_fn)
        else:
            self.loss_fn = None

    def export_onnx(
        self,
        input_size: tuple,
        half: bool = False,
        device: str | int = "cpu",
        onnx_file: Optional[str | Path] = None,
        dynamic_batch_size: bool = False,
        simplify: bool = False,
    ) -> None:
        # Warnings
        if half and device == "cpu":
            raise Exception("half only compatible with GPU export")
        # if half and dynamic_batch_size:
        #     raise Exception("half not compatible with dynamic")

        x = torch.randn(input_size)

        if half:
            x = x.half()
            self.model.half()
            if self.loss_fn is not None:
                self.loss_fn.half()

        if device != "cpu":
            torch.cuda.set_device(device)
            x = x.cuda()
            self.model.cuda()
            if self.loss_fn is not None:
                self.loss_fn.cuda()

        if onnx_file is None:
            f = self.cfg.model.weights.with_suffix(".onnx")
        else:
            f = self.run_dir / onnx_file

        input_name = "input"
        output_name = "output"
        if dynamic_batch_size:
            dynamic_axes = {
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"},
            }
        else:
            dynamic_axes = None

        torch.onnx.export(
            self.model,
            x,
            f,  # type: ignore
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,  # type: ignore
        )
        print(f"Saved the [bold magenta]onnx[/bold magenta] model: [cyan]{f}[/cyan]")

        print("Starting to simplify...")
        if simplify:
            model = onnx.load(f)  # type: ignore
            model, check = onnxsim.simplify(model)
            assert check, "assert check failed"
            onnx.save(model, f)  # type: ignore
        print(f"Saved the [bold magenta]simplified[/bold magenta] model: [cyan]{f}[/cyan]")

    # def _load_weights(self, module: nn.Module, model_or_loss: Literal["model", "loss"]):
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
    #     # 导出时需要weights名
    #     self.weights = weights
