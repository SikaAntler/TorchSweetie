from collections import OrderedDict
from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
import torch.nn as nn
from rich import print

from ..utils import KEY_B, KEY_E, LOSSES, MODELS, URL_B, URL_E, get_config, load_weights


class ClsExporter:
    def __init__(
        self, root_dir: str, cfg_file: str, run_dir: str, exp_dir: str, weights: str
    ) -> None:
        # Get the root path (project path)
        ROOT = Path(root_dir)

        # Get the absolute path of config file and load it
        self.cfg_file = ROOT / cfg_file
        self.cfg = get_config(ROOT, self.cfg_file)

        # Running directory, used to record results and models
        self.run_dir = ROOT / run_dir / self.cfg_file.stem / exp_dir
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
            self.model = nn.Sequential(
                OrderedDict(
                    {
                        "model": self.model,
                        "loss": self.loss_fn,
                    }
                )
            )
        else:
            self.loss_fn = None

    def export_onnx(
        self,
        input_size: tuple[int, int, int, int],
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
            f,  # pyright: ignore
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        print("Starting to simplify...")
        if simplify:
            model = onnx.load(f)
            model, check = onnxsim.simplify(model)
            assert check, "assert check failed"
            onnx.save(model, f)
        print(f"Saved the {KEY_B}simplified{KEY_E} model: {URL_B}{f}{URL_E}")
