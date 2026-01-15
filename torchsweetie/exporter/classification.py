from collections import OrderedDict
from pathlib import Path
from typing import Optional

import onnx
import onnxsim
import torch
from rich import print
from torch import Tensor, nn

from ..data import ClsDataPack
from ..utils import (
    DIR_B,
    DIR_E,
    KEY_B,
    KEY_E,
    LOSSES,
    MODELS,
    URL_B,
    URL_E,
    load_config,
    load_weights,
)


class ONNXExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_size: tuple[int, int, int, int]) -> None:
        super().__init__()

        self.model = model

        batch_size, _, H, W = input_size
        self.targets = torch.LongTensor([0] * batch_size)
        self.ori_sizes = torch.tensor([(H, W)] * batch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(ClsDataPack(x, self.targets, self.ori_sizes))


class ClsExporter:
    def __init__(
        self, cfg_file: Path, exp_dir: Path, weights: str, requires_loss: bool = True
    ) -> None:
        # Configuration
        self.cfg_file = cfg_file.absolute()
        self.cfg = load_config(self.cfg_file)
        print(f"Configuration file: {URL_B}{self.cfg_file}{URL_E}")

        # Running directory, used to record results and models
        self.exp_dir = exp_dir.absolute()
        assert self.exp_dir.exists()
        print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")

        # Model
        model_weights = self.exp_dir / weights
        self.cfg.model.weights = model_weights
        self.model = MODELS.create(self.cfg.model)

        # Loss Function
        if requires_loss:
            loss_fn: nn.Module = LOSSES.create(self.cfg.loss)
            if list(loss_fn.parameters()) != []:
                loss_weights = exp_dir / f"{model_weights.stem}-loss{model_weights.suffix}"
                load_weights(loss_fn, loss_weights)
                self.model = nn.Sequential(
                    OrderedDict(
                        {
                            "model": self.model,
                            "loss": loss_fn,
                        }
                    )
                )

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

        assert len(input_size) == 4
        x = torch.randn(input_size)

        model = ONNXExportWrapper(self.model, input_size)

        if half:
            x = x.half()
            model.half()

        if device != "cpu":
            x = x.cuda()
            model.cuda()

        if onnx_file is None:
            f = self.cfg.model.weights.with_suffix(".onnx")
        else:
            f = self.exp_dir / onnx_file

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
            model,
            x,
            f,  # pyright: ignore
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )
        print(f"Saved the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        print("Starting to simplify...")
        if simplify:
            onnx_model = onnx.load(f)
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
            onnx.save(onnx_model, f)
        print(f"Saved the {KEY_B}simplified{KEY_E} model: {URL_B}{f}{URL_E}")
