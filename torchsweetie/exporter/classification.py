from pathlib import Path
from typing import Literal, Optional

import onnx
import onnxsim
import torch
from rich import print
from torch.nn import Module

from torchsweetie.utils import MODELS, get_config, load_weights


class ClsExporter:
    def __init__(self, cfg_file: Path | str, times: int, best_or_last: str) -> None:
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

    def export_onnx(
        self,
        input_size: tuple,
        f: Optional[str] = None,
        half: bool = False,
        dynamic_batch_size: bool = False,
        simplify: bool = False,
    ) -> None:
        # Warnings
        if half and dynamic_batch_size:
            raise "half only compatible with GPU export"

        x = torch.randn(input_size).cuda()

        if f is None:
            f = self.weights.with_suffix(".onnx")

        if half:
            self.model.half()
            x = x.half()

        input_name = "input"
        output_name = "output"
        if dynamic_batch_size:
            dynamic_axes = {
                input_name: {"0", "batch_size"},
                output_name: {"0", "batch_size"},
            }
        else:
            dynamic_axes = None

        torch.onnx.export(
            self.model,
            x,
            f,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            model = onnx.load(f)
            model, check = onnxsim.simplify(model)
            assert check, "assert check failed"
            onnx.save(model, f)

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
            print(f"{len(weights)} weights have been found, {weights[0]} has been used")
        weights = weights[0]
        load_weights(module, weights)
        module.cuda()
        # 导出时需要weights名
        self.weights = weights
