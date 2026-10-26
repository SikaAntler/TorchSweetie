import json
from datetime import datetime
from pathlib import Path
from typing import Literal, override
from zoneinfo import ZoneInfo

import onnx
import pandas as pd
import torch
import torch.nn.functional as F
from rich import print
from torch import Tensor, nn

from ..data import ClsDataPack, ClsModelOutput
from ..models.classifiers import ClsModel
from ..utils import (
    KEY_B,
    KEY_E,
    MODELS,
    URL_B,
    URL_E,
    load_weights_for_model,
)
from .runner import RunnerBase


class ONNXExportWrapper(nn.Module):
    def __init__(
        self, model: ClsModel, input_size: tuple[int, int, int, int], requires_softmax: bool
    ) -> None:
        super().__init__()

        self.model = model

        batch_size, _, H, W = input_size
        self.targets = torch.LongTensor([0] * batch_size)
        self.ori_sizes = torch.tensor([(H, W)] * batch_size)

        self.requires_softmax = requires_softmax

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        data = ClsDataPack(x, self.targets, self.ori_sizes)
        x: ClsModelOutput = self.model(data)

        if self.requires_softmax:
            return F.softmax(x.logits, 1)
        else:
            return x.logits


class ClsExporter(RunnerBase):
    SCOPE = "classification"

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        self.model: ClsModel

        classes_file = self.cfg.train_dataloader.dataset.classes_file
        classes = pd.read_csv(classes_file, header=None)[0].to_list()
        names = json.dumps(classes, ensure_ascii=False, indent=2)
        self.metadata = {
            "date": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "names": names,
            "task": "classification",
        }

    @override
    def build_model(self) -> ClsModel:
        if "scope" not in self.cfg.model:
            self.cfg.model.scope = self.SCOPE
            self.cfg.loss.scope = self.SCOPE

        self.cfg.model.pop("_weights_", None)
        model = MODELS.create(self.cfg.model)
        load_weights_for_model(model, str(self.weights), True)

        return model

    @override
    def build_dataloader(self) -> None:
        return

    @override
    @torch.inference_mode()
    def run(self) -> None:
        pass

    def export_onnx(
        self,
        input_size: tuple[int, int, int, int],
        requires_softmax: bool = False,
        half: bool = False,
        device: Literal["cpu"] | int = "cpu",
        simplify: bool = False,
    ) -> None:
        # Warnings
        if half and device == "cpu":
            raise ValueError("half only compatible with GPU export")

        assert len(input_size) == 4
        x = torch.randn(input_size)

        model = ONNXExportWrapper(self.model, input_size, requires_softmax).eval()

        if half:
            x = x.half()
            model.half()

        if device != "cpu":
            x = x.cuda()
            model.cuda()

        f = self.weights.with_suffix(".onnx")

        torch.onnx.export(
            model,
            x,  # ty: ignore
            f,
            input_names=["input"],
            output_names=["output"],
            external_data=False,
        )
        print(f"Exported the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        onnx_model = onnx.load(f)

        if simplify:
            import onnxslim

            print(f"Simplifying the ONNX model with onnxslim {onnxslim.__version__}...")
            onnx_model = onnxslim.slim(onnx_model)

        # Metadata
        metadata = {p.key: p.value for p in onnx_model.metadata_props}
        metadata.update(self.metadata)
        onnx.helper.set_model_props(onnx_model, metadata)
        onnx.save(onnx_model, f)
        print(f"Added the {KEY_B}metadata{KEY_E}: {URL_B}{f}{URL_E}")
