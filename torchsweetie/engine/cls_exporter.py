import json
from datetime import datetime
from pathlib import Path
from typing import Literal, override

import onnx
import pandas as pd
import torch
from rich import print
from torch import Tensor, nn

from ..data import ClsDataPack
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
    def __init__(self, model: ClsModel, input_size: tuple[int, int, int, int]) -> None:
        super().__init__()

        self.model = model

        batch_size, _, H, W = input_size
        self.targets = torch.LongTensor([0] * batch_size)
        self.ori_sizes = torch.tensor([(H, W)] * batch_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.model.requires_data_pack:
            data = ClsDataPack(x, self.targets, self.ori_sizes)
            x = self.model(data)
        else:
            x = self.model(x)

        return x


class ClsExporter(RunnerBase):
    SCOPE = "classification"

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        self.model: ClsModel

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
        half: bool = False,
        device: Literal["cpu"] | int = "cpu",
        onnx_file: Path | str | None = None,
        dynamic_batch_size: bool = False,
        simplify: bool = False,
    ) -> None:
        # Warnings
        if half and device == "cpu":
            raise ValueError("half only compatible with GPU export")
        # if half and dynamic_batch_size:
        #     raise Exception("half not compatible with dynamic")

        assert len(input_size) == 4
        x = torch.randn(input_size)

        model = ONNXExportWrapper(self.model, input_size)
        model = model.eval()

        if half:
            x = x.half()
            model.half()

        if device != "cpu":
            x = x.cuda()
            model.cuda()

        if onnx_file is None:
            f = self.weights.with_suffix(".onnx")
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
            x,  # ty: ignore
            f,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            external_data=False,
        )
        print(f"Saved the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        # Metadata
        onnx_model = onnx.load(f)
        classes_file = self.cfg.train_dataloader.dataset.classes_file
        classes = pd.read_csv(classes_file, header=None)[0].to_list()
        names = json.dumps(classes, ensure_ascii=False, indent=2)
        self.metadata = {"date": datetime.now().astimezone().isoformat(), "names": names}
        for k, v in self.metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(onnx_model, f)
        print(f"Added the {KEY_B}metadata{KEY_E}: {URL_B}{f}{URL_E}")

        if simplify:
            import onnxslim

            print(f"Starting to slim with onnxslim {onnxslim.__version__}...")
            onnx_model = onnx.load(f)
            onnx_model = onnxslim.slim(onnx_model)
            onnx.save(onnx_model, f)
            print(f"Saved the {KEY_B}simplified{KEY_E} model: {URL_B}{f}{URL_E}")
