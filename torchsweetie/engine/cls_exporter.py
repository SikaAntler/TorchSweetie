import json
from datetime import datetime
from pathlib import Path
from typing import Literal, override
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from rich import print
from torch import Tensor, nn

from ..data import ClsDataImage, ClsDataPack, ClsModelOutput
from ..data.cls_transforms import ResizePad
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
        *,
        requires_softmax: bool = False,
        device: Literal["cpu"] | int = "cpu",
        simplify: bool = False,
        half: bool = False,
        calibration_batches: int = 8,
    ) -> None:
        if half:
            from modelopt.onnx.autocast import convert_to_mixed_precision

        x = torch.randn(input_size)

        model = ONNXExportWrapper(self.model, input_size, requires_softmax).eval()

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
            dynamo=True,
            external_data=False,
        )
        print(f"Exported the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        onnx_model = onnx.load(f)

        if simplify:
            import onnxslim

            print(f"Simplifying the ONNX model with onnxslim {onnxslim.__version__}...")
            onnx_model = onnxslim.slim(onnx_model)

        if half:
            import modelopt

            print(
                f"Converting ONNX to FP16 mixed precision with modelopt {modelopt.__version__}..."
            )

            if calibration_batches != 0:
                calibration_data = self._create_calibration_data(
                    input_size, calibration_batches, self.exp_dir / "calibration.npz"
                )
            else:
                calibration_data = None

            onnx_model = convert_to_mixed_precision(
                str(f), keep_io_types=True, calibration_data=calibration_data, providers=["cpu"]
            )

            f = f.with_stem(f"{f.stem}-fp16")
            onnx.save(onnx_model, f)
            print(f"Saved the {KEY_B}FP16{KEY_E} mixed precision model : {URL_B}{f}{URL_E}")

        # Metadata
        metadata = {p.key: p.value for p in onnx_model.metadata_props}
        metadata.update(self.metadata)
        onnx.helper.set_model_props(onnx_model, metadata)
        onnx.save(onnx_model, f)
        print(f"Added the {KEY_B}metadata{KEY_E}: {URL_B}{f}{URL_E}")

    def _create_calibration_data(
        self, input_size: tuple[int, int, int, int], calibration_batches: int, output_file: Path
    ) -> str:

        batch_size, channels, height, width = input_size
        assert channels == 3, "Training-image calibration requires RGB input (C=3)"

        dataset_file = self.cfg.train_dataloader.dataset.dataset_file
        image_paths = pd.read_csv(dataset_file, header=None)[0].to_list()
        assert len(image_paths) // batch_size >= calibration_batches

        num_images = calibration_batches * batch_size
        indices = np.random.default_rng(0).choice(len(image_paths), num_images, replace=False)
        images = np.empty((num_images, channels, height, width), dtype=np.float32)

        for transform_cfg in self.cfg.train_dataloader.dataset.transforms:
            if transform_cfg.name == "ResizePad":
                resize_pad_cfg = transform_cfg
        resize_pad = ResizePad(resize_pad_cfg.img_size, resize_pad_cfg.pad_value)

        for i, idx in enumerate(indices):
            image = np.array(Image.open(image_paths[idx]))
            assert image is not None
            data = ClsDataImage(image, 0, (0, 0))
            data: ClsDataImage = resize_pad(data)
            images[i] = data.image.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)

        images /= 255.0

        np.savez(output_file, input=images)

        return str(output_file)
