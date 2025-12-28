import json
from datetime import datetime
from pathlib import Path
from typing import override

import onnx
import pandas as pd
import torch
from rich import print
from torch import nn

from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E, load_weights_for_model
from .runner import RunnerBase


class DetExporter(RunnerBase):
    SCOPE = "detection"

    def __init__(self, cfg_file: Path, exp_dir: Path, weights: str) -> None:
        super().__init__(cfg_file, exp_dir, weights)

        self.half = self.cfg.train.get("mixed_precision", "no") == "fp16"

    @override
    def build_model(self) -> nn.Module:
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
        onnx_file: str | Path | None = None,
        simplify: bool = False,
    ) -> None:
        assert len(input_size) == 4
        x = torch.randn(input_size)

        self.model.eval()

        setattr(self.model, "export", True)

        if self.half:
            x = x.half()
            self.model.half()

        x = x.cuda()

        if onnx_file is None:
            f = self.weights.with_suffix(".onnx")
        else:
            f = self.exp_dir / onnx_file

        torch.onnx.export(
            self.model,
            x,  # ty: ignore
            f,
            input_names=["input"],
            output_names=["boxes", "scores", "cls_idxs"],
            external_data=False,
        )
        print(f"Saved the {KEY_B}onnx{KEY_E} model: {URL_B}{f}{URL_E}")

        # Metadata
        onnx_model = onnx.load(f)
        classes_file = self.cfg.train_dataloader.dataset.classes_file
        classes = pd.read_csv(classes_file, header=None)[0].to_list()
        names = json.dumps(classes, ensure_ascii=False, indent=2)
        self.metadata = {"date": datetime.now().isoformat(), "names": names}
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
