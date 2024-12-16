from pathlib import Path

import torch
from rich import print
from torch import Tensor
from tqdm import tqdm

from ..data import create_cls_dataloader
from ..utils import DIR_B, DIR_E, MODELS, get_config


class RetrievalExporter:
    NCOLS = 100

    def __init__(self, cfg_file: str, run_dir: str, exp_dir: str, weights: str) -> None:
        # Get the root path (project path)
        self.root_dir = Path.cwd()

        # Get the absolute path of config file and load it
        self.cfg_file = self.root_dir / cfg_file
        self.cfg = get_config(self.root_dir, self.cfg_file)

        # Running directory, used to record results and models
        self.exp_dir = self.root_dir / run_dir / self.cfg_file.stem / exp_dir
        assert self.exp_dir.exists()
        print(f"Experimental directory: {DIR_B}{self.exp_dir}{DIR_E}")

        # Model
        model_weights = self.exp_dir / weights
        self.cfg.model.weights = model_weights
        print(self.cfg.model.weights)
        self.model = MODELS.create(self.cfg.model)
        self.model.cuda()

        # P.S. Loss function should be dropped

        # Dataloader
        dataloader_cfg = self.cfg.retrieval_dataloader
        self.dataloader = create_cls_dataloader(dataloader_cfg)

        # Store the embeddings output by model
        self.embeddings: Tensor
        self.labels: Tensor

    @torch.no_grad()
    def export(self) -> None:
        pbar = tqdm(desc=f"Exporting", total=len(self.dataloader), ncols=self.NCOLS)

        if hasattr(self, "embeddings"):
            tqdm.write(f"You may run the test twice, since embeddings is not empty.")

        self.model.eval()

        embeddings_list = []
        labels_list = []

        for images, labels in self.dataloader:
            labels_list.append(labels)
            images, labels = images.cuda(), labels.cuda()
            # outputs: (R, N), where R represents the number of Retrieval images
            outputs = self.model(images)
            embeddings_list.append(outputs)

            pbar.update()

        pbar.close()

        self.embeddings = torch.concat(embeddings_list)
        self.labels = torch.concat(labels_list)
