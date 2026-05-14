from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich import print
from torch import nn
from tqdm import tqdm

from ..data import ClsDataPack, create_cls_dataloader
from ..utils import DIR_B, DIR_E, MODELS, URL_B, URL_E, load_config


class SimilarSamplesFinder:
    NCOLS = 100

    def __init__(
        self,
        cfg_file: Path,
        exp_dir: Path,
        weights: str,
        sup_dataset_file: Path,
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
        self.model.fc = nn.Identity()
        self.model.cuda()

        # Dataloader
        dataloader_cfg = self.cfg.train_dataloader
        dataloader_cfg.drop_last = False
        self.dataloader = create_cls_dataloader(dataloader_cfg)
        self.sup_dataloader_cfg = deepcopy(dataloader_cfg)
        self.sup_dataloader_cfg.dataset.csv_file = sup_dataset_file
        self.sup_dataloader = create_cls_dataloader(self.sup_dataloader_cfg)

        # Target names
        target_names = dataloader_cfg.dataset.target_names
        self.target_names = pd.read_csv(target_names, header=None)[0].to_list()

        # Store
        self.features: torch.Tensor | None = None
        self.sup_features: torch.Tensor | None = None

    @torch.inference_mode()
    def extract(self) -> None:
        if self.features is not None or self.sup_features is not None:
            tqdm.write(f"You may run the extraction twice, since the features are not None")

        self.model.eval()

        self.features = self._extract("Main", self.dataloader)

        self.sup_features = self._extract("Sup", self.sup_dataloader)

    def find_similar_samples(self, q: float) -> list[tuple[Path, str]]:
        if self.features is None:
            features_file = Path(self.cfg.train_dataloader.dataset.csv_file).with_suffix(".npy")
            features = np.load(features_file)
        else:
            features = self.features.cpu().numpy()

        if self.sup_features is None:
            sup_features_file = Path(self.sup_dataloader_cfg.dataset.csv_file).with_suffix(".npy")
            sup_features = np.load(sup_features_file)
        else:
            sup_features = self.sup_features.cpu().numpy()

        dataset: list[tuple[Path, str]] = []
        dataset_file = Path(self.cfg.train_dataloader.dataset.csv_file)
        for f, n in pd.read_csv(dataset_file, header=None).itertuples(False):
            dataset.append((Path(f), n))
        assert len(dataset) == features.shape[0], f"Main: {len(dataset)}, {features.shape}"

        sup_dataset: list[tuple[Path, str]] = []
        sup_dataset_file = Path(self.sup_dataloader_cfg.dataset.csv_file)
        for f, n in pd.read_csv(sup_dataset_file, header=None).itertuples(False):
            if n in self.target_names:
                sup_dataset.append((Path(f), n))
        assert len(sup_dataset) == sup_features.shape[0], (
            f"Target: {len(sup_dataset)}, {sup_features.shape}"
        )

        aug_dataset: list[tuple[Path, str]] = []

        for c in self.target_names:
            set_only = [(p, n, f) for (p, n), f in zip(dataset, features) if n == c]
            aug_dataset.extend((p, n) for p, n, _ in set_only)

            set_sup = [(p, n, f) for (p, n), f in zip(sup_dataset, sup_features) if n == c]
            if len(set_sup) == 0:
                continue

            feats_only = np.vstack([f for _, _, f in set_only])
            feats_sup = np.vstack([f for _, _, f in set_sup])

            inner_sim = self._topk_mean_similarity(feats_only @ feats_only.T, 5)
            threshold = np.quantile(inner_sim, q)

            simliarity = self._topk_mean_similarity(feats_sup @ feats_only.T, 5)
            indices = np.argwhere(simliarity >= threshold)
            for idx in indices:
                p, n, _ = set_sup[idx.item()]
                aug_dataset.append((p, n))

        return aug_dataset

    def dump(self) -> None:
        assert self.features is not None and self.sup_features is not None

        features = self.features.cpu().numpy()
        features_file = Path(self.cfg.train_dataloader.dataset.csv_file).with_suffix(".npy")
        np.save(features_file, features)

        sup_features = self.sup_features.cpu().numpy()
        sup_features_file = Path(self.sup_dataloader_cfg.dataset.csv_file).with_suffix(".npy")
        np.save(sup_features_file, sup_features)

    @staticmethod
    def _topk_mean_similarity(simliarity: np.ndarray, topk: int) -> np.ndarray:
        k = min(topk, simliarity.shape[1])
        indices = np.argpartition(simliarity, -k, 1)[:, -k:]

        simliarity = np.take_along_axis(simliarity, indices, 1)
        simliarity = simliarity.mean(1)

        return simliarity

    def _extract(self, desc: str, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        features = []

        pbar = tqdm(desc=desc, total=len(dataloader), ncols=self.NCOLS)
        for data in dataloader:
            data: ClsDataPack
            data.inputs = data.inputs.cuda()
            data.targets = data.targets.cuda()
            data.ori_sizes = data.ori_sizes.cuda()
            outputs = self.model(data)
            features.append(outputs)
            pbar.update()
        pbar.close()

        return F.normalize(torch.concat(features), dim=1)
