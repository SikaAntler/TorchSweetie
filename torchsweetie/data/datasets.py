from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import TRANSFORMS


@dataclass
class ClsDataImage:
    image: ndarray  # (H, W, 3)
    label: int
    ori_size: tuple[int, int]  # (W, H)


@dataclass
class ClsDataTensor:
    image: Tensor
    label: int
    ori_size: tuple


@dataclass
class ClsDataPack:
    inputs: Tensor
    targets: Tensor
    ori_sizes: Tensor


class ClsDataset(Dataset):
    def __init__(self, csv_file: str, target_names: str, transforms: list[DictConfig]) -> None:
        super().__init__()

        dataset = pd.read_csv(csv_file, header=None)
        self.images = dataset[0].to_list()
        self.labels = dataset[1].to_list()

        self.target_names = target_names

        self.transforms = [TRANSFORMS.create(cfg) for cfg in transforms]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> ClsDataTensor:
        img_file, label = self.images[idx], self.labels[idx]

        image = np.array(Image.open(img_file))
        H, W = image.shape[:2]
        if len(image.shape) == 2:
            image = image.reshape(H, W, 1).repeat(2)

        data = ClsDataImage(image, label, (W, H))

        for t in self.transforms:
            data = t(data)

        return data  # pyright: ignore

    @staticmethod
    def collate_fn(batch_list: list[ClsDataTensor]) -> ClsDataPack:
        images = torch.stack([b.image for b in batch_list])
        labels = torch.tensor([b.label for b in batch_list], dtype=torch.long)
        ori_shapes = torch.tensor([b.ori_size for b in batch_list], dtype=torch.float)

        return ClsDataPack(images, labels, ori_shapes)
