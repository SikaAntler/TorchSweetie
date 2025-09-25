from dataclasses import dataclass
from typing import TypedDict

import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..utils import TRANSFORMS


# TODO 改成dataclass
class ClsDataImage(TypedDict):
    image: Image.Image
    label: int
    ori_shape: tuple[int, int]


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

        self.transforms = T.Compose([TRANSFORMS.create(cfg) for cfg in transforms])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> ClsDataTensor:
        image, label = self.images[idx], self.labels[idx]

        image = Image.open(image)
        data = self.transforms({"image": image, "label": label, "ori_shape": image.size})

        return data  # pyright: ignore

    @staticmethod
    def collate_fn(batch_list: list[ClsDataTensor]) -> ClsDataPack:
        images = torch.stack([b.image for b in batch_list])
        labels = torch.tensor([b.label for b in batch_list], dtype=torch.long)
        ori_shapes = torch.tensor([b.ori_size for b in batch_list], dtype=torch.float)

        return ClsDataPack(images, labels, ori_shapes)
