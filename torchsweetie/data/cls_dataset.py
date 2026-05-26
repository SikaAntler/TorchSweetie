from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import TRANSFORMS
from . import ClsTransform


@dataclass
class ClsDataImage:
    image: ndarray  # (H, W, 3)
    label: int
    ori_size: tuple[int, int]  # (W, H)


@dataclass
class ClsDataTensor:
    image: Tensor  # (3, H, W)
    label: int
    ori_size: tuple[int, int]  # (W, H)


@dataclass
class ClsDataPack:
    inputs: Tensor
    targets: Tensor
    ori_sizes: Tensor


class ClsDataset(Dataset):
    SCOPE = "classification"

    def __init__(self, csv_file: str, target_names: str, transforms: list[DictConfig]) -> None:
        super().__init__()

        dataset = pd.read_csv(csv_file, header=None)
        self.target_names = pd.read_csv(target_names, header=None)[0].to_list()

        self.images, self.labels = [], []
        for img_file, name in dataset.itertuples(False):
            if name in self.target_names:
                self.images.append(img_file)
                self.labels.append(name)
            else:
                tqdm.write(f"HINT: {img_file} of {name} is ignored")

        self.transforms: list[ClsTransform] = []
        for cfg in transforms:
            if "scope" not in cfg:
                cfg.scope = self.SCOPE
            self.transforms.append(TRANSFORMS.create(cfg))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> ClsDataTensor:  # ty: ignore
        img_file = self.images[idx]
        label = self.target_names.index(self.labels[idx])

        image = np.array(Image.open(img_file))
        H, W = image.shape[:2]
        if len(image.shape) == 2:
            image = image.reshape(H, W, 1).repeat(2)

        data = ClsDataImage(image, label, (W, H))

        for t in self.transforms:
            data = t(data)

        return ClsDataTensor(self.to_tensor(data.image), data.label, data.ori_size)

    @staticmethod
    def collate_fn(batch_list: list[ClsDataTensor]) -> ClsDataPack:
        images = torch.stack([b.image for b in batch_list])
        labels = torch.tensor([b.label for b in batch_list], dtype=torch.long)
        ori_shapes = torch.tensor([b.ori_size for b in batch_list], dtype=torch.float)

        return ClsDataPack(images, labels, ori_shapes)

    @staticmethod
    def to_tensor(image: ndarray) -> Tensor:
        tensor = torch.from_numpy(image).type(torch.float32)
        tensor /= 255

        # (H, W, 3) -> (3, H, W)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.contiguous()

        return tensor
