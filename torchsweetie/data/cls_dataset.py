from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import TRANSFORMS
from .cls_datastructs import ClsDataImage, ClsDataPack, ClsDataTensor


class ClsTransform(nn.Module, ABC):
    dataset: list[tuple[str, str]]

    @abstractmethod
    def __call__(self, data: ClsDataImage) -> ClsDataImage: ...


class ClsDataset(Dataset):
    SCOPE = "classification"

    def __init__(self, dataset_file: str, classes_file: str, transforms: list[DictConfig]) -> None:
        super().__init__()

        self.dataset: list[tuple[str, str]] = []

        self.classes = pd.read_csv(classes_file, header=None)[0].to_list()

        dataset = pd.read_csv(dataset_file, header=None)
        for img_file, label in dataset.itertuples(False, None):
            if label in self.classes:
                self.dataset.append((img_file, label))
            else:
                tqdm.write(f"HINT: {img_file} of {label} is ignored")

        self.transforms: list[ClsTransform] = []
        for cfg in transforms:
            if "scope" not in cfg:
                cfg.scope = self.SCOPE
            transform = TRANSFORMS.create(cfg)
            assert isinstance(transform, ClsTransform)
            transform.dataset = self.dataset
            self.transforms.append(transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ClsDataTensor:  # ty: ignore
        img_file, label = self.dataset[idx]
        label = self.classes.index(label)

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
