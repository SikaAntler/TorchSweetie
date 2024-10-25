import pandas as pd
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset

from ..utils import TRANSFORMS


class ClsDataset(Dataset):
    def __init__(
        self, csv_file: str, target_names: str, transforms: list[DictConfig]
    ) -> None:
        super().__init__()

        dataset = pd.read_csv(csv_file, header=None)
        self.images = dataset[0].to_list()
        self.labels = dataset[1].to_list()

        self.target_names = target_names

        self.transforms = T.Compose([TRANSFORMS.create(cfg) for cfg in transforms])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, int]:
        image, label = self.images[idx], self.labels[idx]

        image = Image.open(image)
        image = self.transforms(image)

        return image, label  # pyright: ignore
