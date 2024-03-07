import random

import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, cfg: DictConfig, task: str) -> None:
        super().__init__()

        self.cfg = cfg
        self.task = task

        self.images, self.labels = self._load_dataset()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, LongTensor]:
        image, label = self.images[idx], self.labels[idx]

        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.task == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_val(image)

        return image, label

    def _load_dataset(self) -> tuple[list[str], list[int]]:
        dataset = pd.read_csv(self.cfg[self.task], header=None)
        images = dataset[0].to_list()
        labels = dataset[1].to_list()

        return images, labels

    @staticmethod
    def random_resize(image: Image.Image, scale: tuple[float, float]) -> Image.Image:
        ratio = random.uniform(scale[0], scale[1])

        width, height = image.size
        width, height = int(width * ratio), int(height * ratio)
        image = image.resize(size=(width, height))

        return image

    @staticmethod
    def random_transpose(image: Image.Image) -> Image.Image:
        idx = random.randint(0, 6)

        return image.transpose(Image.Transpose(idx))

    @staticmethod
    def resize_longer(
        image: Image.Image, img_size: int, pad_value: tuple[int, int, int]
    ) -> Image.Image:
        width, height = image.size

        new_img = Image.new(image.mode, (img_size, img_size), pad_value)

        ratio = max(width, height) / img_size
        if width >= height:
            image = image.resize(size=(img_size, int(height / ratio)))
            top = (img_size - image.height) // 2
            new_img.paste(image, (0, top))
        elif width < height:
            image = image.resize(size=(int(width / ratio), img_size))
            left = (img_size - image.width) // 2
            new_img.paste(image, (left, 0))

        return new_img

    def transform_train(self, image: Image.Image):
        raise NotImplementedError

    def transform_val(self, image: Image.Image):
        raise NotImplementedError
