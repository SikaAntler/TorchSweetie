import random

import pandas as pd
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    def __init__(self, csv_file: str, target_names: str) -> None:
        super().__init__()

        self.images, self.labels = self._load_dataset(csv_file)
        self.target_names = target_names

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, int]:
        image, label = self.images[idx], self.labels[idx]

        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label

    def transform(self, image: Image.Image) -> FloatTensor:  # pyright: ignore
        raise NotImplementedError

    @staticmethod
    def gray2rgb(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")

        return image

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
        image: Image.Image,
        img_size: tuple[int, int],
        pad_value: tuple[int, int, int],
    ) -> Image.Image:
        width, height = image.size

        img_w, img_h = img_size
        new_img = Image.new(image.mode, (img_w, img_h), pad_value)  # pyright: ignore

        ratio_w = img_w / width
        ratio_h = img_h / height
        if ratio_w >= ratio_h:
            w = int(ratio_h * width)
            image = image.resize(size=(w, img_h))
            left = (img_w - w) // 2
            new_img.paste(image, (left, 0))
        elif ratio_w < ratio_h:
            h = int(ratio_w * height)
            image = image.resize(size=(img_w, h))
            top = (img_h - h) // 2
            new_img.paste(image, (0, top))

        return new_img

    def remain_size(
        self,
        image: Image.Image,
        img_size: tuple[int, int],
        pad_value: tuple[int, int, int],
    ) -> Image.Image:
        width, height = image.size

        img_w, img_h = img_size

        if width <= img_w and height <= img_h:
            new_img = Image.new(
                image.mode, (img_w, img_h), pad_value  # pyright: ignore
            )
            left = (img_w - width) // 2
            top = (img_h - height) // 2
            new_img.paste(image, (left, top))
        else:
            new_img = self.resize_longer(image, img_size, pad_value)

        return new_img

    @staticmethod
    def rotate_vertical(image: Image.Image) -> Image.Image:
        width, height = image.size

        if width > height:
            image = image.rotate(90, Image.Resampling.BILINEAR, True)

        return image

    def _load_dataset(self, csv_file: str) -> tuple[list[str], list[int]]:
        dataset = pd.read_csv(csv_file, header=None)
        images = dataset[0].to_list()
        labels = dataset[1].to_list()

        return images, labels
