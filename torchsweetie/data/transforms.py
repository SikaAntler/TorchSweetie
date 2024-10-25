import random
from typing import Literal

import torchvision.transforms as T
from PIL import Image
from torch import nn

from ..utils import TRANSFORMS

__all__ = [
    "GrayToRGB",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomTranspose",
    "RemainSize",
    "ResizePad",
    "RotateVertical",
    "ToTensor",
]


@TRANSFORMS.register()
class GrayToRGB(nn.Module):
    def forward(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


@TRANSFORMS.register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    pass


@TRANSFORMS.register()
class RandomVerticalFlip(T.RandomVerticalFlip):
    pass


@TRANSFORMS.register()
class RandomTranspose(nn.Module):
    def forward(self, image: Image.Image) -> Image.Image:
        idx = random.randint(0, 6)
        transpose = Image.Transpose(idx)

        return image.transpose(transpose)


@TRANSFORMS.register()
class RemainSize(nn.Module):
    def __init__(self, img_size: int | list[int], pad_value: list[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        if not isinstance(pad_value, tuple):
            self.pad_value = tuple(pad_value)
        else:
            self.pad_value = pad_value

        self.resize = ResizePad(img_size, pad_value)

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        img_w, img_h = self.img_size

        if width <= img_w and height <= img_h:
            new_img = Image.new(
                image.mode, (img_w, img_h), self.pad_value  # pyright: ignore
            )
            left = (img_w - width) // 2
            top = (img_h - height) // 2
            new_img.paste(image, (left, top))
        else:
            new_img = self.resize(image)

        return new_img


@TRANSFORMS.register()
class ResizePad(nn.Module):
    def __init__(self, img_size: int | list[int], pad_value: list[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        if not isinstance(pad_value, tuple):
            self.pad_value = tuple(pad_value)
        else:
            self.pad_value = pad_value

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        img_w, img_h = self.img_size

        new_img = Image.new(
            image.mode, (img_w, img_h), self.pad_value  # pyright: ignore
        )

        ratio_w = img_w / width
        ratio_h = img_h / height
        if ratio_w >= ratio_h:
            w = int(ratio_h * width)
            image = image.resize((w, img_h))
            left = (img_w - w) // 2
            new_img.paste(image, (left, 0))
        else:
            h = int(ratio_w * height)
            image = image.resize((img_w, h))
            top = (img_h - h) // 2
            new_img.paste(image, (0, top))

        return new_img


@TRANSFORMS.register()
class RotateVertical(nn.Module):
    def __init__(
        self,
        direction: Literal["clockwise", "counterclockwise"] = "counterclockwise",
        resampling: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    ) -> None:
        super().__init__()

        match direction:
            case "clockwise":
                self.angle = -90
            case "counterclockwise":
                self.angle = 90

        match resampling:
            case "nearest":
                self.resampling = Image.Resampling.NEAREST
            case "bilinear":
                self.resampling = Image.Resampling.BILINEAR
            case "bicubic":
                self.resampling = Image.Resampling.BICUBIC

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width > height:
            image = image.rotate(self.angle, self.resampling, True)

        return image


@TRANSFORMS.register()
class ToTensor(T.ToTensor):
    pass
