import random
from typing import Any, Literal

import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch import nn

from ..utils import TRANSFORMS

__all__ = [
    "ConvertImageMode",
    "GaussianBlur",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomTranspose",
    "RemainSize",
    "ResizePad",
    "RotateVertical",
    "SplitRGB",
    "SplitRotate",
    "ToRGB",
    "ToTensor",
]


@TRANSFORMS.register()
class ConvertImageMode(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()

        self.mode = mode

    def forward(self, image: Image.Image) -> Image.Image:
        return image.convert(self.mode)


@TRANSFORMS.register()
class GaussianBlur(nn.Module):
    def __init__(self, radius: int) -> None:
        super().__init__()

        self.filter = ImageFilter.GaussianBlur(radius)

    def forward(self, image: Image.Image) -> Image.Image:
        return image.filter(self.filter)


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
    def __init__(self, img_size: int | Any, pad_value: int | Any) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        if isinstance(pad_value, int):
            self.pad_value = pad_value
        else:
            self.pad_value = tuple(pad_value)

        self.resize = ResizePad(img_size, pad_value)

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        img_w, img_h = self.img_size

        if width <= img_w and height <= img_h:
            new_img = Image.new(image.mode, (img_w, img_h), self.pad_value)  # pyright: ignore
            left = (img_w - width) // 2
            top = (img_h - height) // 2
            new_img.paste(image, (left, top))
        else:
            new_img = self.resize(image)

        return new_img


@TRANSFORMS.register()
class ResizePad(nn.Module):
    def __init__(self, img_size: int | Any, pad_value: int | Any) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        if isinstance(pad_value, int):
            self.pad_value = pad_value
        else:
            self.pad_value = tuple(pad_value)

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        img_w, img_h = self.img_size

        new_img = Image.new(image.mode, (img_w, img_h), self.pad_value)  # pyright: ignore

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
class SplitRGB(nn.Module):
    def __init__(self, channel: Literal["R", "G", "B"]) -> None:
        super().__init__()

        match channel:
            case "R":
                self.channel = 0
            case "G":
                self.channel = 1
            case "B":
                self.channel = 2

    def forward(self, image: Image.Image) -> Image.Image:
        assert image.mode == "RGB"

        return image.split()[self.channel]


@TRANSFORMS.register()
class SplitRotate(nn.Module):
    def __init__(self, mode: Literal["horizontal", "vertical", "grid"], lines: int = 1) -> None:
        super().__init__()

        if mode in ["horizontal", "vertical"]:
            if lines != 1:
                raise ValueError(
                    "argument `lines` must be 1 f argument `mode` is `horizontal` or `vertical`"
                )
        elif mode == "grid":
            if lines == 1:
                raise ValueError(
                    "argument `lines` must be bigger than 1 if argument `mode` is `grid`"
                )

        self.mode: Literal["horizontal", "vertical", "grid"] = mode
        self.lines = lines

    def forward(self, image: Image.Image) -> Image.Image:
        match self.mode:
            case "horizontal":
                return self._horizontal(image)
            case "vertical":
                return self._vertical(image)
            case "grid":
                return self._grid(image)

    def _horizontal(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        bbox_top = (0, 0, W, H // 2)
        img_top = image.crop(bbox_top)
        img_top = img_top.transpose(Image.Transpose.ROTATE_180)

        bbox_bottom = (0, H // 2, W, H)
        img_bottom = image.crop(bbox_bottom)
        img_bottom = img_bottom.transpose(Image.Transpose.ROTATE_180)

        new_img = Image.new(image.mode, image.size)
        new_img.paste(img_top, bbox_top)
        new_img.paste(img_bottom, bbox_bottom)

        return new_img

    def _vertical(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        bbox_left = (0, 0, W // 2, H)
        img_left = image.crop(bbox_left)
        img_left = img_left.transpose(Image.Transpose.ROTATE_180)

        bbox_right = (W // 2, 0, W, H)
        img_right = image.crop(bbox_right)
        img_right = img_right.transpose(Image.Transpose.ROTATE_180)

        new_img = Image.new(image.mode, image.size)
        new_img.paste(img_left, bbox_left)
        new_img.paste(img_right, bbox_right)

        return new_img

    def _grid(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        new_img = Image.new(image.mode, image.size)

        for j in range(self.lines):
            for i in range(self.lines):
                left = W // self.lines * i
                top = H // self.lines * j
                right = W if i == self.lines - 1 else W // self.lines * (i + 1)
                bottom = H if j == self.lines - 1 else H // self.lines * (j + 1)

                box = (left, top, right, bottom)
                img = image.crop(box)
                img = img.transpose(Image.Transpose.ROTATE_180)

                new_img.paste(img, box)

        # box_left_top = (0, 0, W // 2, H // 2)
        # img_left_top = image.crop(box_left_top)
        # img_left_top = img_left_top.transpose(Image.Transpose.ROTATE_180)
        #
        # box_right_top = (W // 2, 0, W, H // 2)
        # img_right_top = image.crop(box_right_top)
        # img_right_top = img_right_top.transpose(Image.Transpose.ROTATE_180)
        #
        # box_left_bottom = (0, H // 2, W // 2, H)
        # img_left_bottom = image.crop(box_left_bottom)
        # img_left_bottom = img_left_bottom.transpose(Image.Transpose.ROTATE_180)
        #
        # box_right_bottom = (W // 2, H // 2, W, H)
        # img_right_bottom = image.crop(box_right_bottom)
        # img_right_bottom = img_right_bottom.transpose(Image.Transpose.ROTATE_180)
        #
        # new_img = Image.new(image.mode, image.size)
        # new_img.paste(img_left_top, box_left_top)
        # new_img.paste(img_right_top, box_right_top)
        # new_img.paste(img_left_bottom, box_left_bottom)
        # new_img.paste(img_right_bottom, box_right_bottom)

        return new_img


@TRANSFORMS.register()
class ToRGB(nn.Module):
    def forward(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


@TRANSFORMS.register()
class ToTensor(T.ToTensor):
    pass
