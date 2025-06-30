import random
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch import Tensor, nn

from ..data import ImageData
from ..utils import TRANSFORMS

__all__ = [
    "ColorGrading",
    "ColorSeperation",
    "ContourHighlight",
    "ConvertImageMode",
    "GaussianBlur",
    "GridRotation",
    "RandomColorJitter",
    "RandomColorJitterByRange",
    "RandomGaussianBlur",
    "RandomGrid",
    "RandomGridRotation",
    "RandomSharpen",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomTranspose",
    "RemainSize",
    "ResizePad",
    "RotateVertical",
    "Sharpen",
    "SplitRotate",
    "ToRGB",
    "ToTensor",
]


@TRANSFORMS.register()
class ColorGrading(nn.Module):
    def __init__(self, factor: list[float]) -> None:
        super().__init__()

        assert len(factor) == 3
        self.factor = factor

    def forward(self, image: Image.Image) -> Image.Image:
        array = np.array(image, dtype=np.float32)

        for i in range(3):
            array[:, :, i] *= self.factor[i]
        array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array)


@TRANSFORMS.register()
class ColorSeperation(nn.Module):
    def __init__(self, channel: Literal["R", "G", "B"]) -> None:
        super().__init__()

        self.channel = "RGB".index(channel)

    def forward(self, image: Image.Image) -> Image.Image:
        assert image.mode == "RGB"

        return image.getchannel(self.channel)


@TRANSFORMS.register()
class ContourHighlight(nn.Module):
    def __init__(
        self, threshold: int | Literal["otsu"], color: Sequence[int], thickness: int = 1
    ) -> None:
        super().__init__()

        if (not isinstance(threshold, int)) or (threshold == "otsu"):
            assert KeyError(f"threshold should be integer or `otsu`, not {threshold}")
        self.threshold = threshold

        if not isinstance(color, tuple):
            color = tuple(color)
        self.color = color

        self.thickness = thickness

    def forward(self, data: ImageData) -> ImageData:
        array = np.array(data["image"], dtype=np.uint8)  # (R, G, B)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        if isinstance(self.threshold, int):
            _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        elif self.threshold == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        array = cv2.drawContours(array, contours, -1, self.color, self.thickness)

        data["image"] = Image.fromarray(array)

        return data


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

        assert radius >= 2
        self.filter = ImageFilter.GaussianBlur(radius)

    def forward(self, image: Image.Image) -> Image.Image:
        return image.filter(self.filter)


@TRANSFORMS.register()
class GridRotation(nn.Module):
    ROTATION_MAP = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }

    def __init__(self, grid: int, rotation: Literal[90, 180, 270]) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        self.rotation = self.ROTATION_MAP[rotation]

    def forward(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        for j in range(self.grid):
            for i in range(self.grid):
                left = W // self.grid * i
                top = H // self.grid * j
                right = W if i == self.grid - 1 else W // self.grid * (i + 1)
                bottom = H if j == self.grid - 1 else H // self.grid * (j + 1)

                box = (left, top, right, bottom)
                img = image.crop(box)
                img = img.transpose(self.rotation)

                image.paste(img, box)

        return image


@TRANSFORMS.register()
class RandomColorJitter(nn.Module):
    def __init__(self, r: float, g: float, b: float) -> None:
        super().__init__()

        assert 0 <= r <= 1
        self.r = r

        assert 0 <= g <= 1
        self.g = g

        assert 0 <= b <= 1
        self.b = b

    def forward(self, data: ImageData) -> ImageData:
        image = data["image"]
        assert image.mode == "RGB"

        red, green, blue = image.split()
        if self.r != 0:
            red = self._jitter(red, self.r)
        if self.g != 0:
            green = self._jitter(green, self.g)
        if self.b != 0:
            blue = self._jitter(blue, self.b)

        data["image"] = Image.merge("RGB", [red, green, blue])

        return data

    @staticmethod
    def _jitter(img: Image.Image, p: float) -> Image.Image:
        prob = 1 + random.uniform(-p, p)
        array = np.array(img, dtype=np.float32) * prob
        array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array)


@TRANSFORMS.register()
class RandomColorJitterByRange(nn.Module):
    def __init__(self, dist_file: str, background: bool) -> None:
        super().__init__()

        self.color = np.arange(256)
        self.dist = np.load(dist_file)
        self.background = background

    def forward(self, data: ImageData) -> ImageData:
        image = data["image"]
        label = data["label"]
        assert image.mode == "RGB"

        array = np.array(image, dtype=np.uint8)

        for c in range(3):
            channel = array[:, :, c]
            prob = np.random.choice(self.color, 1, p=self.dist[label, c])[0]

            if self.background:
                prob /= channel.mean()
            else:
                _, mask = cv2.threshold(channel, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                prob /= channel[mask == 1].mean()
                prob = np.ones_like(channel, dtype=np.float32) * prob
                prob[mask == 0] = 1

            channel = channel.astype(np.float32) * prob
            array[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        data["image"] = Image.fromarray(array)

        return data


@TRANSFORMS.register()
class RandomGaussianBlur(nn.Module):
    def __init__(self, radius: int, prob: float) -> None:
        super().__init__()

        assert radius >= 2
        self.prob = prob

        assert 0.0 <= prob <= 1.0
        self.filter = ImageFilter.GaussianBlur(radius)

    def forward(self, image: Image.Image) -> Image.Image:
        if random.random() <= self.prob:
            image = image.filter(self.filter)

        return image


@TRANSFORMS.register()
class RandomGrid(nn.Module):
    def __init__(self, grid: int, prob: float) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        assert 0.0 <= prob <= 1.0
        self.prob = prob

    def forward(self, image: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return image

        W, H = image.size

        tile_list = []
        for j in range(self.grid):
            for i in range(self.grid):
                left = W // self.grid * i
                top = H // self.grid * j
                right = W // self.grid * (i + 1)
                bottom = H // self.grid * (j + 1)
                img = image.crop((left, top, right, bottom))
                tile_list.append(img)
        random.shuffle(tile_list)

        W, H = W // self.grid * self.grid, H // self.grid * self.grid
        new_img = Image.new(image.mode, (W, H))
        for j in range(self.grid):
            for i in range(self.grid):
                img = tile_list[j * self.grid + i]
                left = W // self.grid * i
                top = H // self.grid * j
                right = W // self.grid * (i + 1)
                bottom = H // self.grid * (j + 1)
                new_img.paste(img, (left, top, right, bottom))

        return new_img


@TRANSFORMS.register()
class RandomGridRotation(nn.Module):
    ROTATION_MAP = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }

    def __init__(self, grid: int, rotation: Literal[90, 180, 270], prob: float) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        self.rotation = self.ROTATION_MAP[rotation]

        assert 0.0 <= prob <= 1.0
        self.prob = prob

    def forward(self, image: Image.Image) -> Image.Image:
        W, H = image.size

        for j in range(self.grid):
            for i in range(self.grid):
                if random.random() > self.prob:
                    continue

                left = W // self.grid * i
                top = H // self.grid * j
                right = W if i == self.grid - 1 else W // self.grid * (i + 1)
                bottom = H if j == self.grid - 1 else H // self.grid * (j + 1)

                box = (left, top, right, bottom)
                img = image.crop(box)
                img = img.transpose(Image.Transpose.ROTATE_180)

                image.paste(img, box)

        return image


@TRANSFORMS.register()
class RandomSharpen(nn.Module):
    def __init__(self, prob: float) -> None:
        super().__init__()

        assert 0.0 <= prob <= 1.0
        self.prob = prob

        self.filter = ImageFilter.SHARPEN()

    def forward(self, image: Image.Image) -> Image.Image:
        if random.random() <= self.prob:
            image = image.filter(self.filter)

        return image


@TRANSFORMS.register()
class RandomSwapGrid(nn.Module):
    def __init__(self, grid: int, prob: float) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        assert 0.0 <= prob <= prob
        self.prob = prob

    def forward(self, image: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return image

        idx_1 = random.randint(0, self.grid**2 - 1)
        idx_2 = random.randint(0, self.grid**2 - 1)

        W, H = image.size
        box_1 = self._box(W, H, idx_1)
        box_2 = self._box(W, H, idx_2)

        tile_1 = image.crop(box_1)
        tile_2 = image.crop(box_2)

        image.paste(tile_1, box_2)
        image.paste(tile_2, box_1)

        return image

    def _box(self, W: int, H: int, idx: int) -> tuple[int, int, int, int]:
        i = idx // self.grid
        j = idx % self.grid

        left = W // self.grid * i
        top = H // self.grid * j
        right = W // self.grid * (i + 1)
        bottom = H // self.grid * (j + 1)

        return left, top, right, bottom


@TRANSFORMS.register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, data: ImageData) -> ImageData:
        data["image"] = super().forward(data["image"])  # pyright: ignore

        return data


@TRANSFORMS.register()
class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, data: ImageData) -> ImageData:
        data["image"] = super().forward(data["image"])  # pyright: ignore

        return data


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

    def forward(self, data: ImageData) -> ImageData:
        image = data["image"]
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

        data["image"] = new_img

        return data


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
class Sharpen(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.filter = ImageFilter.SHARPEN()

    def forward(self, image: Image.Image) -> Image.Image:
        return image.filter(self.filter)


@TRANSFORMS.register()
class SplitRotate(nn.Module):
    def __init__(self, mode: Literal["horizontal", "vertical"]) -> None:
        super().__init__()

        assert mode in ["horizontal", "vertical"], ValueError(
            "argument `lines` must be 1 f argument `mode` is `horizontal` or `vertical`"
        )
        self.mode: Literal["horizontal", "vertical"] = mode

    def forward(self, image: Image.Image) -> Image.Image:
        match self.mode:
            case "horizontal":
                return self._horizontal(image)
            case "vertical":
                return self._vertical(image)

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


@TRANSFORMS.register()
class ToRGB(nn.Module):
    def forward(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


@TRANSFORMS.register()
class ToTensor(T.ToTensor):
    def __call__(self, data: ImageData) -> Tensor:
        return super().__call__(data["image"])
