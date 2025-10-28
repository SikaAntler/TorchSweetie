import random
from typing import Literal, Sequence

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch import nn

from ..data import ClsDataImage, ClsDataTensor
from ..utils import TRANSFORMS

__all__ = [
    "ColorBroken",
    "ColorGrading",
    "ColorSeperation",
    "ContourHighlight",
    "ConvertImageMode",
    "GaussianBlur",
    "GridRotation",
    "RandomColorJitter",
    "RandomColorJitterByRange",
    "RandomGaussianBlur",
    "RandomGaussianBlurClasswise",
    "RandomGaussianBlurByClarity",
    "RandomGrid",
    "RandomGridRotation",
    "RandomSharpen",
    "RandomSwapGrid",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomTranspose",
    "Resize",
    "ResizeCrop",
    "ResizePad",
    "ResizeRemain",
    "Sharpen",
    "SplitRotate",
    "StandarizeSize",
    "ToRGB",
    "ToTensor",
    "VerticalRotate",
]


@TRANSFORMS.register()
class ColorBroken(nn.Module):
    def __init__(self, channel: Literal["R", "G", "B"], fill: int) -> None:
        super().__init__()

        self.channel = "RGB".index(channel)
        self.fill = fill

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        assert data.image.mode == "RGB"

        array = np.array(data.image, np.uint8)
        array[:, :, self.channel] = self.fill
        data.image = Image.fromarray(array)

        return data


@TRANSFORMS.register()
class ColorGrading(nn.Module):
    def __init__(self, factor: list[float]) -> None:
        super().__init__()

        assert len(factor) == 3
        self.factor = factor

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        assert data.image.mode == "RGB"

        array = np.array(data.image, dtype=np.float32)
        for i in range(3):
            array[:, :, i] *= self.factor[i]
        array = np.clip(array, 0, 255).astype(np.uint8)

        data.image = Image.fromarray(array)

        return data


@TRANSFORMS.register()
class ColorSeperation(nn.Module):
    def __init__(self, channel: Literal["R", "G", "B"]) -> None:
        super().__init__()

        self.channel = "RGB".index(channel)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        assert data.image.mode == "RGB"

        data.image = data.image.getchannel(self.channel)

        return data


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

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        array = np.array(data.image, dtype=np.uint8)  # (R, G, B)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        if isinstance(self.threshold, int):
            _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        elif self.threshold == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        array = cv2.drawContours(array, contours, -1, self.color, self.thickness)

        data.image = Image.fromarray(array)

        return data


@TRANSFORMS.register()
class ConvertImageMode(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()

        self.mode = mode

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = data.image.convert(self.mode)

        return data


@TRANSFORMS.register()
class GaussianBlur(nn.Module):
    def __init__(self, radius: int) -> None:
        super().__init__()

        assert radius >= 2
        self.filter = ImageFilter.GaussianBlur(radius)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = data.image.filter(self.filter)

        return data


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

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        W, H = data.image.size

        for j in range(self.grid):
            for i in range(self.grid):
                left = W // self.grid * i
                top = H // self.grid * j
                right = W if i == self.grid - 1 else W // self.grid * (i + 1)
                bottom = H if j == self.grid - 1 else H // self.grid * (j + 1)

                box = (left, top, right, bottom)
                img = data.image.crop(box)
                img = img.transpose(self.rotation)

                data.image.paste(img, box)

        return data


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

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        assert data.image.mode == "RGB"

        red, green, blue = data.image.split()
        red = self._jitter(red, self.r)
        green = self._jitter(green, self.g)
        blue = self._jitter(blue, self.b)

        data.image = Image.merge("RGB", [red, green, blue])

        return data

    @staticmethod
    def _jitter(img: Image.Image, p: float) -> Image.Image:
        if p == 0:
            return img

        prob = 1 + random.uniform(-p, p)
        array = np.array(img, dtype=np.float32) * prob
        array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array)


@TRANSFORMS.register()
class RandomColorJitterByRange(nn.Module):
    def __init__(self, dist_file: str, background: bool, channels: str = "RGB") -> None:
        super().__init__()

        self.color = np.arange(256)
        self.dist = np.load(dist_file)
        self.background = background

        assert set(channels).issubset("RGB")
        self.channels = channels

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        assert data.image.mode == "RGB"

        array = np.array(data.image, dtype=np.uint8)

        for i, c in enumerate("RGB"):
            if c not in self.channels:
                continue

            channel = array[:, :, i]
            prob = np.random.choice(self.color, 1, p=self.dist[data.label, i])[0]

            if self.background:
                prob /= channel.mean()
            else:
                _, mask = cv2.threshold(channel, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                prob /= channel[mask == 1].mean()
                prob = np.ones_like(channel, dtype=np.float32) * prob
                prob[mask == 0] = 1

            channel = channel.astype(np.float32) * prob
            array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)

        data.image = Image.fromarray(array)

        return data


@TRANSFORMS.register()
class RandomGaussianBlur(nn.Module):
    def __init__(self, radius: int, prob: float) -> None:
        super().__init__()

        assert radius >= 2
        self.filter = ImageFilter.GaussianBlur(radius)

        assert 0.0 <= prob <= 1.0
        self.prob = prob

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        if random.random() <= self.prob:
            data.image = data.image.filter(self.filter)

        return data


@TRANSFORMS.register()
class RandomGaussianBlurClasswise(nn.Module):
    def __init__(self, csv_file: str) -> None:
        super().__init__()

        thresh = pd.read_csv(csv_file)

        radius = []
        prob = []
        for _, r, p in thresh.itertuples():
            radius.append(r)
            prob.append(p)

        self.radius = tuple(radius)
        self.prob = tuple(prob)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        if random.random() <= self.prob[data.label]:
            data.image = data.image.filter(ImageFilter.GaussianBlur(self.radius[data.label]))

        return data


@TRANSFORMS.register()
class RandomGaussianBlurByClarity(nn.Module):
    def __init__(self, radiuses: list[int], csv_file: str) -> None:
        super().__init__()

        self.radiuses = sorted(radiuses, reverse=True)

        clarity = pd.read_csv(csv_file)
        self.clarity_min = clarity["min"].to_numpy()
        self.clarity_max = clarity["max"].to_numpy()

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        clarity_min = self.clarity_min[data.label]
        clarity_max = self.clarity_max[data.label]

        clarity = self._tenengrad(data.image)
        prob = (clarity - clarity_min) / (clarity_max - clarity_min)
        if random.random() > prob:
            return data

        radius = random.choice(self.radiuses)

        while radius in self.radiuses:
            image = data.image.filter(ImageFilter.GaussianBlur(radius))
            clarity = self._tenengrad(image)
            if clarity >= clarity_min:
                data.image = image
                break
            else:
                radius -= 2

        return data

    @staticmethod
    def _tenengrad(image: Image.Image) -> float:
        gray = np.array(image.convert("L"))
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.sqrt(grad_x**2 + grad_y**2)
        clarity = np.mean(np.square(tenengrad[tenengrad != 0]))

        return clarity.item()


@TRANSFORMS.register()
class RandomGrid(nn.Module):
    def __init__(self, grid: int, prob: float) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        assert 0.0 <= prob <= 1.0
        self.prob = prob

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        if random.random() > self.prob:
            return data

        W, H = data.image.size

        tile_list = []
        for j in range(self.grid):
            for i in range(self.grid):
                left = W // self.grid * i
                top = H // self.grid * j
                right = W // self.grid * (i + 1)
                bottom = H // self.grid * (j + 1)
                img = data.image.crop((left, top, right, bottom))
                tile_list.append(img)
        random.shuffle(tile_list)

        W, H = W // self.grid * self.grid, H // self.grid * self.grid
        data.image = Image.new(data.image.mode, (W, H))
        for j in range(self.grid):
            for i in range(self.grid):
                img = tile_list[j * self.grid + i]
                left = W // self.grid * i
                top = H // self.grid * j
                right = W // self.grid * (i + 1)
                bottom = H // self.grid * (j + 1)
                data.image.paste(img, (left, top, right, bottom))

        return data


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

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        W, H = data.image.size

        for j in range(self.grid):
            for i in range(self.grid):
                if random.random() > self.prob:
                    continue

                left = W // self.grid * i
                top = H // self.grid * j
                right = W if i == self.grid - 1 else W // self.grid * (i + 1)
                bottom = H if j == self.grid - 1 else H // self.grid * (j + 1)

                box = (left, top, right, bottom)
                img = data.image.crop(box)
                img = img.transpose(Image.Transpose.ROTATE_180)

                data.image.paste(img, box)

        return data


@TRANSFORMS.register()
class RandomSharpen(nn.Module):
    def __init__(self, prob: float) -> None:
        super().__init__()

        assert 0.0 <= prob <= 1.0
        self.prob = prob

        self.filter = ImageFilter.SHARPEN()

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        if random.random() <= self.prob:
            data.image = data.image.filter(self.filter)

        return data


@TRANSFORMS.register()
class RandomSwapGrid(nn.Module):
    def __init__(self, grid: int, prob: float) -> None:
        super().__init__()

        assert grid >= 2
        self.grid = grid

        assert 0.0 <= prob <= prob
        self.prob = prob

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        if random.random() > self.prob:
            return data

        idx_1 = random.randint(0, self.grid**2 - 1)
        idx_2 = random.randint(0, self.grid**2 - 1)

        W, H = data.image.size
        box_1 = self._box(W, H, idx_1)
        box_2 = self._box(W, H, idx_2)

        tile_1 = data.image.crop(box_1)
        tile_2 = data.image.crop(box_2)

        data.image.paste(tile_1, box_2)
        data.image.paste(tile_2, box_1)

        return data

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
    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = super().forward(data.image)  # pyright: ignore

        return data


@TRANSFORMS.register()
class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = super().forward(data.image)  # pyright: ignore

        return data


@TRANSFORMS.register()
class RandomTranspose(nn.Module):
    def forward(self, data: ClsDataImage) -> ClsDataImage:
        idx = random.randint(0, 6)
        transpose = Image.Transpose(idx)

        data.image = data.image.transpose(transpose)

        return data


@TRANSFORMS.register()
class Resize(nn.Module):
    def __init__(self, img_size: int | Sequence[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)
            assert len(self.img_size) == 2

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = data.image.resize(self.img_size)

        return data


@TRANSFORMS.register()
class ResizeCrop(nn.Module):
    def __init__(self, img_size: int | Sequence[int], pad_value: int | Sequence[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)
            assert len(self.img_size) == 2

        if isinstance(pad_value, int):
            self.pad_value = pad_value
        else:
            self.pad_value = tuple(pad_value)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        W, H = data.image.size
        img_w, img_h = self.img_size

        ratio_w = img_w / W
        ratio_h = img_h / H
        if ratio_w >= ratio_h:
            h = int(ratio_w * H)
            image = data.image.resize((img_w, h))
            top = (h - img_h) // 2
            bottom = top + img_h
            data.image = image.crop((0, top, img_w, bottom))
        else:
            w = int(ratio_h * W)
            image = data.image.resize((w, img_h))
            left = (w - img_w) // 2
            right = left + img_w
            data.image = image.crop((left, 0, right, img_h))

        return data


@TRANSFORMS.register()
class ResizePad(nn.Module):
    def __init__(self, img_size: int | Sequence[int], pad_value: int | Sequence[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)
            assert len(self.img_size) == 2

        if isinstance(pad_value, int):
            self.pad_value = pad_value
        else:
            self.pad_value = tuple(pad_value)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        W, H = data.image.size

        img_w, img_h = self.img_size
        new_img = Image.new(data.image.mode, self.img_size, self.pad_value)  # pyright: ignore

        ratio_w = img_w / W
        ratio_h = img_h / H
        if ratio_w >= ratio_h:
            w = int(ratio_h * W)
            image = data.image.resize((w, img_h))
            left = (img_w - w) // 2
            new_img.paste(image, (left, 0))
        else:
            h = int(ratio_w * H)
            image = data.image.resize((img_w, h))
            top = (img_h - h) // 2
            new_img.paste(image, (0, top))

        data.image = new_img

        return data


@TRANSFORMS.register()
class ResizeRemain(nn.Module):
    def __init__(self, img_size: int | Sequence[int], pad_value: int | Sequence[int]) -> None:
        super().__init__()

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = tuple(img_size)
            assert len(self.img_size) == 2

        if isinstance(pad_value, int):
            self.pad_value = pad_value
        else:
            self.pad_value = tuple(pad_value)

        self.resize = ResizePad(img_size, pad_value)

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        width, height = data.image.size
        img_w, img_h = self.img_size

        if width <= img_w and height <= img_h:
            new_img = Image.new(data.image.mode, self.img_size, self.pad_value)  # pyright: ignore
            left = (img_w - width) // 2
            top = (img_h - height) // 2
            new_img.paste(data.image, (left, top))
            data.image = new_img
        else:
            data = self.resize(data)

        return data


@TRANSFORMS.register()
class Sharpen(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.filter = ImageFilter.SHARPEN()

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = data.image.filter(self.filter)

        return data


@TRANSFORMS.register()
class SplitRotate(nn.Module):
    def __init__(self, mode: Literal["horizontal", "vertical"]) -> None:
        super().__init__()

        assert mode in ["horizontal", "vertical"], ValueError(
            "argument `lines` must be 1 f argument `mode` is `horizontal` or `vertical`"
        )
        self.mode: Literal["horizontal", "vertical"] = mode

    def forward(self, data: ClsDataImage) -> ClsDataImage:
        match self.mode:
            case "horizontal":
                data.image = self._horizontal(data.image)
            case "vertical":
                data.image = self._vertical(data.image)

        return data

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
class StandarizeSize(nn.Module):
    def __init__(self, w_mean: float, w_std: float, h_mean: float, h_std: float) -> None:
        super().__init__()

        self.w_mean = w_mean
        self.w_std = w_std
        self.h_mean = h_mean
        self.h_std = h_std

    def forward(self, data: ClsDataTensor) -> ClsDataTensor:
        w = (data.ori_size[0] - self.w_mean) / self.w_std
        h = (data.ori_size[1] - self.h_mean) / self.h_std
        data.ori_size = (w, h)

        return data


@TRANSFORMS.register()
class ToRGB(nn.Module):
    def forward(self, data: ClsDataImage) -> ClsDataImage:
        data.image = data.image.convert("RGB")

        return data


@TRANSFORMS.register()
class ToTensor(T.ToTensor):
    def __call__(self, data: ClsDataImage) -> ClsDataTensor:
        image = super().__call__(data.image)

        return ClsDataTensor(image, data.label, data.ori_shape)


@TRANSFORMS.register()
class VerticalRotate(nn.Module):
    def forward(self, data: ClsDataImage) -> ClsDataImage:
        W, H = data.image.size
        assert data.ori_shape == (W, H)

        if W > H:
            data.image = data.image.transpose(Image.Transpose.ROTATE_90)
            data.ori_shape = (H, W)

        return data
