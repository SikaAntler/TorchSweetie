import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from numpy import ndarray
from torch import Tensor


@dataclass
class BBox:
    auto: bool
    label: str
    score: float
    left: float
    top: float
    right: float
    bottom: float
    count: int

    @classmethod
    def from_dict(cls, bbox: dict) -> BBox:
        return cls(**bbox)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def width(self) -> float:
        return self.right - self.left + 1

    @property
    def height(self) -> float:
        return self.bottom - self.top + 1

    @property
    def lw(self) -> tuple[float, float]:
        return max(self.width, self.height), min(self.width, self.height)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def x(self) -> int:
        return int(round(self.left))

    @property
    def y(self) -> int:
        return int(round(self.top))

    @property
    def w(self) -> int:
        return int(round(self.width))

    @property
    def h(self) -> int:
        return int(round(self.height))

    @property
    def cx(self) -> int:
        return int(round(self.center_x))

    @property
    def cy(self) -> int:
        return int(round(self.center_y))


@dataclass
class Annotation:
    folder: str
    filename: str
    width: int
    height: int
    bboxes: list[BBox]

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        annotation = cls(
            data["folder"],
            data["filename"],
            data["width"],
            data["height"],
            [BBox.from_dict(bbox) for bbox in data["bboxes"]],
        )

        return annotation

    @classmethod
    def from_json(cls, ann_file: Path | str) -> Annotation:
        with open(ann_file, "r", encoding="utf-8") as fr:
            data = json.load(fr)

        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, ann_file: Path | str) -> None:
        with open(ann_file, "w", encoding="utf-8") as fw:
            json.dump(self.to_dict(), fw, ensure_ascii=False, indent=2)


@dataclass
class DetDataImage:
    image: ndarray  # (H, W, 3)
    ori_size: tuple[int, int]  # [W, H]
    bboxes: list[BBox]


@dataclass
class DetDataTensor:
    image: Tensor  # (3, H, W)
    ori_size: tuple[int, int]  # [W, H]
    cls_idxs: Tensor  # (N,)
    boxes: Tensor  # (N, 4), where 4 = BoxFormat


@dataclass
class DetDataPack:
    img_idxs: Tensor  # (N')
    images: Tensor  # (B, 3, H, W)
    ori_sizes: list[tuple[int, int]]  # (B, 2), where 2 = [W, H]
    cls_idxs: Tensor  # (N')
    boxes: Tensor  # (N', 4)


class DetResult(NamedTuple):
    boxes: Tensor
    scores: Tensor
    cls_idxs: Tensor
