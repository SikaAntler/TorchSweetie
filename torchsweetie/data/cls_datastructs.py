from dataclasses import dataclass

from numpy import ndarray
from torch import Tensor


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
