from dataclasses import dataclass
from typing import NamedTuple

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


class ClsModelOutput(NamedTuple):
    """分类模型输出结构

    Args:
        logits: (B, C) 模型的最终输出，C为类别数量
        embeddings: (B, N) 模型骨干网络的原始输出，F为特征通道数

    Notes:
        在本项目中，C可简单认为是检测头（通常为fc层）之后的，N是检测头之前的输出
    """

    logits: Tensor
    embeddings: Tensor
