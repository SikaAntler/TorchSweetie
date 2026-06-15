"""The set of some useful losses.

In order to uniform naming convention and reduce confusions, there are some rules you need to know:
- The outputs of classification models (i.e. ResNet) is named 'logits',
    the shape is (B, C), where B means 'Batch Size' and C means 'Number of Classes or Categories'.
- Some loss functions need the tensor before the linear layer, this tensor is named 'embeddings',
    the shape is (B, N), where B means 'Batch Size' and N means 'Number of Features or Channels'.

!!! note
    You maybe confused about N and C, the reason is that,
    C represents 'Classes' or 'Categories' in some papers or codes,
    yet C represents 'Channels' in the others.
    Therefore, in this project, C means the number of classes **after** fc,
    N means the number of features **before** fc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn

from ..data import ClsDataPack
from ..utils import LOSSES

SCOPE = "classification"


class ClsLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor | dict[str, Tensor]: ...


@LOSSES.register(scope=SCOPE)
class BalancedSoftmaxLoss(ClsLoss):
    """Implementation of the paper 'Balanced Meta-Softmax for Long-Tailed Visual Recognition'."""

    def __init__(self, dist_file: Path | str) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        self.register_buffer("samples_per_class", torch.tensor(dist, torch.float32))

        self.loss_fn = nn.CrossEntropyLoss()

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        # labels: (B,)
        # (C,) -> (1, C) -> (B, C)
        spc = self.samples_per_class.unsqueeze(0).expand(*logits.shape)
        logits = logits + spc.log()

        return self.loss_fn(logits, data)


@LOSSES.register(scope=SCOPE)
class BCELoss(ClsLoss):
    # labels_one_hot = torch.zeros_like(logits)
    # labels = labels_one_hot.scatter(1, labels.view(-1, 1), 1)
    # loss = -(labels * logits.log() + (1 - labels) * (1 - logits).log()).mean()
    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        return F.binary_cross_entropy(logits, data.targets)


# class ClassBalancedCrossEntropyLoss(nn.Module):
#     def __init__(self, dist_file, beta: float) -> None:
#         super().__init__()
#
#         with open(dist_file, "rb") as fr:
#             dist = pickle.load(fr)
#
#         effective_nums = torch.FloatTensor(dist).pow(beta)
#         weights = (1.0 - beta) / (1.0 - effective_nums)
#
#         # F.cross_entropy() will do normalization for us
#         # weights = weights / weights.sum() * len(dist)
#         # weights /= weights.sum()
#
#         self.register_buffer("weights", weights)
#
#     def forward(self, logits: FloatTensor, labels: LongTensor):
#         # used_weights = weights[labels]
#         # used_weights /= used_weights.sum()
#         # loss = -(ce_loss.squeeze() * used_weights).sum()
#         return F.cross_entropy(logits, labels, self.weights)


@LOSSES.register(scope=SCOPE)
class CrossEntropyLoss(ClsLoss):
    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        return F.cross_entropy(logits, data.targets)


@LOSSES.register(scope=SCOPE)
class EffectiveNumberLoss(ClsLoss):
    def __init__(self, dist_file: Path | str, beta: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        # reciprocal of the effective numbers
        weight = (1.0 - beta) / (1.0 - np.power(beta, dist))
        weight = weight / weight.sum()  # 原文代码中有 * len(dist)，不解
        weight = FloatTensor(weight)

        self.loss_fn = nn.CrossEntropyLoss(weight)

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register(scope=SCOPE)
class FocalLoss(ClsLoss):
    """Implementation of the paper 'Focal Loss for Dense Object Detection'."""

    def __init__(self, gamma: float, alpha: float) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        # labels: (B,)
        logits = logits.softmax(1).gather(1, data.targets.view(-1, 1))  # (B,)

        loss = self.alpha * (1 - logits).pow(self.gamma) * logits.log()
        loss = -loss.mean()

        return loss


@LOSSES.register(scope=SCOPE)
class LogitAdjustedLoss(ClsLoss):
    """Implementation of the paper 'LONG-TAIL LEARNING VIA LOGIT ADJUSTMENT'."""

    def __init__(self, dist_file, tau: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        dist = FloatTensor(dist)
        prior_prob = dist / dist.sum()
        self.register_buffer("adjustment", torch.log(prior_prob * tau))

        self.loss_fn = nn.CrossEntropyLoss()

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        logits = logits + self.adjustment
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register(scope=SCOPE)
class ReWeightCELoss(ClsLoss):
    def __init__(self, dist_file: Path | str) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        weight = 1 / dist
        weight /= weight.sum()
        weight = FloatTensor(weight)
        self.loss_fn = nn.CrossEntropyLoss(weight)

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register(scope=SCOPE)
class TauNormalizedLoss(ClsLoss):
    def __init__(self, tau: float) -> None:
        super().__init__()

        self.tau = tau

        self.loss_fn = nn.CrossEntropyLoss()

    @override
    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        norm = F.normalize(logits.detach(), dim=1)
        logits = logits / norm**self.tau
        loss = self.loss_fn(logits, data.targets)

        return loss
