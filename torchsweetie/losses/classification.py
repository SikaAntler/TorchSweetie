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


from math import sqrt
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import FloatTensor, LongTensor, Tensor

from ..utils import LOSSES

__all__ = [
    "BalancedSoftmaxLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "CenterLoss",
    "CEWithLinearLoss",
    "CrossEntropyLoss",
    "EffectiveNumberLoss",
    "FocalLoss",
    "LogitAdjustedLoss",
    "NormalizedCenterLoss",
    "ReWeightCELoss",
    "TauNormalizedLoss",
]


class BalancedSoftmaxLoss(nn.Module):
    """Implementation of the paper 'Balanced Meta-Softmax for Long-Tailed Visual Recognition'."""

    def __init__(self, dist_file: Union[Path, str]) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        self.register_buffer("samples_per_class", FloatTensor(dist))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, labels: LongTensor):
        # logits: (B, C)
        # labels: (B,)
        # (C,) -> (1, C) -> (B, C)
        spc = self.samples_per_class.unsqueeze(0).expand(*logits.shape)
        logits = logits + spc.log()

        return self.loss_fn(logits, labels)


@LOSSES.register
def balancedSoftmaxLoss(cfg: DictConfig):
    return BalancedSoftmaxLoss(cfg.dist_file)


class BCELoss(nn.BCELoss):
    pass


@LOSSES.register
def bCELoss(cfg: DictConfig):
    # labels_one_hot = torch.zeros_like(logits)
    # labels = labels_one_hot.scatter(1, labels.view(-1, 1), 1)
    # loss = -(labels * logits.log() + (1 - labels) * (1 - logits).log()).mean()
    return BCELoss()


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    pass


@LOSSES.register
def bCEWithLogitsLoss(cfg: DictConfig):
    return BCEWithLogitsLoss()


class CenterLoss(nn.Module):
    """Implementation of the paper 'A Discriminative Feature Learning Approach for Deep Face Recognition'."""

    def __init__(self, in_features: int, num_classes: int, lambda_: float) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_  # lambda in paper is conflict with Python keyword
        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels: LongTensor = None):
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            centers = self.centers[labels]  # (B, C)
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, labels)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@LOSSES.register
def centerLoss(cfg: DictConfig):
    return CenterLoss(cfg.in_features, cfg.num_classes, cfg.lambda_)


class CEWithLinearLoss(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()

        self.fc = nn.Linear(num_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: LongTensor = None):
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            return self.loss_fn(logits, labels)
        else:
            return logits


@LOSSES.register
def cEWithLinearLoss(cfg: DictConfig):
    return CEWithLinearLoss(cfg.num_features, cfg.num_classes)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass


@LOSSES.register
def crossEntropyLoss(cfg: DictConfig):
    # -logits.log_softmax(1).gather(1, labels.view(-1, 1)).mean()
    label_smoothing = cfg.get("label_smoothing", 0)

    return CrossEntropyLoss(label_smoothing=label_smoothing)


class EffectiveNumberLoss(nn.Module):
    def __init__(self, dist_file: Union[Path, str], beta: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        # reciprocal of the effective numbers
        weight = (1.0 - beta) / (1.0 - np.power(beta, dist))
        weight = weight / weight.sum()  # 原文代码中有 * len(dist)，不解

        self.loss_fn = nn.CrossEntropyLoss(weight=FloatTensor(weight))

    def forward(self, logits: Tensor, labels: LongTensor):
        loss = self.loss_fn(logits, labels)

        return loss


@LOSSES.register
def effectiveNumberLoss(cfg: DictConfig):
    return EffectiveNumberLoss(cfg.dist_file, cfg.beta)


class FocalLoss(nn.Module):
    """Implementation of the paper 'Focal Loss for Dense Object Detection'."""

    def __init__(self, gamma: float, alpha: float) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, labels: LongTensor):
        # logits: (B, C)
        # labels: (B,)
        logits = logits.softmax(1).gather(1, labels.view(-1, 1))  # (B,)

        loss = self.alpha * (1 - logits).pow(self.gamma) * logits.log()
        loss = -loss.mean()

        return loss


@LOSSES.register
def focalLoss(cfg: DictConfig):
    return FocalLoss(cfg.gamma, cfg.alpha)


class LogitAdjustedLoss(nn.Module):
    """Implementation of the paper 'LONG-TAIL LEARNING VIA LOGIT ADJUSTMENT'."""

    def __init__(self, dist_file, tau: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        dist = FloatTensor(dist)
        prior_prob = dist / dist.sum()
        self.register_buffer("adjustment", torch.log(prior_prob * tau))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, labels: LongTensor):
        logits = logits + self.adjustment
        loss = self.loss_fn(logits, labels)

        return loss


@LOSSES.register
def logitAdjustedLoss(cfg: DictConfig):
    return LogitAdjustedLoss(cfg.dist_file, cfg.tau)


class NormalizedCenterLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, lambda_: float) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_
        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, labels: LongTensor = None):
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            embeddings = F.normalize(embeddings.detach(), dim=1)

            centers = F.normalize(self.centers.detach(), dim=1)
            centers = centers[labels]
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, labels)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@LOSSES.register
def normalizedCenterLoss(cfg: DictConfig):
    return NormalizedCenterLoss(cfg.in_features, cfg.num_classes, cfg.lambda_)


class ReWeightCELoss(nn.Module):
    def __init__(self, dist_file: Union[Path, str]) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        weight = 1 / dist
        weight /= weight.sum()
        self.loss_fn = nn.CrossEntropyLoss(weight=FloatTensor(weight))

    def forward(self, logits: Tensor, labels: LongTensor):
        loss = self.loss_fn(logits, labels)

        return loss


@LOSSES.register
def reWeightCELoss(cfg: DictConfig):
    return ReWeightCELoss(cfg.dist_file)


class TauNormalizedLoss(nn.Module):
    def __init__(self, tau: float) -> None:
        super().__init__()

        self.tau = tau

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, labels: LongTensor):
        # logits: (B, C)
        norm = F.normalize(logits.detach(), dim=1)
        logits = logits / norm**self.tau
        loss = self.loss_fn(logits, labels)

        return loss


@LOSSES.register
def tauNormalizedLoss(cfg: DictConfig):
    return TauNormalizedLoss(cfg.tau)
