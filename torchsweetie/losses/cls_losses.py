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
import torch.nn.functional as F
from torch import FloatTensor, Tensor, nn

from ..data import ClsDataPack
from ..utils import LOSSES

SCOPE = "classification"


@LOSSES.register(scope=SCOPE)
class BalancedSoftmaxLoss(nn.Module):
    """Implementation of the paper 'Balanced Meta-Softmax for Long-Tailed Visual Recognition'."""

    def __init__(self, dist_file: Union[Path, str]) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        self.register_buffer("samples_per_class", FloatTensor(dist))

        self.loss_fn = CrossEntropyLoss()

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        # labels: (B,)
        # (C,) -> (1, C) -> (B, C)
        spc = self.samples_per_class.unsqueeze(0).expand(*logits.shape)
        logits = logits + spc.log()

        return self.loss_fn(logits, data)


@LOSSES.register(scope=SCOPE)
class BCELoss(nn.BCELoss):
    # labels_one_hot = torch.zeros_like(logits)
    # labels = labels_one_hot.scatter(1, labels.view(-1, 1), 1)
    # loss = -(labels * logits.log() + (1 - labels) * (1 - logits).log()).mean()
    def forward(self, input: Tensor, data: ClsDataPack) -> Tensor:
        return super().forward(input, data.targets)


@LOSSES.register(scope=SCOPE)
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input: Tensor, data: ClsDataPack) -> Tensor:
        return super().forward(input, data.targets)


@LOSSES.register(scope=SCOPE)
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

    def forward(self, embeddings: Tensor, data: ClsDataPack) -> Tensor:
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            centers = self.centers[data.targets]  # (B, C)
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, data.targets)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@LOSSES.register(scope=SCOPE)
class CEWithLinearLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()

        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, data: ClsDataPack) -> Tensor:
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            return self.loss_fn(logits, data.targets)
        else:
            return logits


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
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, data: ClsDataPack) -> Tensor:
        return super().forward(input, data.targets)


@LOSSES.register(scope=SCOPE)
class EffectiveNumberLoss(nn.Module):
    def __init__(self, dist_file: Union[Path, str], beta: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        # reciprocal of the effective numbers
        weight = (1.0 - beta) / (1.0 - np.power(beta, dist))
        weight = weight / weight.sum()  # 原文代码中有 * len(dist)，不解
        weight = FloatTensor(weight)

        self.loss_fn = nn.CrossEntropyLoss(weight)

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register(scope=SCOPE)
class FocalLoss(nn.Module):
    """Implementation of the paper 'Focal Loss for Dense Object Detection'."""

    def __init__(self, gamma: float, alpha: float) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        # labels: (B,)
        logits = logits.softmax(1).gather(1, data.targets.view(-1, 1))  # (B,)

        loss = self.alpha * (1 - logits).pow(self.gamma) * logits.log()
        loss = -loss.mean()

        return loss


# class InfluenceBalancedLoss(nn.Module):
#     def __init__(self, in_features: int, num_classes: int, alpha: float) -> None:
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.fc = nn.Linear(in_features, num_classes)
#
#     def forward(self, embeddings: FloatTensor, labels: LongTensor):
#         # embeddings: (BS, C)
#         # labels: (BS,)
#         logits = self.fc(embeddings)  # (BS, N)
#         if self.training:
#             grads = logits.softmax(1) - F.one_hot(labels, self.num_classes)  # (BS, N)
#             grads = grads.abs().sum(1)  # (BS,)
#
#             ib = grads * logits.abs().sum(1)  # (BS,)
#             ib = self.alpha / ib
#
#             loss = F.cross_entropy(logits, labels, reduction="none")
#             loss = (loss * ib).mean()
#
#             return loss
#         else:
#             return logits


# class LabelDistributionAwareMarginLoss(nn.Module):
#     def __init__(self, dist_file, max_m: float, s: float, weight=None) -> None:
#         super().__init__()
#
#         with open(dist_file, "rb") as fr:
#             dist = pickle.load(fr)
#         dist = torch.FloatTensor(dist)
#
#         # m_list = 1.0 / dist.sqrt().sqrt()
#         m_list = dist.pow(-1 / 4)
#         m_list = m_list * (max_m / m_list.max())
#
#         self.m_list = m_list
#         self.s = s
#         self.weight = weight
#
#     def forward(self, x: FloatTensor, targets: LongTensor):
#         # index = torch.zeros_like(x, dtype=torch.uint8)
#         # index = index.scatter(1, targets.view(-1, 1), 1)
#
#         index = torch.zeros_like(x, dtype=torch.bool)
#         index = index.scatter(1, targets.view(-1, 1), True)
#
#         # index_float = index.type(torch.float32) # (BS, C)
#         # batch_m = torch.matmul(
#         #     self.m_list[None], index_float.transpose(0, 1)
#         # )  # (1, C) @ (C, BS) = (1, BS)
#         # batch_m = self.m_list[None] @ index_float.T
#         m_list = self.m_list.type_as(targets)
#         batch_m = m_list[targets]  # (BS,)
#
#         batch_m = batch_m.view(-1, 1)  # (BS, 1)
#         x_m = x - batch_m
#
#         output = torch.where(index, x_m, x)
#
#         return F.cross_entropy(self.s * output, targets, weight=self.weight)


@LOSSES.register(scope=SCOPE)
class LogitAdjustedLoss(nn.Module):
    """Implementation of the paper 'LONG-TAIL LEARNING VIA LOGIT ADJUSTMENT'."""

    def __init__(self, dist_file, tau: float) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        dist = FloatTensor(dist)
        prior_prob = dist / dist.sum()
        self.register_buffer("adjustment", torch.log(prior_prob * tau))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        logits = logits + self.adjustment
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register("FeatureCenterConstraint", SCOPE)
class NormalizedCenterLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, lambda_: float) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        # self.register_buffer("centers", centers)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_
        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, data: ClsDataPack) -> Tensor:
        # embeddings: (B, N)
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, C)

        if self.training:
            # embeddings = F.normalize(embeddings.detach(), dim=1)
            embeddings = F.normalize(embeddings, dim=1)

            centers = F.normalize(self.centers, dim=1)
            centers = centers[data.targets]
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, data.targets)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


# class NormTempCenterLoss(nn.Module):
#     def __init__(
#         self, in_features: int, num_classes: int, temperature: float, lambda_: float
#     ) -> None:
#         super().__init__()
#
#         bound = sqrt(1 / in_features)
#         centers = torch.empty(num_classes, in_features)
#         nn.init.uniform_(centers, -bound, bound)
#         self.centers = nn.Parameter(centers)
#
#         self.temperature = temperature
#
#         self.lambda_ = lambda_
#         self.fc = nn.Linear(in_features, num_classes)
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
#         # embeddings: (B, C), where C is channels before fc
#         # logits: (B, N), where N is the num of classes
#         logits = self.fc(embeddings)
#
#         if self.training:
#             embeddings = F.normalize(embeddings, dim=1)
#
#             centers = F.normalize(self.centers, dim=1)
#             centers = centers[labels]
#             center_loss = (embeddings - centers).square().sum(1).mean()
#
#             ce_loss = self.loss_fn(logits / self.temperature, labels)
#
#             return self.lambda_ * center_loss + ce_loss
#         else:
#             return logits


# class ProxyNCA(nn.Module):
#     def __init__(self, in_features, num_classes, label_smoothing, scaling_x, scaling_p) -> None:
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.proxies = nn.Parameter(torch.randn(num_classes, in_features) / 8)
#         self.label_smoothing = label_smoothing
#         self.scaling_x = scaling_x
#         self.scaling_p = scaling_p
#
#     def forward(self, embeddings: FloatTensor, labels: LongTensor):
#         proxies = F.normalize(self.proxies, dim=1) * self.scaling_p
#         embeddings = F.normalize(embeddings, dim=1) * self.scaling_x
#         distances = torch.cdist(proxies, embeddings) ** 2
#
#         labels = F.one_hot(labels, num_classes=self.num_classes)
#         labels *= 1 - self.label_smoothing
#         labels[labels == 0] = self.label_smoothing / (self.num_classes - 1)
#
#         loss = -labels * F.log_softmax(-distances, dim=1)
#         loss = loss.sum(-1).mean()
#
#         return loss


@LOSSES.register(scope=SCOPE)
class ReWeightCELoss(nn.Module):
    def __init__(self, dist_file: Union[Path, str]) -> None:
        super().__init__()

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()
        weight = 1 / dist
        weight /= weight.sum()
        weight = FloatTensor(weight)
        self.loss_fn = nn.CrossEntropyLoss(weight)

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        loss = self.loss_fn(logits, data.targets)

        return loss


@LOSSES.register(scope=SCOPE)
class TauNormalizedLoss(nn.Module):
    def __init__(self, tau: float) -> None:
        super().__init__()

        self.tau = tau

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, data: ClsDataPack) -> Tensor:
        # logits: (B, C)
        norm = F.normalize(logits.detach(), dim=1)
        logits = logits / norm**self.tau
        loss = self.loss_fn(logits, data.targets)

        return loss


# class TripletMarginLossHard(nn.Module):
#     def __init__(self, margin: float) -> None:
#         super().__init__()
#
#         self.loss_fn = nn.MarginRankingLoss(margin)
#
#     def forward(self, embeddings: FloatTensor, labels: LongTensor):
#         batch_size = len(labels)
#         distances = embeddings.pow(2).sum(1, keepdim=True).expand(batch_size, batch_size)
#         distances = distances + distances.T
#         distances = torch.addmm(distances, embeddings, embeddings.T, alpha=-2)
#         distances = distances.clamp(1e-12).sqrt()
#
#         masks = labels.expand(batch_size, batch_size)
#         masks = masks == masks.T  # True是同类，False不同
#
#         ap_list = []  # anchor-positive
#         an_list = []  # anchor-negative
#         for i in range(batch_size):
#             ap_list.append(distances[i][masks[i]].max())
#             an_list.append(distances[i][~masks[i]].min())
#         ap = torch.stack(ap_list)
#         an = torch.stack(an_list)
#
#         # loss = max(0, -y * (x1 - x2) + margin)
#         # when y=1, -x1 + x2 + margin
#         # so x2 is anchor-positive, x1 is anchor-negative
#         loss = self.loss_fn(an, ap, torch.ones_like(an))
#
#         return loss


# class TripletLoss(nn.Module):
#     def __init__(self, margin: float, in_features: int, num_classes: int) -> None:
#         super().__init__()
#
#         self.triplet_loss_fn = TripletMarginLossHard(margin)
#         self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
#         self.fc = nn.Linear(in_features, num_classes)
#
#     def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
#         logits = self.fc(embeddings)
#         if self.training:
#             triplet_loss = self.triplet_loss_fn(embeddings, labels)
#             ce_loss = self.cross_entropy_loss_fn(logits, labels)
#             return triplet_loss + ce_loss
#         else:
#             return logits
