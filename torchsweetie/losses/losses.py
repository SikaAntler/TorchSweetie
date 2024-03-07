"""The set of some useful losses.

In order to uniform naming convention and reduce confusions, there are some rules you need to know.

1. The outputs of classification models (i.e. ResNet) are named 'logits',
   the shape is (B, N), where B means 'Batch Size' and N means 'Number of Classes or Categories',
   you maybe confused why use N instead of C in some papers and code, the reason is that 
In some papers or codes, C represents 'Classes' or 'Categories',
i.e. the shape of the output of ResNet is (B, C)

"""

import pickle
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from ..config import Config
from .builder import register_loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, dist_file) -> None:
        super().__init__()

        with open(dist_file, "rb") as fr:
            dist = pickle.load(fr)

        self.register_buffer("samples_per_class", torch.FloatTensor(dist))
        # self.samples_per_class = torch.FloatTensor(dist)

    def forward(self, logits: FloatTensor, labels: LongTensor):
        # logits: (B, N), N means num of classes after fc
        # labels: (B,)
        # spc = self.samples_per_class.type_as(logits)  # send to same device
        # (N,) -> (1, N) -> (B, N)
        spc = self.samples_per_class.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        return F.cross_entropy(logits, labels)


@register_loss
def balancedSoftmaxLoss(cfg: Config):
    return BalancedSoftmaxLoss(cfg.dist_file)


@register_loss
def bCELoss(cfg: Config):
    # labels_one_hot = torch.zeros_like(logits)
    # labels = labels_one_hot.scatter(1, labels.view(-1, 1), 1)
    # loss = -(labels * logits.log() + (1 - labels) * (1 - logits).log()).mean()
    return nn.BCELoss()


@register_loss
def bCEWithLogitsLoss(cfg: Config):
    return nn.BCEWithLogitsLoss()


class CenterLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, lambda_: float) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_  # lambda in paper is conflict with Python keyword

        self.fc = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        # embeddings: (B, C), C means channels before fc
        # labels: (B,)
        logits = self.fc(embeddings)  # (B, N)

        if self.training:
            centers = self.centers[labels]  # (B, C)
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, labels)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@register_loss
def centerLoss(cfg: Config):
    return CenterLoss(cfg.in_features, cfg.num_classes, cfg.lambda_)


class CenterFocalLoss(CenterLoss):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        lambda_: float,
        gamma: float,
        alpha: float,
    ) -> None:
        super().__init__(in_features, num_classes, lambda_)

        self.loss_fn = FocalLoss(gamma, alpha)


@register_loss
def centerFocalLoss(cfg: Config):
    return CenterFocalLoss(
        cfg.in_features, cfg.num_classes, cfg.lambda_, cfg.gamma, cfg.alpha
    )


class ClassBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, dist_file, beta: float) -> None:
        super().__init__()

        with open(dist_file, "rb") as fr:
            dist = pickle.load(fr)

        effective_nums = torch.FloatTensor(dist).pow(beta)
        weights = (1.0 - beta) / (1.0 - effective_nums)

        # F.cross_entropy() will do normalization for us
        # weights = weights / weights.sum() * len(dist)
        # weights /= weights.sum()

        self.register_buffer("weights", weights)

    def forward(self, logits: FloatTensor, labels: LongTensor):
        # used_weights = weights[labels]
        # used_weights /= used_weights.sum()
        # loss = -(ce_loss.squeeze() * used_weights).sum()
        return F.cross_entropy(logits, labels, self.weights)


@register_loss
def classBalancedCrossEntropyLoss(cfg: Config):
    return ClassBalancedCrossEntropyLoss(cfg.dist_file, cfg.beta)


@register_loss
def crossEntropyLoss(cfg: Config):
    # -logits.log_softmax(1).gather(1, labels.view(-1, 1)).mean()

    label_smoothing = cfg.label_smoothing
    if label_smoothing is None:
        label_smoothing = 0
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: FloatTensor, labels: LongTensor):
        # logits: (B, N), N means num of classes after fc
        # labels: (B,)
        logits = logits.softmax(1).gather(1, labels.view(-1, 1))  # (B,)

        loss = self.alpha * (1 - logits).pow(self.gamma) * logits.log()
        loss = -loss.mean()

        return loss


@register_loss
def focalLoss(cfg: Config):
    return FocalLoss(cfg.gamma, cfg.alpha)


class InfluenceBalancedLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, alpha: float) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: FloatTensor, labels: LongTensor):
        # embeddings: (BS, C)
        # labels: (BS,)
        logits = self.fc(embeddings)  # (BS, N)
        if self.training:
            grads = logits.softmax(1) - F.one_hot(labels, self.num_classes)  # (BS, N)
            grads = grads.abs().sum(1)  # (BS,)

            ib = grads * logits.abs().sum(1)  # (BS,)
            ib = self.alpha / ib

            loss = F.cross_entropy(logits, labels, reduction="none")
            loss = (loss * ib).mean()

            return loss
        else:
            return logits


class LabelDistributionAwareMarginLoss(nn.Module):
    def __init__(self, dist_file, max_m: float, s: float, weight=None) -> None:
        super().__init__()

        with open(dist_file, "rb") as fr:
            dist = pickle.load(fr)
        dist = torch.FloatTensor(dist)

        # m_list = 1.0 / dist.sqrt().sqrt()
        m_list = dist.pow(-1 / 4)
        m_list = m_list * (max_m / m_list.max())

        self.m_list = m_list
        self.s = s
        self.weight = weight

    def forward(self, x: FloatTensor, targets: LongTensor):
        # index = torch.zeros_like(x, dtype=torch.uint8)
        # index = index.scatter(1, targets.view(-1, 1), 1)

        index = torch.zeros_like(x, dtype=torch.bool)
        index = index.scatter(1, targets.view(-1, 1), True)

        # index_float = index.type(torch.float32) # (BS, C)
        # batch_m = torch.matmul(
        #     self.m_list[None], index_float.transpose(0, 1)
        # )  # (1, C) @ (C, BS) = (1, BS)
        # batch_m = self.m_list[None] @ index_float.T
        m_list = self.m_list.type_as(targets)
        batch_m = m_list[targets]  # (BS,)

        batch_m = batch_m.view(-1, 1)  # (BS, 1)
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, targets, weight=self.weight)


@register_loss
def labelDistributionAwareMarginLoss(cfg: Config):
    return LabelDistributionAwareMarginLoss(cfg.dist_file, cfg.max_m, cfg.s, cfg.weight)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, dist_file, tau: float) -> None:
        super().__init__()

        with open(dist_file, "rb") as fr:
            dist = pickle.load(fr)
        dist = torch.FloatTensor(dist)

        prior_prob = dist / dist.sum()
        self.register_buffer("adjustment", torch.log(prior_prob * tau))

    def forward(self, logits: FloatTensor, labels: LongTensor):
        logits = logits + self.adjustment
        return F.cross_entropy(logits, labels, reduction="mean")


@register_loss
def logitAdjustedLoss(cfg: Config):
    return LogitAdjustedLoss(cfg.dist_file, cfg.tau)


class NormalizedSoftmaxLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, temperature: float) -> None:
        super().__init__()

        weight = torch.empty(num_classes, in_features)
        bound = sqrt(1 / weight.shape[1])
        nn.init.uniform_(weight, -bound, bound)
        self.weight = nn.Parameter(weight)

        self.temperature = temperature

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        # normalize the output of conv here instead of feature extractor for compatibility
        # there is a LayerNorm layer in the original paper, but not here
        embeddings = F.normalize(embeddings, dim=1)
        norm_weight = F.normalize(self.weight, dim=1)
        logits = F.linear(embeddings, norm_weight)

        if self.training:
            loss = self.loss_fn(logits / self.temperature, labels)
            return loss
        else:
            return logits


@register_loss
def normalizedSoftmaxLoss(cfg: Config):
    return NormalizedSoftmaxLoss(cfg.in_features, cfg.num_classes, cfg.temperature)


class NormCenterLoss(nn.Module):
    def __init__(self, in_features: int, num_classes: int, lambda_: float) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_
        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        # embeddings: (B, C), where C is channels before fc
        # logits: (B, N), where N is the num of classes
        logits = self.fc(embeddings)

        if self.training:
            embeddings = F.normalize(embeddings, dim=1)

            centers = F.normalize(self.centers, dim=1)
            centers = centers[labels]
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits, labels)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@register_loss
def normCenterLoss(cfg: Config):
    return NormCenterLoss(cfg.in_features, cfg.num_classes, cfg.lambda_)


class NormTempCenterLoss(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, temperature: float, lambda_: float
    ) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)
        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.temperature = temperature

        self.lambda_ = lambda_
        self.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        # embeddings: (B, C), where C is channels before fc
        # logits: (B, N), where N is the num of classes
        logits = self.fc(embeddings)

        if self.training:
            embeddings = F.normalize(embeddings, dim=1)

            centers = F.normalize(self.centers, dim=1)
            centers = centers[labels]
            center_loss = (embeddings - centers).square().sum(1).mean()

            ce_loss = self.loss_fn(logits / self.temperature, labels)

            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@register_loss
def normTempCenterLoss(cfg: Config):
    return NormTempCenterLoss(
        cfg.in_features, cfg.num_classes, cfg.temperature, cfg.lambda_
    )


class NormalizedCenterLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float,
        lambda_: float,
    ) -> None:
        super().__init__()

        bound = sqrt(1 / in_features)

        centers = torch.empty(num_classes, in_features)
        nn.init.uniform_(centers, -bound, bound)
        self.centers = nn.Parameter(centers)

        self.lambda_ = lambda_  # lambda in paper is conflict with Python keyword

        weight = torch.empty(num_classes, in_features)
        nn.init.uniform_(weight, -bound, bound)
        self.weight = nn.Parameter(weight)

        self.temperature = temperature

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        # embeddings: (B, C), C means channels before fc
        # labels: (B,)
        embeddings = F.normalize(embeddings, dim=1)

        norm_weight = F.normalize(self.weight, dim=1)
        logits = F.linear(embeddings, norm_weight)  # (B, N)

        if self.training:
            centers = self.centers[labels]  # (B, C)
            center_loss = (embeddings - centers).square().sum(1).mean()
            ce_loss = self.loss_fn(logits / self.temperature, labels)
            return self.lambda_ * center_loss + ce_loss
        else:
            return logits


@register_loss
def normalizedCenterLoss(cfg: Config):
    return NormalizedCenterLoss(
        cfg.in_features, cfg.num_classes, cfg.temperature, cfg.lambda_
    )


class NormalizedCenterFocalLoss(NormalizedCenterLoss):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float,
        lambda_: float,
        gamma,
        alpha,
    ) -> None:
        super().__init__(in_features, num_classes, temperature, lambda_)

        self.loss_fn = FocalLoss(gamma, alpha)


@register_loss
def normalizedCenterFocalLoss(cfg: Config):
    return NormalizedCenterFocalLoss(
        cfg.in_features,
        cfg.num_classes,
        cfg.temperature,
        cfg.lambda_,
        cfg.gamma,
        cfg.alpha,
    )


class NormalizedFocalLoss(NormalizedSoftmaxLoss):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        temperature: float,
        gamma: float,
        alpha: int,
    ) -> None:
        super().__init__(in_features, num_classes, temperature)

        self.loss_fn = FocalLoss(gamma, alpha)


@register_loss
def normalizedFocalLoss(cfg: Config):
    return NormalizedFocalLoss(
        cfg.in_features, cfg.num_classes, cfg.temperature, cfg.gamma, cfg.alpha
    )


class ProxyNCA(nn.Module):
    def __init__(
        self, in_features, num_classes, label_smoothing, scaling_x, scaling_p
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.proxies = nn.Parameter(torch.randn(num_classes, in_features) / 8)
        self.label_smoothing = label_smoothing
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, embeddings: FloatTensor, labels: LongTensor):
        proxies = F.normalize(self.proxies, dim=1) * self.scaling_p
        embeddings = F.normalize(embeddings, dim=1) * self.scaling_x
        distances = torch.cdist(proxies, embeddings) ** 2

        labels = F.one_hot(labels, num_classes=self.num_classes)
        labels *= 1 - self.label_smoothing
        labels[labels == 0] = self.label_smoothing / (self.num_classes - 1)

        loss = -labels * F.log_softmax(-distances, dim=1)
        loss = loss.sum(-1).mean()

        return loss


@register_loss
def proxyNCA(cfg: Config):
    return ProxyNCA(
        cfg.in_features,
        cfg.num_classes,
        cfg.label_smoothing,
        cfg.scaling_x,
        cfg.scaling_p,
    )


class TripletMarginLossHard(nn.Module):
    def __init__(self, margin: float) -> None:
        super().__init__()

        self.loss_fn = nn.MarginRankingLoss(margin)

    def forward(self, embeddings: FloatTensor, labels: LongTensor):
        batch_size = len(labels)
        distances = (
            embeddings.pow(2).sum(1, keepdim=True).expand(batch_size, batch_size)
        )
        distances = distances + distances.T
        distances = torch.addmm(distances, embeddings, embeddings.T, alpha=-2)
        distances = distances.clamp(1e-12).sqrt()

        masks = labels.expand(batch_size, batch_size)
        masks = masks == masks.T  # True是同类，False不同

        ap_list = []  # anchor-positive
        an_list = []  # anchor-negative
        for i in range(batch_size):
            ap_list.append(distances[i][masks[i]].max())
            an_list.append(distances[i][~masks[i]].min())
        ap = torch.stack(ap_list)
        an = torch.stack(an_list)

        # loss = max(0, -y * (x1 - x2) + margin)
        # when y=1, -x1 + x2 + margin
        # so x2 is anchor-positive, x1 is anchor-negative
        loss = self.loss_fn(an, ap, torch.ones_like(an))

        return loss


@register_loss
def tripletMarginLossHard(cfg: Config):
    return TripletMarginLossHard(cfg.margin)


class TripletLoss(nn.Module):
    def __init__(self, margin: float, in_features: int, num_classes: int) -> None:
        super().__init__()

        self.triplet_loss_fn = TripletMarginLossHard(margin)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        logits = self.fc(embeddings)
        if self.training:
            triplet_loss = self.triplet_loss_fn(embeddings, labels)
            ce_loss = self.cross_entropy_loss_fn(logits, labels)
            return triplet_loss + ce_loss
        else:
            return logits


@register_loss
def tripletLoss(cfg: Config):
    return TripletLoss(cfg.margin, cfg.in_features, cfg.num_classes)


class TripletCenterLoss(nn.Module):
    def __init__(
        self, margin: float, in_features: int, num_classes: int, lambda_: float
    ) -> None:
        super().__init__()

        self.triplet_loss_fn = TripletMarginLossHard(margin)
        self.center_loss_fn = CenterLoss(in_features, num_classes, lambda_)

    def forward(self, embeddings: FloatTensor, labels: LongTensor = None):
        if self.training:
            triplet_loss = self.triplet_loss_fn(embeddings, labels)
            center_loss = self.center_loss_fn(embeddings, labels)
            return triplet_loss + center_loss
        else:
            return self.center_loss_fn(embeddings)


@register_loss
def tripletCenterLoss(cfg: Config):
    return TripletCenterLoss(cfg.margin, cfg.in_features, cfg.num_classes, cfg.lambda_)
