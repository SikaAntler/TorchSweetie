from pathlib import Path
from typing import Iterator, Optional, Sized, Union

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torch import FloatTensor
from torch.utils.data import Sampler

from ..utils import SAMPLERS

__all__ = [
    "ReSamplerBase",
    "ClassBalancedSampler",
    "SquareRootSampler",
    "ClassBalancedBatchSampler",
]


def divide_labels_into_classes(labels: list[int], num_classes: int):
    labels = np.array(labels)
    classes_labels = []
    for cls in range(num_classes):
        cls_labels = np.argwhere(labels == cls)[:, 0]
        classes_labels.append(cls_labels)

    return classes_labels


class ReSamplerBase(Sampler[int]):
    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)

        self.labels = np.array(data_source.labels)

    def _prob_to_weights(self, prob: ndarray):
        weights = np.empty_like(self.labels, dtype=prob.dtype)
        for i, p in enumerate(prob):
            samples = self.labels == i
            weights[samples] = p / samples.sum()
        self.weights = FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[int]:
        samplers = torch.multinomial(self.weights, len(self), replacement=True)
        yield from iter(samplers.tolist())


class ClassBalancedSampler(ReSamplerBase):
    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)

        num_classes = len(np.unique(self.labels))
        prob = np.ones(num_classes) / num_classes

        self._prob_to_weights(prob)


@SAMPLERS.register
def classBalancedSampler(cfg: DictConfig, data_source: Sized):
    return ClassBalancedSampler(data_source)


class SquareRootSampler(ReSamplerBase):
    def __init__(self, data_source: Sized, dist_file: Union[Path, str]) -> None:
        super().__init__(data_source)

        dist = pd.read_csv(dist_file, header=None, index_col=None)[1].to_numpy()

        dist_sqrt = np.sqrt(dist)
        prob = dist_sqrt / dist_sqrt.sum()

        self._prob_to_weights(prob)


@SAMPLERS.register
def squareRootSampler(cfg: DictConfig, data_source: Sized):
    return SquareRootSampler(data_source, cfg.dist_file)


class ClassBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source: Sized,
        num_classes: int,
        num_sample_classes: int,
        samples_per_class: int,
    ) -> None:
        super().__init__(data_source)

        self.num_labels = len(data_source)
        self.num_classes = num_classes
        self.num_sample_classes = num_sample_classes
        self.samples_per_class = samples_per_class

        self.labels = divide_labels_into_classes(data_source.labels, num_classes)

    def __len__(self):
        return self.num_labels // (self.num_sample_classes * self.samples_per_class)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            yield self._sample_batch()

    def _sample_batch(self):
        sampled_classes = np.random.choice(
            self.num_classes, self.num_sample_classes, replace=False
        )
        sampled_indices = []
        for cls in sampled_classes:
            indices = np.random.choice(
                self.labels[cls], self.samples_per_class, replace=True
            )
            sampled_indices.extend(indices)
        return sampled_indices


@SAMPLERS.register
def classBalancedBatchSampler(cfg: DictConfig, data_source: Optional[Sized]):
    return ClassBalancedBatchSampler(
        data_source, cfg.num_classes, cfg.num_sample_classes, cfg.samples_per_class
    )
