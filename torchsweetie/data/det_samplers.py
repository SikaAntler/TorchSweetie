import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from ..utils import SAMPLERS


@SAMPLERS.register(scope="detection")
def WeightedRandomSampler(
    num_samples: int, weights_file: str | None = None, pow: float = 0.5, replacement: bool = True
) -> Sampler:
    if weights_file is None:
        weights = np.arange(num_samples, 0, -1)  # 默认从高到低排序
    else:
        weights = pd.read_csv(weights_file, header=None)[0].to_numpy()
    weights = np.pow(weights, pow)

    return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement)
