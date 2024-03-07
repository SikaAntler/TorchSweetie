from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..utils import DATASETS, SAMPLERS
from .datasets import ClsDataset


def create_cls_dataloader(
    cfg: DictConfig,
    task: str,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset: ClsDataset = DATASETS.create(cfg.dataset, task)

    # 判断task是因为验证和测试的时候不需要sampler了
    if cfg.get("sampler") and task == "train":
        sampler = SAMPLERS.create(cfg.sampler, dataset)
        dataloader = DataLoader(
            dataset,
            cfg.batch_size,
            sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=drop_last,
        )
    elif cfg.get("batch_sampler") and task == "train":
        batch_sampler = SAMPLERS.create(cfg.batch_sampler, dataset)
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            cfg.batch_size,
            shuffle,
            num_workers=8,
            pin_memory=True,
            drop_last=drop_last,
        )

    return dataloader
