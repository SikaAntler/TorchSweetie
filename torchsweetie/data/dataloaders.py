from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..utils import BATCH_SAMPLERS
from .datasets import ClsDataset


def create_cls_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = ClsDataset(**cfg.dataset)

    # 判断task是因为验证和测试的时候不需要sampler了

    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", False)
    drop_last = cfg.get("drop_last", False)
    persistent_workers = cfg.get("persistent_workers", False)

    batch_sampler_cfg = cfg.get("batch_sampler")
    if batch_sampler_cfg is not None:
        batch_sampler = BATCH_SAMPLERS.create(batch_sampler_cfg)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )
    else:
        batch_size = cfg.batch_size
        shuffle = cfg.get("shuffle", False)
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

    return dataloader
