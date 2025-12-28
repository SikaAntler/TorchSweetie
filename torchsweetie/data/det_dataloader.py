from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .det_dataset import DetDataset


def create_det_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = DetDataset(**cfg.dataset)

    num_workers = cfg.get("num_workers", 0)
    pin_memory = cfg.get("pin_memory", False)
    drop_last = cfg.get("drop_last", False)
    persistent_workers = cfg.get("persistent_workers", False)
    batch_size = cfg.batch_size
    shuffle = cfg.get("shuffle", False)

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    return dataloader
