from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..utils import BATCH_SAMPLERS, SAMPLERS
from .det_dataset import DetDataset


def create_det_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = DetDataset(**cfg.dataset)

    sampler_cfg = cfg.get("sampler")
    if sampler_cfg:
        if "scope" not in sampler_cfg:
            sampler_cfg.scope = "detection"
        sampler = SAMPLERS.create(sampler_cfg)
    else:
        sampler = None

    batch_sampler_cfg = cfg.get("batch_sampler")
    if batch_sampler_cfg:
        if "scope" not in batch_sampler_cfg:
            batch_sampler_cfg.scope = "detection"
        batch_sampler = BATCH_SAMPLERS.create(batch_sampler_cfg)
    else:
        batch_sampler = None

    dataloader = DataLoader(
        dataset,
        cfg.batch_size,
        cfg.get("shuffle", False),
        sampler,
        batch_sampler,
        cfg.get("num_workers", 0),
        dataset.collate_fn,
        cfg.get("pin_memory", False),
        cfg.get("drop_last", False),
        persistent_workers=cfg.get("persistent_workers", False),
    )

    return dataloader
