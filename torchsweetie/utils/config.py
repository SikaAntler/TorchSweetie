from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


def get_config(
    root_dir: Path, cfg_file: Path | str, inherit_keyword: str = "base"
) -> DictConfig:
    if isinstance(cfg_file, str):
        cfg_file = Path(cfg_file)

    if not cfg_file.is_absolute():
        cfg_file = root_dir / cfg_file

    with open(cfg_file, "r", encoding="utf-8") as fr:
        cfg: dict = yaml.safe_load(fr)  # pyright: ignore

    cfg_base = None
    if inherit_keyword in cfg:
        base = cfg.pop(inherit_keyword)
        cfg_base = get_config(root_dir, base, inherit_keyword)

    cfg: DictConfig = OmegaConf.create(cfg)

    if cfg_base is None:
        return cfg
    else:
        return OmegaConf.merge(cfg_base, cfg)  # pyright: ignore
