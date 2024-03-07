from pathlib import Path
from typing import Union

import yaml
from omegaconf import DictConfig, OmegaConf


def get_config(cfg_file: Union[Path, str], inherit_keyword: str = "base") -> DictConfig:
    with open(cfg_file, "r", encoding="utf-8") as fr:
        cfg: dict = yaml.safe_load(fr)

    cfg_base = None
    if inherit_keyword in cfg:
        base = cfg.pop(inherit_keyword)
        cfg_base = get_config(base, inherit_keyword)

    cfg: DictConfig = OmegaConf.create(cfg)

    if cfg_base is None:
        return cfg
    else:
        return OmegaConf.merge(cfg_base, cfg)
