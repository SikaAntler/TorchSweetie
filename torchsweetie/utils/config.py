from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

KEYWORD_BASE = "_base_"
KEYWORD_DELETE = "_delete_"


def _handle_delete(base_cfg: DictConfig, cfg: DictConfig) -> None:
    for key, value in cfg.items():
        # 如果base配置中不存在这个key，则跳过
        if key not in base_cfg:
            continue

        # 如果value不是字典类型，则跳过
        if not isinstance(value, DictConfig):
            continue

        # 如果value中含有delete关键字，则删除base配置中对应的key/value
        if KEYWORD_DELETE in value:
            is_deleted = value.pop(KEYWORD_DELETE)
            if is_deleted:
                base_cfg.pop(key)
        else:  # 重复检查value子配置
            _handle_delete(value, base_cfg[key])


def load_config(cfg_file: Path) -> DictConfig:
    with open(cfg_file, "r", encoding="utf-8") as fr:
        context: dict = yaml.safe_load(fr)

    cfg = OmegaConf.create(context)

    if KEYWORD_BASE in cfg:
        base = cfg.pop(KEYWORD_BASE)
        base_cfg_file = cfg_file.parent / base
        base_cfg = load_config(base_cfg_file)
        _handle_delete(base_cfg, cfg)
        return OmegaConf.merge(base_cfg, cfg)  # pyright: ignore
    else:
        return cfg


def save_config(cfg: DictConfig, cfg_file: Path) -> None:
    OmegaConf.save(cfg, cfg_file)
