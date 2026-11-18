"""YAML configuration with dot access and derived paths."""
from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(f"config has no key '{k}'") from e

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path, overrides=None) -> Config:
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    cfg.cache_dir = Path(cfg.cache_dir)
    cfg.runs_dir = Path(cfg.runs_dir)
    cfg.lr_drops = {int(k): float(v) for k, v in (cfg.get("lr_drops") or {}).items()}
    return cfg


def read_basins(cfg) -> list:
    return [l.strip().zfill(8) for l in Path(cfg.basin_file).read_text().splitlines() if l.strip()]


def run_dir(cfg) -> Path:
    return cfg.runs_dir / f"{cfg.family}_fold{cfg.fold}_seed{cfg.seed}"
