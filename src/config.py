"""Load config.yaml and resolve project-relative paths."""
import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cache = None


def load_config(path=CONFIG_PATH):
    """Return a fresh (deep-copied) config dict so callers can override settings safely."""
    global _cache
    if _cache is None or path != CONFIG_PATH:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if path != CONFIG_PATH:
            return cfg
        _cache = cfg
    return copy.deepcopy(_cache)


def project_path(rel):
    return PROJECT_ROOT / rel


def dataset_names(cfg=None):
    cfg = cfg or load_config()
    return list(cfg["datasets"].keys())
