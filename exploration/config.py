from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

PRESETS = {
    "easy": {"size": 15, "poi_count": 2, "max_steps": 512, "monster_interval": 3},
    "medium": {"size": 21, "poi_count": 3, "max_steps": 1024, "monster_interval": 2},
    "hard": {"size": 27, "poi_count": 5, "max_steps": 1536, "monster_interval": 2},
}
DEFAULTS = {
    "seed": 11,
    "variant": "main",
    "timesteps": 5_000_000,
    "device": "auto",
    "torch_threads": 1,
    "n_envs": 16,
    "vector_backend": "subproc",
    "train_presets": ["easy", "medium"],
    "output": "runs/main-11",
    "eval_every": 250_000,
    "validation_count": 50,
    "seeds": {"train": [0, 9999], "validation": [10000, 10049], "test": [20000, 20099]},
    "reward": {"seen": 0.02, "visited": 0.01, "poi": 1.0, "step": -0.001, "wall": -0.02, "death": -1.0},
    "ppo": {
        "n_steps": 256, "batch_size": 256, "n_epochs": 4,
        "learning_rate": 0.0003, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "target_kl": 0.03,
        "vf_coef": 0.5, "max_grad_norm": 0.5,
    },
}


def load_config(path: str | Path | None = None) -> dict:
    config = deepcopy(DEFAULTS)
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    intervals = list(config["seeds"].values())
    for low, high in intervals:
        if low < 0 or high < low:
            raise ValueError("Seed intervals must be nonnegative inclusive ranges")
    for i, (low, high) in enumerate(intervals):
        if any(max(low, lo) <= min(high, hi) for lo, hi in intervals[i + 1:]):
            raise ValueError("Training, validation, and test seed sets must be disjoint")
    if config["variant"] not in {"main", "no_memory", "no_novelty"}:
        raise ValueError("Unknown experiment variant")
    if not config["train_presets"] or any(p not in PRESETS for p in config["train_presets"]):
        raise ValueError("Unknown training preset")
    if config["n_envs"] < 1 or config["timesteps"] < 1 or config["eval_every"] < 1:
        raise ValueError("Environment count, timesteps, and evaluation interval must be positive")
    if not 1 <= config["validation_count"] <= config["seeds"]["validation"][1] - config["seeds"]["validation"][0] + 1:
        raise ValueError("validation_count exceeds the validation manifest")


def seed_values(config: dict, split: str, count: int | None = None) -> list[int]:
    low, high = config["seeds"][split]
    values = list(range(low, high + 1))
    if count is not None:
        if not 1 <= count <= len(values):
            raise ValueError(f"count must be between 1 and {len(values)} for {split}")
        values = values[:count]
    return values
