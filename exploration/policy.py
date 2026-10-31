from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from exploration import OBSERVATION_VERSION


class ExplorationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=200)
        self.local = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
        )
        self.memory = nn.Sequential(
            nn.Conv2d(7, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        )

    def forward(self, observations):
        return torch.cat((self.local(observations["local"]),
                          self.memory(observations["memory"]), observations["scalars"]), dim=1)


def make_model(env, config):
    return PPO(
        "MultiInputPolicy", env, seed=config["seed"], device=config["device"],
        policy_kwargs={"features_extractor_class": ExplorationFeatures,
                       "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                       "normalize_images": False},
        tensorboard_log=str(Path(config["output"]) / "tensorboard"),
        verbose=1, **config["ppo"],
    )


def save_checkpoint(model, path, config, **extra):
    path = Path(path).with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    metadata = {
        "observation_version": OBSERVATION_VERSION,
        "variant": config["variant"], "seed": config["seed"],
        "training_steps": int(model.num_timesteps), "config": config,
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("minigrid", "stable-baselines3", "torch", "numpy", "gymnasium")},
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **extra,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path, device="cpu"):
    path = Path(path).with_suffix(".zip")
    manifest = path.with_suffix(".json")
    if not path.is_file() or not manifest.is_file():
        raise ValueError(f"Checkpoint requires both {path.name} and {manifest.name}")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Checkpoint observation format is incompatible with this build")
    if metadata.get("variant") not in {"main", "no_memory", "no_novelty"}:
        raise ValueError("Checkpoint has an unsupported experiment variant")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("Checkpoint checksum does not match its metadata")
    model = PPO.load(path, device=device)
    expected = {"local": (5, 7, 7), "memory": (7, 55, 55), "scalars": (8,)}
    actual = {k: tuple(v.shape) for k, v in model.observation_space.spaces.items()}
    if actual != expected or model.action_space.n != 4:
        raise ValueError("Checkpoint observation/action spaces do not match this environment")
    return model, metadata
