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
from exploration.config import checkpoint_memory_frame, validate_memory_frame


def align_memory(memory: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Read [B, 7, 55, 55] maps with the agent centered and forward upward.

    ``heading`` contains direction indices (east, south, west, north). The
    agent-marker channel supplies pose. All-zero maps remain zero, including
    the no-memory ablation. Normal nonzero inputs require one agent marker.
    Integer gather plus a validity mask implements a copy with unknown padding,
    without interpolation, wrapping, host transfers, or changes to stored maps.
    """
    if memory.ndim != 4 or tuple(memory.shape[1:]) != (7, 55, 55):
        raise ValueError("Memory alignment expects shape [batch, 7, 55, 55]")
    if heading.shape != (memory.shape[0],):
        raise ValueError("Memory alignment expects one heading index per map")
    side, center = 55, 27
    marker = memory[:, 6].flatten(1).argmax(dim=1)
    agent_x, agent_y = marker % side, marker // side
    directions = memory.new_tensor(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=torch.long)
    forward = directions[heading]
    right = directions[(heading + 1) % 4]
    v, u = torch.meshgrid(torch.arange(side, device=memory.device),
                          torch.arange(side, device=memory.device), indexing="ij")
    ahead, across = center - v, u - center
    sx = agent_x[:, None, None] + ahead * forward[:, 0, None, None] + across * right[:, 0, None, None]
    sy = agent_y[:, None, None] + ahead * forward[:, 1, None, None] + across * right[:, 1, None, None]
    valid = (sx >= 0) & (sx < side) & (sy >= 0) & (sy < side)
    indices = (sy.clamp(0, side - 1) * side + sx.clamp(0, side - 1)).flatten(1)
    aligned = memory.flatten(2).gather(2, indices[:, None].expand(-1, 7, -1))
    return aligned.reshape_as(memory) * valid[:, None].to(memory.dtype)


class ExplorationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, memory_frame="start_aligned"):
        super().__init__(observation_space, features_dim=200)
        self.memory_frame = validate_memory_frame(memory_frame)
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
        memory = observations["memory"]
        if self.memory_frame == "agent_aligned":
            memory = align_memory(memory, observations["scalars"][:, :4].argmax(dim=1))
        return torch.cat((self.local(observations["local"]),
                          self.memory(memory), observations["scalars"]), dim=1)


def make_model(env, config):
    return PPO(
        "MultiInputPolicy", env, seed=config["seed"], device=config["device"],
        policy_kwargs={"features_extractor_class": ExplorationFeatures,
                       "features_extractor_kwargs": {"memory_frame": validate_memory_frame(config.get("memory_frame", "start_aligned"))},
                       "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                       "normalize_images": False},
        tensorboard_log=str(Path(config["output"]) / "tensorboard"),
        verbose=1, **config["ppo"],
    )


def check_model_memory_frame(model, expected):
    """Check both live extractors and the constructor options saved by SB3."""
    validate_memory_frame(expected)
    saved = model.policy.features_extractor_kwargs.get("memory_frame", "start_aligned")
    frames = [saved] + [getattr(getattr(model.policy, name), "memory_frame", "start_aligned")
                        for name in ("features_extractor", "pi_features_extractor", "vf_features_extractor")]
    if any(frame != expected for frame in frames):
        raise ValueError("Checkpoint/model memory frame does not match its metadata/configuration")


def save_checkpoint(model, path, config, **extra):
    frame = validate_memory_frame(config.get("memory_frame", "start_aligned"))
    check_model_memory_frame(model, frame)
    if {"memory_frame", "config"} & extra.keys():
        raise ValueError("Checkpoint extras cannot override memory frame or configuration")
    path = Path(path).with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    metadata = {
        "observation_version": OBSERVATION_VERSION,
        "variant": config["variant"], "seed": config["seed"],
        "memory_frame": frame,
        "training_steps": int(model.num_timesteps), "config": {**config, "memory_frame": frame},
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
    frame = checkpoint_memory_frame(metadata)
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("Checkpoint checksum does not match its metadata")
    model = PPO.load(path, device=device)
    expected = {"local": (5, 7, 7), "memory": (7, 55, 55), "scalars": (8,)}
    actual = {k: tuple(v.shape) for k, v in model.observation_space.spaces.items()}
    if actual != expected or model.action_space.n != 4:
        raise ValueError("Checkpoint observation/action spaces do not match this environment")
    check_model_memory_frame(model, frame)
    metadata = {**metadata, "memory_frame": frame}
    return model, metadata
