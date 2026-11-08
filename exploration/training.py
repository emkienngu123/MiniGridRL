from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from filelock import FileLock, Timeout
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from exploration.agents import PPOAgent
from exploration.config import checkpoint_memory_frame, seed_values, validate_config
from exploration.evaluation import episode_metrics, evaluate, write_csv
from exploration.memory import make_env
from exploration.policy import load_checkpoint, make_model, save_checkpoint


class TrainingEnv(gym.Env):
    """Seed sampler for the training partition; never samples validation/test."""

    def __init__(self, config, rank):
        self.config = config
        self.rank = rank
        self.rng = np.random.default_rng(np.random.SeedSequence([config["seed"], rank]))
        self.inner = make_env(config["train_presets"][0], config["variant"], config["reward"])
        self.observation_space = self.inner.observation_space
        self.action_space = self.inner.action_space
        self.completed_metrics = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), self.rank]))
        preset = self.config["train_presets"][int(self.rng.integers(len(self.config["train_presets"])))]
        low, high = self.config["seeds"]["train"]
        map_seed = int(self.rng.integers(low, high + 1))
        self.inner.close()
        self.inner = make_env(preset, self.config["variant"], self.config["reward"])
        return self.inner.reset(seed=map_seed)

    def step(self, action):
        result = self.inner.step(action)
        if result[2] or result[3]:
            self.completed_metrics = episode_metrics(self.inner, result[2], result[3])
        return result

    def close(self):
        self.inner.close()


class ValidationCallback(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = Path(config["output"])
        self.next_evaluation = config["eval_every"]
        self.best_score = (-float("inf"),) * 3
        self.episodes = []
        self.validations = []
        self.started = time.perf_counter()
        self.start_steps = 0
        for name, target in (("training_episodes.csv", self.episodes), ("validation.csv", self.validations)):
            path = self.output / name
            if path.exists():
                with path.open(encoding="utf-8", newline="") as stream:
                    target.extend(csv.DictReader(stream))

    def _on_training_start(self):
        self.start_steps = self.num_timesteps
        self.next_evaluation = (self.num_timesteps // self.config["eval_every"] + 1) * self.config["eval_every"]
        previous = self.output / "best.json"
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            if checkpoint_memory_frame(old) != self.config.get("memory_frame", "start_aligned"):
                raise ValueError("Existing best checkpoint memory frame does not match the run")
            if old.get("validation_score"):
                self.best_score = tuple(old["validation_score"])

    def _on_step(self):
        for index, done in enumerate(self.locals["dones"]):
            if done:
                row = self.training_env.get_attr("completed_metrics", indices=index)[0]
                if row:
                    self.episodes.append({"training_step": self.num_timesteps, **row})
                    for metric in ("revealed_coverage", "poi_fraction", "survived", "loop_rate", "wall_collision_rate"):
                        self.logger.record_mean(f"exploration/{metric}", row[metric])
        if self.num_timesteps >= self.next_evaluation:
            self.validate()
            self.next_evaluation += self.config["eval_every"]
        return True

    def validate(self):
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if self.model.device.type == "cuda" else None)
        try:
            self._validate()
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            if states[3] is not None:
                torch.cuda.set_rng_state_all(states[3])

    def _validate(self):
        path = save_checkpoint(self.model, self.output / "latest.zip", self.config)
        # Separate CPU inference model, environment instances, and seed partition.
        agent = PPOAgent(path, "cpu")
        folder = self.output / "validation" / str(self.num_timesteps)
        report = evaluate([agent], self.config["train_presets"],
                          seed_values(self.config, "validation", self.config["validation_count"]),
                          folder, split="validation")
        groups = list(report["summary"].values())
        score = tuple(float(np.mean([g["metrics"][metric]["mean"] for g in groups]))
                      for metric in ("coverage_auc", "poi_fraction", "survived"))
        self.validations.append({"step": self.num_timesteps, "coverage_auc": score[0],
                                 "poi_fraction": score[1], "survived": score[2]})
        for metric, value in zip(("coverage_auc", "poi_fraction", "survived"), score):
            self.logger.record(f"validation/{metric}", value)
        if score > self.best_score:
            self.best_score = score
            save_checkpoint(self.model, self.output / "best.zip", self.config, validation_score=list(score))
        write_csv(self.output / "training_episodes.csv", self.episodes)
        write_csv(self.output / "validation.csv", self.validations)
        elapsed = time.perf_counter() - self.started
        print(f"Validation {self.num_timesteps}: {score}; elapsed {elapsed:.1f}s", flush=True)


def train(config, resume=None):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(output / ".training.lock"), timeout=0):
            return _train_unlocked(config, resume)
    except Timeout as exc:
        raise ValueError(f"Another process is already training in {output}") from exc


def validate_resume(metadata, config):
    if metadata["variant"] != config["variant"] or metadata["seed"] != config["seed"]:
        raise ValueError("Resume seed/variant must match the saved run")
    if checkpoint_memory_frame(metadata) != config.get("memory_frame", "start_aligned"):
        raise ValueError("Resume memory frame must match the saved run; train a fresh aligned policy")


def _train_unlocked(config, resume=None):
    validate_config(config)
    config = {**config, "memory_frame": config.get("memory_frame", "start_aligned")}
    # Reject incompatible resume requests before constructing worker environments.
    resumed_model = None
    if resume:
        resumed_model, metadata = load_checkpoint(resume, config["device"])
        validate_resume(metadata, config)
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.yaml").exists() and not resume:
        raise ValueError(f"Run directory {output} already exists; use a new --output or --resume")
    factories = []
    for rank in range(config["n_envs"]):
        def factory(rank=rank):
            return Monitor(TrainingEnv(config, rank))
        factories.append(factory)
    vector_type = SubprocVecEnv if config["vector_backend"] == "subproc" and config["n_envs"] > 1 else DummyVecEnv
    env = vector_type(factories)
    callback = ValidationCallback(config)
    started = time.perf_counter()
    try:
        if resume:
            model = resumed_model
            config["resume_from"] = str(resume)
            config["resume_initial_steps"] = metadata["training_steps"]
            model.set_env(env)
        else:
            model = make_model(env, config)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        initial_steps = model.num_timesteps
        log_name = f"logs-resume-{initial_steps}" if resume else "logs"
        model.set_logger(configure(str(output / log_name), ["stdout", "csv", "tensorboard"]))
        # timesteps is a total target when resuming, not an additional budget.
        remaining = max(0, config["timesteps"] - initial_steps)
        if remaining:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not bool(resume))
            if not callback.validations or callback.validations[-1]["step"] != model.num_timesteps:
                callback.validate()
        final = save_checkpoint(model, output / "final.zip", config)
        elapsed = time.perf_counter() - started
        result = {"initial_steps": initial_steps, "final_steps": model.num_timesteps,
                  "resume_from": str(resume) if resume else None,
                  "elapsed_seconds_including_validation": elapsed,
                  "transitions_per_second_including_validation": (model.num_timesteps - initial_steps) / max(elapsed, 1e-9),
                  "device": str(model.device), "checkpoint": str(final),
                  "memory_frame": config["memory_frame"],
                  "note": "PPO rounds the requested budget up to a complete rollout. Resume restarts environment sampling; it is not bitwise continuation."}
        (output / "runtime.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        env.close()
