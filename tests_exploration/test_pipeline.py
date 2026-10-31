import json

import numpy as np
import pytest

from exploration.agents import FrontierAgent, RandomAgent
from exploration.config import load_config, seed_values, validate_config
from exploration.evaluation import (
    episode_metrics,
    paired_comparisons,
    run_episode,
    summarize,
)
from exploration.memory import ObservationMemory, make_env


def test_seed_partitions_disjoint():
    config = load_config()
    splits = [set(seed_values(config, split)) for split in ("train", "validation", "test")]
    assert not splits[0] & splits[1] and not splits[0] & splits[2] and not splits[1] & splits[2]
    config["seeds"]["test"] = [9999, 20000]
    with pytest.raises(ValueError, match="disjoint"):
        validate_config(config)


def test_frontier_deterministic_observation_only():
    a = run_episode(FrontierAgent(), "easy", 10, env_kwargs={"max_steps": 100})
    b = run_episode(FrontierAgent(), "easy", 10, env_kwargs={"max_steps": 100})
    assert a[3] == b[3]
    assert a[0]["revealed_coverage"] == b[0]["revealed_coverage"]
    assert a[0]["revealed_coverage"] > 0.1


def test_random_reproducibility():
    a, b = RandomAgent(11), RandomAgent(11)
    a.reset(42)
    b.reset(42)
    assert [a.act({}) for _ in range(30)] == [b.act({}) for _ in range(30)]


def test_frozen_coverage_after_death():
    env = make_env(max_steps=100)
    env.reset(seed=10)
    env.coverage_history = [10, 20]
    env.unwrapped.step_count = 1
    metrics = episode_metrics(env, terminated=True)
    expected = (15 + 99 * 20) / (100 * len(env.unwrapped.floor_cells))
    assert metrics["coverage_auc"] == pytest.approx(expected)
    assert metrics["survived"] == 0
    env.close()


def test_episode_and_run_variation_are_separate():
    rows = []
    for seed, values in ((11, (0.2, 0.4)), (22, (0.6, 0.8))):
        for value in values:
            row = {"agent": "main", "preset": "easy", "training_seed": seed}
            for metric in ("revealed_coverage", "visited_coverage", "coverage_auc", "poi_fraction", "survived",
                           "monster_collisions", "wall_collision_rate", "movement_efficiency", "repeated_state_rate",
                           "loop_rate", "inference_ms_mean"):
                row[metric] = value
            rows.append(row)
    stats = summarize(rows)["main/easy"]["metrics"]["coverage_auc"]
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["run_means"] == pytest.approx({"11": 0.3, "22": 0.7})
    assert stats["run_mean_std"] is not None


def test_paired_bootstrap_uses_shared_maps_only():
    rows = [{"agent": a, "preset": "easy", "seed": s, "coverage_auc": v}
            for a, s, v in [("main", 1, 0.6), ("random", 1, 0.2), ("main", 2, 0.9)]]
    result = paired_comparisons(rows)[0]
    assert result["paired_maps"] == 1
    assert result["coverage_auc_difference"] == pytest.approx(0.4)


def test_memory_has_no_environment_dependency():
    memory = ObservationMemory()
    env = make_env()
    raw, _ = env.unwrapped.reset(seed=8)
    memory.update(raw)
    assert set(vars(memory)) == {"map", "position", "visits", "seen_floor", "discovered_poi", "heading"}
    env.close()


def test_mapping_matches_ground_truth_relative_pose():
    for preset in ("easy", "medium", "hard"):
        env = make_env(preset)
        env.reset(seed=87)
        origin = tuple(env.unwrapped.agent_pos)
        for action in np.random.default_rng(33).integers(4, size=160):
            _, _, terminated, truncated, _ = env.step(int(action))
            true = tuple(env.unwrapped.agent_pos)
            assert env.memory.position == (27 + true[0] - origin[0], 27 + true[1] - origin[1])
            for x, y in env.memory.seen_floor:
                assert (x - 27 + origin[0], y - 27 + origin[1]) in env.unwrapped.floor_cells
            if terminated or truncated:
                break
        env.close()


def test_visible_monster_memory_does_not_persist_when_hidden():
    from minigrid.core.constants import OBJECT_TO_IDX

    memory = ObservationMemory()
    raw = {"image": np.full((7, 7, 3), 0, dtype=np.uint8), "direction": 0,
           "feedback": np.array([0, 0, 0, 1], dtype=np.float32)}
    raw["image"][3, 5, 0] = OBJECT_TO_IDX["ball"]
    memory.update(raw)
    assert np.count_nonzero(memory.map[5]) == 1
    raw["image"][3, 5, 0] = OBJECT_TO_IDX["unseen"]
    memory.update(raw)
    assert np.count_nonzero(memory.map[5]) == 0
    assert memory.seen_floor  # Static terrain knowledge survives the occlusion.


def test_checkpoint_roundtrip_and_incompatible_metadata(tmp_path):
    import torch

    from exploration.policy import load_checkpoint, make_model, save_checkpoint
    torch.set_num_threads(1)
    config = load_config()
    config.update(device="cpu", output=str(tmp_path))
    config["ppo"].update(n_steps=8, batch_size=8, n_epochs=1)
    env = make_env(max_steps=16)
    model = make_model(env, config)
    model.learn(16)
    path = save_checkpoint(model, tmp_path / "model.zip", config)
    obs, _ = env.reset(seed=3)
    expected = int(model.predict(obs, deterministic=True)[0])
    loaded, metadata = load_checkpoint(path)
    assert int(loaded.predict(obs, deterministic=True)[0]) == expected
    assert metadata["training_steps"] == 16
    manifest = path.with_suffix(".json")
    metadata["observation_version"] = "incompatible"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path)
    env.close()


def test_training_resume_keeps_total_budget_and_existing_history(tmp_path):
    from copy import deepcopy

    from exploration.training import train

    config = load_config()
    config.update(device="cpu", output=str(tmp_path / "run"), n_envs=1,
                  vector_backend="dummy", timesteps=16, eval_every=16,
                  validation_count=1, train_presets=["easy"])
    config["ppo"].update(n_steps=8, batch_size=8, n_epochs=1)
    first = train(config)
    assert first["final_steps"] == 16
    config["timesteps"] = 32
    second = train(config, resume=tmp_path / "run" / "latest.zip")
    assert second["initial_steps"] == 16
    assert second["final_steps"] == 32
    history = (tmp_path / "run" / "validation.csv").read_text(encoding="utf-8")
    assert "16," in history and "32," in history
    before = (tmp_path / "run" / "config.yaml").read_text(encoding="utf-8")
    wrong_seed = deepcopy(config)
    wrong_seed["seed"] = 22
    with pytest.raises(ValueError, match="seed/variant"):
        train(wrong_seed, resume=tmp_path / "run" / "latest.zip")
    assert (tmp_path / "run" / "config.yaml").read_text(encoding="utf-8") == before
