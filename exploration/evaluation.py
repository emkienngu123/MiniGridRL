from __future__ import annotations

import csv
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from exploration.memory import make_env


def episode_metrics(env, terminated=False, truncated=False):
    """Evaluator-only ground-truth denominators. Never added to observations/info."""
    base = env.unwrapped
    steps = base.step_count
    cells = len(base.floor_cells)
    history = np.asarray(env.coverage_history, dtype=float) / cells
    # Coverage after death remains fixed through the planned episode budget.
    padded = np.pad(history, (0, base.max_steps + 1 - len(history)), mode="edge")
    visited = int(np.count_nonzero(env.memory.visits))
    discoveries = len(env.memory.discovered_poi)
    return {
        "seed": int(base.np_random_seed), "preset": base.preset, "steps": steps,
        "episode_limit": base.max_steps,
        "revealed_coverage": float(history[-1]),
        "visited_coverage": visited / cells,
        "coverage_auc": float(np.mean((padded[:-1] + padded[1:]) / 2)),
        "poi_discovered": discoveries, "poi_fraction": discoveries / base.poi_count,
        "survived": int(truncated and not terminated),
        "completion": int(truncated and not terminated),
        "termination": "monster" if terminated else ("time_limit" if truncated else "running"),
        "monster_collisions": env.monster_collisions,
        "wall_collisions": env.wall_collisions,
        "wall_collision_rate": env.wall_collisions / max(1, steps),
        "movement_efficiency": max(0, visited - 1) / max(1, steps),
        "repeated_state_rate": env.repeated_steps / max(1, steps),
        "loop_rate": env.loop_steps / max(1, steps),
        "reward": env.total_reward,
    }


def run_episode(agent, preset, seed, *, variant="main", reward=None, env_kwargs=None,
                capture=False, frame_stride=8):
    env = make_env(preset, variant=variant, reward=reward, **(env_kwargs or {}))
    obs, _ = env.reset(seed=int(seed))
    agent.reset(int(seed))
    times = []
    frames, actions = [], []
    terminated = truncated = False
    try:
        if capture:
            frames.append(env.unwrapped.get_frame(tile_size=20))
        while not (terminated or truncated):
            start = time.perf_counter()
            action = agent.act(obs)
            act_time = time.perf_counter() - start
            obs, _, terminated, truncated, _ = env.step(action)
            times.append((act_time + env.preprocessing_seconds) * 1000)
            actions.append(action)
            if capture and (len(actions) % frame_stride == 0 or terminated or truncated):
                frames.append(env.unwrapped.get_frame(tile_size=20))
        metrics = episode_metrics(env, terminated, truncated)
        metrics.update({
            "agent": agent.name, "training_seed": getattr(agent, "training_seed", None),
            "policy_seed": getattr(agent, "policy_seed", None),
            "inference_ms_mean": float(np.mean(times)),
            "inference_ms_p50": float(np.percentile(times, 50)),
            "inference_ms_p95": float(np.percentile(times, 95)),
        })
        curves = []
        for step in sorted(set(range(0, env.unwrapped.max_steps + 1, 32)) | {env.unwrapped.max_steps}):
            index = min(step, len(env.coverage_history) - 1)
            curves.append({"agent": agent.name, "training_seed": metrics["training_seed"],
                           "policy_seed": metrics["policy_seed"], "preset": preset, "seed": seed,
                           "step": step, "budget_fraction": step / env.unwrapped.max_steps,
                           "coverage": env.coverage_history[index] / len(env.unwrapped.floor_cells),
                           "poi_fraction": env.discovery_history[index] / env.unwrapped.poi_count})
        return metrics, curves, frames, actions
    finally:
        env.close()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["agent"], row["preset"])].append(row)
    result = {}
    metrics = ["revealed_coverage", "visited_coverage", "coverage_auc", "poi_fraction", "survived",
               "monster_collisions", "wall_collision_rate", "movement_efficiency", "repeated_state_rate",
               "loop_rate", "inference_ms_mean"]
    for (agent, preset), records in sorted(groups.items()):
        summary = {"episodes": len(records), "metrics": {}}
        step_count = sum(row.get("steps", 0) for row in records)
        summary["inference_step_mean_ms"] = (
            sum(row["inference_ms_mean"] * row.get("steps", 0) for row in records) / step_count
            if step_count else None
        )
        for metric in metrics:
            values = np.asarray([row[metric] for row in records], dtype=float)
            by_run = defaultdict(list)
            for row in records:
                by_run[row.get("training_seed") if row.get("training_seed") is not None else row.get("policy_seed")].append(float(row[metric]))
            means = [np.mean(v) for v in by_run.values()]
            summary["metrics"][metric] = {
                "mean": float(values.mean()),
                "episode_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "run_means": {str(k): float(np.mean(v)) for k, v in by_run.items()},
                "run_mean_std": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            }
        result[f"{agent}/{preset}"] = summary
    return result


def paired_comparisons(rows, resamples=2000):
    # Average runs within each map, then bootstrap paired map differences. This
    # interval describes map variation, not uncertainty across training seeds.
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["preset"], row["agent"])][int(row["seed"])].append(row["coverage_auc"])
    results = []
    rng = np.random.default_rng(90210)
    for (preset, agent), values in sorted(grouped.items()):
        if agent in {"random", "frontier"}:
            continue
        for baseline in ("random", "frontier"):
            base = grouped.get((preset, baseline), {})
            seeds = sorted(set(values) & set(base))
            if not seeds:
                continue
            diffs = np.array([np.mean(values[s]) - np.mean(base[s]) for s in seeds])
            samples = rng.choice(diffs, size=(resamples, len(diffs)), replace=True).mean(axis=1)
            results.append({"preset": preset, "agent": agent, "baseline": baseline, "paired_maps": len(seeds),
                            "coverage_auc_difference": float(diffs.mean()),
                            "map_bootstrap_95ci": list(map(float, np.percentile(samples, [2.5, 97.5])))})
    return results


def evaluate(agents, presets, seeds, output, *, env_kwargs=None, split="validation", reward=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for agent in agents:
        variant = getattr(agent, "metadata", {}).get("variant", "main")
        agent_reward = getattr(agent, "metadata", {}).get("config", {}).get("reward", reward)
        for preset in presets:
            for seed in seeds:
                row, curve, _, _ = run_episode(agent, preset, seed, variant=variant,
                                                reward=agent_reward, env_kwargs=env_kwargs)
                rows.append(row)
                curves.extend(curve)
            latest = rows[-len(seeds):]
            print(f"{agent.name} {preset}: coverage={np.mean([r['revealed_coverage'] for r in latest]):.3f}, "
                  f"survival={np.mean([r['survived'] for r in latest]):.3f}", flush=True)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "curves.csv", curves)
    provenance = []
    for agent in agents:
        metadata = getattr(agent, "metadata", {})
        provenance.append({"agent": agent.name, "training_seed": getattr(agent, "training_seed", None),
                           "policy_seed": getattr(agent, "policy_seed", None),
                           "checkpoint": getattr(agent, "checkpoint", None),
                           "training_steps": metadata.get("training_steps"), "sha256": metadata.get("sha256"),
                           "completed_training_steps": getattr(agent, "completed_training_steps", None),
                           "training_config": metadata.get("config"),
                           "inference_device": str(agent.model.device) if hasattr(agent, "model") else "cpu"})
    report = {"split": split, "seeds": list(seeds), "presets": list(presets),
              "python": platform.python_version(), "platform": platform.platform(), "agents": provenance,
              "summary": summarize(rows), "paired_comparisons": paired_comparisons(rows)}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
