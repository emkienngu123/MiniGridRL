"""Audit a representative Hard demonstration using a fixed validation-selected model.

These Hard episodes select presentation examples only, never model weights or settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_seeds(seeds):
    import torch

    from exploration.agents import PPOAgent
    from exploration.evaluation import episode_metrics
    from exploration.memory import make_env

    torch.set_num_threads(1)
    checkpoint = ROOT / "checkpoints/aligned/main_agent_aligned-33.zip"
    agent = PPOAgent(checkpoint, device="cpu")
    records, example = [], None
    for seed in seeds:
        env = make_env("hard", variant=agent.metadata["variant"], reward=agent.metadata["config"]["reward"])
        try:
            obs, _ = env.reset(seed=seed)
            agent.reset(seed)
            actions, states, adjacent = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                actions.append(action)
                base = env.unwrapped
                px, py = map(int, base.agent_pos)
                mx, my = map(int, base.monster_pos)
                distance = abs(px - mx) + abs(py - my)
                if distance == 1:
                    adjacent.append(len(actions))
                if seed == 10039:
                    states.append({"step": len(actions), "agent": [px, py], "heading": int(base.agent_dir),
                                   "monster": [mx, my], "manhattan_distance": distance,
                                   "monster_visible": bool(obs["memory"][5].sum()),
                                   "seen_floor": len(env.memory.seen_floor),
                                   "pois": len(env.memory.discovered_poi)})
            metrics = episode_metrics(env, terminated, truncated)
            intervals = []
            for step in adjacent:
                if intervals and step == intervals[-1][1] + 1:
                    intervals[-1][1] = step
                else:
                    intervals.append([step, step])
            records.append({**metrics, "adjacent_steps_count": len(adjacent),
                            "adjacent_intervals_count": len(intervals)})
            if seed == 10039:
                example = {"agent": agent.name, "seed": seed, "preset": "hard",
                           "checkpoint_sha256": agent.metadata["sha256"], "metrics": metrics,
                           "actions": actions, "states_after_step": states,
                           "adjacent_steps": adjacent, "adjacent_intervals": intervals}
        finally:
            env.close()
    return records, example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="submission/final/hard-example-selection")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four CPU workers")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a new directory within the repository; existing audit evidence is preserved")
    output.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    records, example = [], None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_seeds, list(range(10000 + i, 10050, args.workers))) for i in range(args.workers)]
        for job in as_completed(jobs):
            rows, candidate = job.result()
            records.extend(rows)
            if candidate:
                example = candidate
            print(f"Checked {len(records)}/50 Hard validation episodes on CPU", flush=True)
    import numpy as np

    from exploration.evaluation import write_csv

    records.sort(key=lambda row: row["seed"])
    assert len(records) == 50 and example is not None
    median = float(np.median([row["revealed_coverage"] for row in records]))
    ranked = sorted(records, key=lambda row: (abs(row["revealed_coverage"] - median), row["seed"]))
    distance = abs(example["metrics"]["revealed_coverage"] - median)
    nearest = [row["seed"] for row in records if np.isclose(abs(row["revealed_coverage"] - median),
                                                        abs(ranked[0]["revealed_coverage"] - median), atol=1e-12, rtol=0)]
    summary = {"checkpoint": "checkpoints/aligned/main_agent_aligned-33.zip",
               "checkpoint_sha256": sha(ROOT / "checkpoints/aligned/main_agent_aligned-33.zip"),
               "training_seed": 33, "device": "cpu", "map_seeds": [10000, 10049], "episodes": 50,
               "median_coverage": median, "example_seed": 10039,
               "example_coverage": example["metrics"]["revealed_coverage"],
               "example_distance_to_median": distance, "nearest_median_candidates": nearest,
               "example_is_nearest_median_candidate": 10039 in nearest,
               "example_adjacent_steps": example["adjacent_steps"],
               "example_adjacent_intervals": example["adjacent_intervals"],
               "selection": "Reviewer proposed map 10039 after a 50-map audit. Independently checked against all 50: chosen for near-median coverage and a survived close pass. This is a qualitative example, not a random episode.",
               "information_boundary": "Hard maps do not select the checkpoint or hyperparameters. Evaluator-only proximity is used solely to annotate the demonstration.",
               "not_primary_evidence": True}
    write_csv(output / "episodes.csv", records)
    (output / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "selected-trace.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
