"""Isolated CPU action+observation-memory preprocessing latency, raw samples."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
sys.path.insert(0, str(ROOT))
from exploration.agents import create_agent  # noqa: E402
from exploration.evaluation import write_csv  # noqa: E402
from exploration.memory import make_env  # noqa: E402

OUT = ROOT / "results/memory-alignment-comparison"


def main():
    if json.loads((OUT / "status.json").read_text())["phase"] != "complete":
        raise RuntimeError("Wait for parallel evaluation to finish before isolated profiling")
    torch.set_num_threads(1)
    agents = [create_agent("random", seed=s) for s in (11, 22, 33)] + [create_agent("frontier")]
    for family in ("no_novelty", "no_memory", "main", "main_agent_aligned"):
        agents.extend(create_agent("ppo", ROOT / f"runs/{family}-{s}/best.zip", device="cpu") for s in (11, 22, 33))
    samples = []
    parameters = {}
    for agent in agents:
        metadata = getattr(agent, "metadata", {})
        if hasattr(agent, "model"):
            parameters[agent.name] = {"total": sum(p.numel() for p in agent.model.policy.parameters()),
                                      "map_encoder": sum(p.numel() for p in agent.model.policy.features_extractor.memory.parameters())}
        for preset in ("easy", "medium", "hard"):
            for seed in (10040, 10041):
                env = make_env(preset, variant=metadata.get("variant", "main"), reward=metadata.get("config", {}).get("reward"))
                obs, _ = env.reset(seed=seed)
                agent.reset(seed)
                try:
                    while True:
                        preprocessing = env.preprocessing_seconds * 1000
                        start = time.perf_counter()
                        action = agent.act(obs)
                        inference = (time.perf_counter() - start) * 1000
                        obs, _, terminated, truncated, _ = env.step(action)
                        samples.append({"agent": agent.name, "training_seed": getattr(agent, "training_seed", None),
                                        "policy_seed": getattr(agent, "policy_seed", None), "preset": preset,
                                        "seed": seed, "step": env.unwrapped.step_count, "policy_ms": inference,
                                        "preprocessing_ms": preprocessing, "total_ms": inference + preprocessing})
                        if terminated or truncated:
                            break
                finally:
                    env.close()
        print(f"Profiled {agent.name} {getattr(agent, 'training_seed', None)}", flush=True)
    write_csv(OUT / "isolated-latency-samples.csv", samples)
    summary = {}
    for family in sorted({r["agent"] for r in samples}):
        group = [r for r in samples if r["agent"] == family]
        summary[family] = {"actions": len(group), "total_ms_mean": float(np.mean([r["total_ms"] for r in group])),
                           "total_ms_p50": float(np.percentile([r["total_ms"] for r in group], 50)),
                           "total_ms_p95": float(np.percentile([r["total_ms"] for r in group], 95)),
                           "policy_ms_mean": float(np.mean([r["policy_ms"] for r in group])),
                           "preprocessing_ms_mean": float(np.mean([r["preprocessing_ms"] for r in group]))}
    result = {"protocol": "One CPU process, torch threads=1; policy action plus preprocessing of its input observation; no physics/rendering/loading; first action included",
              "maps": [10040, 10041], "presets": ["easy", "medium", "hard"], "episodes": 96,
              "summary": summary, "parameters": parameters}
    (OUT / "isolated-latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
