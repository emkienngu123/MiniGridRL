"""Fixed illustrative map, seed 11 policies; ground truth used for drawing only."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
sys.path.insert(0, str(ROOT))
from exploration.agents import create_agent  # noqa: E402
from exploration.evaluation import episode_metrics  # noqa: E402
from exploration.memory import make_env  # noqa: E402

OUT = ROOT / "results/memory-alignment-comparison"


def main():
    torch.set_num_threads(1)
    families = ("main", "no_memory", "main_agent_aligned", "frontier")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    rows = []
    for family, ax in zip(families, axes):
        agent = create_agent("frontier") if family == "frontier" else create_agent("ppo", ROOT / f"runs/{family}-11/best.zip")
        meta = getattr(agent, "metadata", {})
        env = make_env("hard", variant=meta.get("variant", "main"), reward=meta.get("config", {}).get("reward"))
        obs, _ = env.reset(seed=30000)
        agent.reset(30000)
        base = env.unwrapped
        terrain = np.zeros((base.width, base.height), dtype=float)
        for x, y in base.floor_cells:
            terrain[y, x] = 1
        initial = tuple(base.agent_pos)
        monster = tuple(base.monster_pos)
        points = sorted(base.poi_positions)
        path = [initial]
        trace = []
        phases = {"before_90_percent": {"actions": 0, "loops": 0}, "after_90_percent": {"actions": 0, "loops": 0}}
        try:
            while True:
                before = len(env.memory.seen_floor) / len(base.floor_cells)
                old_loops = env.loop_steps
                action = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                key = "before_90_percent" if before < .9 else "after_90_percent"
                phases[key]["actions"] += 1
                phases[key]["loops"] += env.loop_steps - old_loops
                path.append(tuple(base.agent_pos))
                trace.append({"action": action, "position": list(map(int, base.agent_pos)), "monster": list(map(int, base.monster_pos)),
                              "coverage": len(env.memory.seen_floor) / len(base.floor_cells), "reward": reward})
                if terminated or truncated:
                    break
            metrics = episode_metrics(env, terminated, truncated)
            folder = OUT / "qualitative" / family
            folder.mkdir(parents=True, exist_ok=True)
            payload = {"map_seed": 30000, "preset": "hard", "training_seed": 11 if family != "frontier" else None,
                       "agent": family, "illustrative_only": True, "metrics": metrics, "loop_phases": phases, "trace": trace}
            (folder / "trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.append({k: v for k, v in payload.items() if k != "trace"})
            ax.imshow(terrain, cmap="gray", vmin=0, vmax=1, origin="upper")
            route = np.array(path)
            ax.plot(route[:, 0], route[:, 1], color="#517dc4", alpha=.65, lw=1)
            ax.scatter(route[::16, 0], route[::16, 1], c=np.arange(len(route))[::16], cmap="plasma", s=7)
            ax.scatter(*zip(*points), marker="*", s=65, color="#30b06d", edgecolor="#226e45", zorder=5)
            ax.scatter(*initial, marker="o", s=35, facecolors="none", edgecolors="#1769cc", zorder=6)
            ax.scatter(*monster, marker="x", s=30, color="#d43e40", zorder=5)
            ax.scatter(*path[-1], marker="X" if terminated else "s", s=35, color="#df6247", zorder=7)
            names = {"main": "Original map", "no_memory": "Without map", "main_agent_aligned": "Aligned map", "frontier": "Frontier"}
            ax.set_title(f"{names[family]}\nCoverage {metrics['revealed_coverage']:.1%}; POIs {metrics['poi_discovered']}/5\n{metrics['termination']}", fontsize=10)
            ax.set(xticks=[], yticks=[])
        finally:
            env.close()
    fig.suptitle("Illustrative Hard map 30000; trained-policy seed 11 — evaluator terrain shown for explanation")
    fig.text(.5, .02, "Blue circle: start   Green stars: POIs   Red x: initial monster   Orange: end   Paths colored by elapsed actions", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .9))
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/illustrative-paths.png", dpi=160)
    plt.close(fig)
    (OUT / "qualitative-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
