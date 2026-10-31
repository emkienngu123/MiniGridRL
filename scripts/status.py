"""Read durable run artifacts and report progress without touching training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = {"target_transitions": 45_000_000, "completed_runs": 0, "runs": []}
    total = 0
    for variant in ("main", "no_memory", "no_novelty"):
        for seed in (11, 22, 33):
            folder = ROOT / "runs" / f"{variant}-{seed}"
            steps = 0
            final = folder / "final.json"
            if final.exists():
                steps = json.loads(final.read_text(encoding="utf-8"))["training_steps"]
            else:
                for progress in folder.glob("logs*/progress.csv"):
                    try:
                        with progress.open(encoding="utf-8", newline="") as stream:
                            for row in csv.DictReader(stream):
                                steps = max(steps, int(float(row.get("time/total_timesteps", "0") or "0")))
                    except (OSError, ValueError, TypeError):
                        pass  # A concurrently written final row can be incomplete.
            complete = steps >= 5_000_000 and final.exists()
            result["completed_runs"] += int(complete)
            total += min(steps, 5_000_000)
            result["runs"].append({"variant": variant, "seed": seed, "steps": steps,
                                   "complete": complete, "started": folder.exists()})
    result["completed_transitions"] = total
    result["fraction"] = total / result["target_transitions"]
    state = ROOT / "results" / "full-status.json"
    if state.exists():
        result["pipeline"] = json.loads(state.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
