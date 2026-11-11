"""Verify completed measurements and curate selected checkpoints/history.

Uses only existing results. Never trains or selects on confirmation scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/aligned-reward-followup"
METRICS = (
    "revealed_coverage",
    "visited_coverage",
    "coverage_auc",
    "poi_fraction",
    "survived",
    "loop_rate",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_path(record):
    family, seed = record["family"], record["seed"]
    folder = ROOT / "checkpoints"
    if family == "main_agent_aligned":
        folder /= "aligned"
    elif family == "no_novelty_agent_aligned":
        folder /= "aligned-no-bonuses"
    return folder / f"{family}-{seed}.zip"


def main():
    status = json.loads((OUT / "status.json").read_text())
    assert status["phase"] == "complete"
    summary = json.loads((OUT / "summary.json").read_text())
    plan = json.loads((OUT / "protocol.json").read_text())
    for name, expected in plan["protected_sha256"].items():
        assert sha(ROOT / name) == expected, name
    with (OUT / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 5700
    groups = defaultdict(list)
    by_episode = {}
    for row in episodes:
        assert 40000 <= int(row["seed"]) <= 40099
        groups[row["agent"], row["preset"]].append(row)
        key = tuple(
            row[k] for k in ("agent", "training_seed", "policy_seed", "preset", "seed")
        )
        assert key not in by_episode
        by_episode[key] = row
    for (agent, preset), rows in groups.items():
        assert len(rows) == (100 if agent == "frontier" else 300)
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in rows])
            expected = summary["summary"][f"{agent}/{preset}"]["metrics"][metric]
            assert np.isclose(values.mean(), expected["mean"], atol=1e-12, rtol=0)
            assert np.isclose(
                values.std(ddof=1), expected["episode_std"], atol=1e-12, rtol=0
            )
    # Independent reproduction of the declared paired-map bootstrap.
    rng = np.random.default_rng(20261111)
    for preset in ("easy", "medium", "hard"):
        indices = rng.integers(0, 100, size=(10000, 100))
        for contrast in (
            r for r in summary["paired_comparisons"] if r["preset"] == preset
        ):
            for metric, expected in contrast["metrics"].items():
                arrays = []
                for agent in (contrast["left"], contrast["right"]):
                    rows = groups[agent, preset]
                    arrays.append(
                        np.array(
                            [
                                np.mean(
                                    [
                                        float(r[metric])
                                        for r in rows
                                        if int(r["seed"]) == seed
                                    ]
                                )
                                for seed in range(40000, 40100)
                            ]
                        )
                    )
                diffs = arrays[0] - arrays[1]
                assert np.isclose(
                    diffs.mean(), expected["mean_difference"], atol=1e-12, rtol=0
                )
                assert np.allclose(
                    np.percentile(diffs[indices].mean(axis=1), [2.5, 97.5]),
                    expected["map_bootstrap_95ci"],
                    atol=1e-12,
                    rtol=0,
                )
    curves = defaultdict(list)
    with (OUT / "curves.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(
                row[k]
                for k in ("agent", "training_seed", "policy_seed", "preset", "seed")
            )
            curves[key].append(row)
    assert set(curves) == set(by_episode)
    for key, rows in curves.items():
        rows.sort(key=lambda r: int(r["step"]))
        ep = by_episode[key]
        steps = [int(r["step"]) for r in rows]
        assert (
            len(steps) == len(set(steps))
            and steps[0] == 0
            and steps[-1] == int(ep["episode_limit"])
        )
        for field, endpoint in (
            ("coverage", "revealed_coverage"),
            ("poi_fraction", "poi_fraction"),
        ):
            values = [float(r[field]) for r in rows]
            assert all(b >= a - 1e-12 for a, b in pairwise(values))
            assert abs(values[-1] - float(ep[endpoint])) < 1e-12
            assert all(
                abs(float(r[field]) - float(ep[endpoint])) < 1e-12
                for r in rows
                if int(r["step"]) >= int(ep["steps"])
            )

    frozen = json.loads((OUT / "frozen-checkpoints.json").read_text())
    curated = []
    for record in frozen:
        source = ROOT / record["checkpoint"]
        target = checkpoint_path(record)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            shutil.copy2(source.with_suffix(".json"), target.with_suffix(".json"))
        assert sha(target) == record["sha256"]
        if record["family"] == "no_novelty_agent_aligned":
            history = OUT / "training" / f"{record['family']}-{record['seed']}"
            history.mkdir(parents=True, exist_ok=True)
            for name in (
                "best.json",
                "final.json",
                "runtime.json",
                "config.yaml",
                "validation.csv",
                "training_episodes.csv",
            ):
                shutil.copy2(source.parent / name, history / name)
            shutil.copy2(
                source.parent / "logs/progress.csv", history / "logs-progress.csv"
            )
            curated.append({**record, "checkpoint": str(target.relative_to(ROOT))})
    (ROOT / "checkpoints/aligned-no-bonuses/index.json").write_text(
        json.dumps(curated, indent=2), encoding="utf-8"
    )
    verification = {
        "status": "passed",
        "episodes": 5700,
        "curve_episodes": len(curves),
        "curve_rows": sum(map(len, curves.values())),
        "paired_bootstrap_draws": 10000,
        "protected_hashes_unchanged": True,
        "curated_models_verified": len(frozen),
    }
    (OUT / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    manifest = {
        "files": {
            str(p.relative_to(OUT)): sha(p)
            for p in OUT.rglob("*")
            if p.is_file()
            and p.suffix in {".json", ".csv", ".yaml", ".md"}
            and p.name not in {"manifest.json", "status.json"}
        },
        "selected_models": {
            str(checkpoint_path(r).relative_to(ROOT)): r["sha256"] for r in frozen
        },
        "verification": verification,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
