"""Verify curated models or reproduce the frozen CPU confirmation, without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_confirmation import checkpoint_path

STUDY = ROOT / "results/aligned-reward-followup"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="results/confirmation-reproduced")
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or not 1 <= args.workers <= 4:
        parser.error("count must be 1-100 and workers 1-4")
    records = json.loads((STUDY / "frozen-checkpoints.json").read_text())
    for record in records:
        path = checkpoint_path(record)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["sha256"] == record["sha256"] and meta["seed"] == record["seed"]
    if args.verify_only:
        print("Verified all 15 curated selected checkpoints; no training or episodes executed.")
        return
    output = (ROOT / args.output).resolve()
    if (ROOT / "results") not in output.parents or output.exists():
        raise ValueError("Choose a new, nonexistent folder inside results/ to preserve recorded evidence")
    output.mkdir(parents=True)
    jobs = [("random", ["--agents", "random"]), ("frontier", ["--agents", "frontier"])]
    jobs += [(f"{r['family']}-{r['seed']}", ["--agents", "ppo", "--checkpoints", str(checkpoint_path(r))]) for r in records]
    active = []
    position = 0
    try:
        while position < len(jobs) or active:
            while position < len(jobs) and len(active) < args.workers:
                name, options = jobs[position]
                position += 1
                part = output / name
                part.mkdir()
                log = (part / "console.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", "-m", "exploration", "evaluate", "--config",
                           str(STUDY / "confirmation-config.yaml"), *options, "--split", "test", "--count",
                           str(args.count), "--output", str(part)]
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=0x08000000 if os.name == "nt" else 0)
                active.append((child, log, name))
            for child, log, name in active[:]:
                code = child.poll()
                if code is not None:
                    log.close()
                    active.remove((child, log, name))
                    if code:
                        raise RuntimeError(f"Evaluation failed: {name}; inspect its console.log")
                    print(f"Completed {name}", flush=True)
            if active:
                time.sleep(1)
    finally:
        for child, log, _ in active:
            child.terminate()
            child.wait()
            log.close()
    fields = ("agent", "training_seed", "policy_seed", "preset", "seed")
    with (STUDY / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        reference = {tuple(r[k] for k in fields): r for r in csv.DictReader(stream)}
    count = 0
    for name, _ in jobs:
        with (output / name / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                expected = reference[tuple(row[k] for k in fields)]
                for key, value in row.items():
                    if key.startswith("inference_ms"):
                        continue
                    if key in {"agent", "preset", "termination", "memory_frame"} or value == "":
                        assert value == expected[key], (name, key)
                    else:
                        assert abs(float(value) - float(expected[key])) < 1e-10, (name, key)
                count += 1
    assert count == 57 * args.count
    result = {"status": "passed", "episodes": count, "all_nontiming_metrics_match_archived": True,
              "note": "CPU verification using curated models; no runs/ layout, training, or CUDA required"}
    (output / "reproduction-check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
