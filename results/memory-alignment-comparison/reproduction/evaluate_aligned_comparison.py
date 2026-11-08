"""Frozen-checkpoint comparison on a fresh, predeclared held-out partition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
sys.path.insert(0, str(ROOT))
from exploration.config import load_config  # noqa: E402

OUT = ROOT / "results/memory-alignment-comparison"
FAMILIES = ("main_agent_aligned", "main", "no_memory", "no_novelty")
SEEDS = (11, 22, 33)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    checkpoints = []
    for family in FAMILIES:
        for seed in SEEDS:
            path = ROOT / f"runs/{family}-{seed}/best.zip"
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if digest(path) != meta["sha256"]:
                raise ValueError(f"Checkpoint checksum failed: {path}")
            checkpoints.append({"family": family, "seed": seed, "path": str(path.relative_to(ROOT)),
                                "sha256": meta["sha256"], "selected_steps": meta["training_steps"],
                                "validation_score": meta["validation_score"], "config": meta["config"]})
    protected_paths = ["results/full/summary.json", "results/full/episodes.csv", "results/full/curves.csv",
                       "submission/technical-report.pdf", "submission/manifest.json", "checkpoints/index.json"]
    protocol = {"map_seeds": [30000, 30099], "presets": ["easy", "medium", "hard"],
                "selection": "best checkpoints already selected solely on validation before fresh test evaluation",
                "action_selection": "greedy PPO; deterministic frontier; random policy seeds 11/22/33",
                "device": "cpu", "torch_threads_per_process": 1, "parallel_processes": 4,
                "latency_note": "Concurrent episode timings are descriptive; report isolated latency separately",
                "checkpoint_manifest": checkpoints,
                "original_artifact_sha256": {p: digest(ROOT / p) for p in protected_paths},
                "target_episodes": 4800}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing protocol differs; refuse to overwrite comparison")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    config = load_config(ROOT / "configs/memory-aligned.yaml")
    config["seeds"]["test"] = [30000, 30099]
    evaluation_config = OUT / "evaluation-config.yaml"
    evaluation_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    jobs = [{"name": "random", "args": ["--agents", "random"]},
            {"name": "frontier", "args": ["--agents", "frontier"]}]
    for seed in SEEDS:
        for family in FAMILIES:
            jobs.append({"name": f"{family}-{seed}", "args": ["--agents", "ppo", "--checkpoints", f"runs/{family}-{seed}/best.zip"]})
    active = []
    state = {"phase": "evaluation", "started_utc": datetime.now(timezone.utc).isoformat(),
             "jobs": [{"name": job["name"], "state": "queued"} for job in jobs]}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (OUT / "status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    index = 0
    try:
        while index < len(jobs) or active:
            while index < len(jobs) and len(active) < 4:
                job = jobs[index]
                record = state["jobs"][index]
                index += 1
                folder = OUT / "parts" / job["name"]
                if (folder / "summary.json").exists():
                    record["state"] = "complete"
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / "console.log").open("a", encoding="utf-8")
                args = [sys.executable, "-u", "-m", "exploration", "evaluate", "--config", str(evaluation_config),
                        *job["args"], "--split", "test", "--count", "100", "--output", str(folder)]
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=0x08000000)
                record.update(state="running", pid=child.pid)
                active.append((child, log, record))
            save()
            for child, log, record in active[:]:
                if child.poll() is not None:
                    log.close()
                    active.remove((child, log, record))
                    if child.returncode:
                        record.update(state="failed", exit_code=child.returncode)
                        raise RuntimeError(f"Evaluation failed: {record['name']}")
                    record["state"] = "complete"
                    print(f"Completed {record['name']}", flush=True)
            if active:
                time.sleep(3)
        for path, expected in protocol["original_artifact_sha256"].items():
            if digest(ROOT / path) != expected:
                raise AssertionError(f"Original artifact changed: {path}")
        state["phase"] = "complete"
        save()
    except Exception as exc:
        state.update(phase="failed", error=repr(exc))
        save()
        for child, log, _ in active:
            child.terminate()
            child.wait()
            log.close()
        raise


if __name__ == "__main__":
    main()
