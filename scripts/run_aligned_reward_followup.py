"""Durable reward-ablation training and a predeclared confirmation comparison.

No submission exporters, checkpoint replacements, or test-based model selection.
Run with the CUDA Python runtime; evaluator children use the CPU runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exploration.config import load_config, policy_label
from exploration.evaluation import summarize, write_csv

OUT = ROOT / "results/aligned-reward-followup"
SEEDS = (11, 22, 33)
PRESETS = ("easy", "medium", "hard")
FAMILIES = ("main", "no_memory", "no_novelty", "main_agent_aligned", "no_novelty_agent_aligned")
CONFIRMATION = (40000, 40099)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def preflight():
    full = load_config(ROOT / "configs/memory-aligned.yaml")
    ablated = load_config(ROOT / "configs/aligned-no-bonuses.yaml")
    assert {k: v for k, v in full.items() if k not in {"variant", "output"}} == {
        k: v for k, v in ablated.items() if k not in {"variant", "output"}}
    assert ablated["variant"] == "no_novelty" and ablated["memory_frame"] == "agent_aligned"
    # A read-only audit of recorded map-seed columns, not a rollout on the reserved seeds.
    scanned = 0
    for path in (ROOT / "results").rglob("episodes.csv"):
        if OUT in path.parents:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("seed"):
                    value = int(float(row["seed"]))
                    if CONFIRMATION[0] <= value <= CONFIRMATION[1]:
                        raise ValueError(f"Reserved confirmation seed already appears in {path}")
                    scanned += 1
    return {"matched_configuration": True, "recorded_episode_rows_audited": scanned,
            "note": "No reserved seeds found in recorded result episodes. This cannot audit unrecorded external/manual use."}


def protocol():
    evidence = preflight()
    protected = {}
    for folder in (ROOT / "results/memory-alignment-comparison", ROOT / "submission/aligned"):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv", ".pdf", ".mp4"}:
                protected[str(path.relative_to(ROOT))] = sha(path)
    protected["configs/aligned-no-bonuses.yaml"] = sha(ROOT / "configs/aligned-no-bonuses.yaml")
    for path in (ROOT / "exploration").glob("*.py"):
        protected[str(path.relative_to(ROOT))] = sha(path)
    existing = []
    for family in FAMILIES[:-1]:
        for seed in SEEDS:
            path = ROOT / f"runs/{family}-{seed}/best.zip"
            meta = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == meta["sha256"]
            existing.append({"family": family, "seed": seed, "checkpoint": str(path.relative_to(ROOT)),
                             "sha256": meta["sha256"], "selected_steps": meta["training_steps"]})
    return {"training_config": "configs/aligned-no-bonuses.yaml", "training_seeds": list(SEEDS),
            "transitions_per_run": 5000000, "confirmation_map_seeds": list(CONFIRMATION),
            "presets": list(PRESETS), "conditions": list(FAMILIES), "primary_episodes": 5700,
            "selection": "Each run's best Easy/Medium validation AUC, then discovery, then survival; never confirmation maps",
            "hypotheses": ["Seen/visit bonuses improve aligned-policy coverage/AUC", "Alignment gains extend to previously unused map seeds"],
            "inference": "Greedy PPO, deterministic frontier, random policy seeds 11/22/33; CPU single-thread children",
            "uncertainty": "Episode/run variation separate; 10000 paired-map bootstrap draws conditional on trained policies",
            "primary_contrasts": [["main_agent_aligned", "no_novelty_agent_aligned"],
                                  ["main_agent_aligned", "no_memory"], ["main_agent_aligned", "main"],
                                  ["main_agent_aligned", "frontier"], ["main_agent_aligned", "random"]],
            "preflight": evidence, "existing_checkpoints": existing, "protected_sha256": protected,
            "development_note": "Earlier 20000 and 30000 seed ranges were exposed; 40000-40099 reserved before new training/testing. No tuning after confirmation."}


def verify_protected(plan):
    for name, expected in plan["protected_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Protected artifact changed: {name}")


def run_training(state, config):
    state["phase"] = "training"
    for seed in SEEDS:
        output = ROOT / f"runs/no_novelty_agent_aligned-{seed}"
        output.mkdir(parents=True, exist_ok=True)
        expected = {**config, "seed": seed, "device": "cuda", "output": str(output.relative_to(ROOT))}
        final = output / "final.json"
        entry = state["training"].setdefault(str(seed), {})
        if final.exists():
            meta = json.loads(final.read_text())
            assert meta["seed"] == seed and meta["variant"] == "no_novelty" and meta["memory_frame"] == "agent_aligned"
            assert meta["training_steps"] >= config["timesteps"]
            entry.update(status="complete", training_steps=meta["training_steps"])
            save(OUT / "status.json", state)
            continue
        args = [sys.executable, "-u", "-m", "exploration", "train", "--config", str(ROOT / "configs/aligned-no-bonuses.yaml"),
                "--seed", str(seed), "--device", "cuda", "--output", expected["output"]]
        latest = output / "latest.zip"
        if latest.exists():
            old = json.loads(latest.with_suffix(".json").read_text())
            # Existing training.validate_resume is intentionally narrower; the runner
            # additionally requires all scientific settings to match the frozen plan.
            actual = dict(old["config"])
            for key in ("resume_from", "resume_initial_steps"):
                actual.pop(key, None)
            assert actual == expected, f"Resume configuration changed: seed {seed}"
            args += ["--resume", str(latest)]
        elif (output / "config.yaml").exists():
            raise ValueError(f"Interrupted run without a resumable checkpoint: {output}")
        with (output / "console.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=0x08000000)
            entry.update(status="running", pid=child.pid, output=str(output.relative_to(ROOT)))
            save(OUT / "status.json", state)
            result = child.wait()
        if result:
            entry.update(status="failed", exit_code=result)
            save(OUT / "status.json", state)
            raise RuntimeError(f"Training seed {seed} failed; inspect its console.log. No automatic scientific changes.")
        assert json.loads(final.read_text())["training_steps"] >= config["timesteps"]
        entry.update(status="complete", training_steps=json.loads(final.read_text())["training_steps"])
        save(OUT / "status.json", state)


def freeze_checkpoints(plan):
    records = []
    for family in FAMILIES:
        for seed in SEEDS:
            source = ROOT / f"runs/{family}-{seed}/best.zip"
            meta = json.loads(source.with_suffix(".json").read_text())
            assert sha(source) == meta["sha256"]
            assert policy_label(meta["variant"], meta.get("memory_frame", "start_aligned")) == family
            final = json.loads(source.with_name("final.json").read_text())
            assert final["training_steps"] >= 5000000
            records.append({"family": family, "seed": seed, "checkpoint": str(source.relative_to(ROOT)),
                            "sha256": meta["sha256"], "selected_steps": meta["training_steps"],
                            "completed_steps": final["training_steps"], "validation_score": meta["validation_score"]})
    for old in plan["existing_checkpoints"]:
        assert any(r["family"] == old["family"] and r["seed"] == old["seed"] and r["sha256"] == old["sha256"] for r in records)
    path = OUT / "frozen-checkpoints.json"
    if path.exists():
        assert json.loads(path.read_text()) == records
    else:
        save(path, records)
    return records


def evaluate(state, records, cpu_python):
    config = load_config(ROOT / "configs/aligned-no-bonuses.yaml")
    config["seeds"]["test"] = list(CONFIRMATION)
    config_path = OUT / "confirmation-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    jobs = [("random", ["--agents", "random"]), ("frontier", ["--agents", "frontier"])]
    jobs += [(f"{r['family']}-{r['seed']}", ["--agents", "ppo", "--checkpoints", r["checkpoint"]]) for r in records]
    state["phase"] = "confirmation_evaluation"
    active = []
    index = 0
    try:
        while index < len(jobs) or active:
            while index < len(jobs) and len(active) < 4:
                name, extra = jobs[index]
                index += 1
                folder = OUT / "parts" / name
                record = state["evaluation"].setdefault(name, {})
                if (folder / "summary.json").exists():
                    summary = json.loads((folder / "summary.json").read_text())
                    assert summary["seeds"] == list(range(CONFIRMATION[0], CONFIRMATION[1] + 1))
                    if name not in {"random", "frontier"}:
                        expected = next(r for r in records if name == f"{r['family']}-{r['seed']}")
                        assert len(summary["agents"]) == 1 and summary["agents"][0]["sha256"] == expected["sha256"]
                    record["status"] = "complete"
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / "console.log").open("a", encoding="utf-8")
                args = [str(cpu_python), "-u", "-m", "exploration", "evaluate", "--config", str(config_path),
                        *extra, "--split", "test", "--count", "100", "--output", str(folder)]
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=0x08000000)
                record.update(status="running", pid=child.pid)
                active.append((child, log, record))
            save(OUT / "status.json", state)
            for child, log, record in active[:]:
                code = child.poll()
                if code is not None:
                    log.close()
                    active.remove((child, log, record))
                    record.update(status="failed" if code else "complete", exit_code=code)
                    if code:
                        raise RuntimeError("Confirmation evaluation failed; no model/configuration changes permitted")
            if active:
                time.sleep(3)
    finally:
        for child, log, _ in active:
            child.terminate()
            child.wait()
            log.close()


def analyze(plan):
    rows, curves = [], []
    for part in sorted((OUT / "parts").iterdir()):
        with (part / "episodes.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append({k: (None if v == "" else v if k in {"agent", "preset", "termination", "memory_frame"}
                                 else int(v) if k in {"seed", "training_seed", "policy_seed", "steps", "episode_limit"}
                                 else float(v)) for k, v in row.items()})
        with (part / "curves.csv").open(encoding="utf-8", newline="") as stream:
            curves += list(csv.DictReader(stream))
    assert len(rows) == 5700 and {r["seed"] for r in rows} == set(range(40000, 40100))
    expected_agents = {"random", "frontier", *FAMILIES}
    assert {r["agent"] for r in rows} == expected_agents
    for agent in expected_agents:
        for preset, limit in zip(PRESETS, (512, 1024, 1536)):
            group = [r for r in rows if r["agent"] == agent and r["preset"] == preset]
            assert len(group) == (100 if agent == "frontier" else 300)
            assert all(r["episode_limit"] == limit for r in group)
            keys = [(r["seed"], r["training_seed"], r["policy_seed"]) for r in group]
            assert len(keys) == len(set(keys)), (agent, preset)
    groups = summarize(rows)
    overall = {family: {metric: float(np.mean([groups[f"{family}/{p}"]["metrics"][metric]["mean"] for p in PRESETS]))
                        for metric in ("revealed_coverage", "visited_coverage", "coverage_auc", "poi_fraction", "survived", "loop_rate")}
               for family in ("random", "frontier", *FAMILIES)}
    comparisons = []
    rng = np.random.default_rng(20261111)
    for preset in PRESETS:
        indices = rng.integers(0, 100, size=(10000, 100))
        for left, right in plan["primary_contrasts"]:
            record = {"preset": preset, "left": left, "right": right, "metrics": {}}
            for metric in ("revealed_coverage", "coverage_auc", "poi_fraction", "survived"):
                by_map = defaultdict(lambda: defaultdict(list))
                for row in rows:
                    if row["preset"] == preset and row["agent"] in {left, right}:
                        by_map[row["seed"]][row["agent"]].append(row[metric])
                assert sorted(by_map) == list(range(40000, 40100))
                differences = np.array([np.mean(v[left]) - np.mean(v[right]) for _, v in sorted(by_map.items())])
                ci = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
                record["metrics"][metric] = {"mean_difference": float(differences.mean()), "map_bootstrap_95ci": ci.tolist()}
            comparisons.append(record)
    write_csv(OUT / "episodes.csv", rows)
    write_csv(OUT / "curves.csv", curves)
    report = {"protocol": plan, "episodes": len(rows), "summary": groups, "overall": overall,
              "paired_comparisons": comparisons, "uncertainty": "Intervals condition on these trained runs; run SD is separate; no multiplicity correction"}
    save(OUT / "summary.json", report)
    text = ["# Aligned memory: reward ablation and reserved-map comparison", "",
            "Confirmation maps 40000-40099, frozen validation-selected policies; no confirmation tuning.", "",
            "| Agent | Coverage % | Visited % | AUC | POIs % | Survival % | Loops % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family, metrics in overall.items():
        values = [f"{metrics[m] * (1 if m == 'coverage_auc' else 100):.{3 if m == 'coverage_auc' else 1}f}" for m in metrics]
        text.append("| " + family + " | " + " | ".join(values) + " |")
    text += ["", "## Predeclared aligned reward contrast", "",
             "Full reward minus no seen/visit bonuses; coverage differences in percentage points. 95% paired-map intervals average training runs within each map."]
    for r in comparisons:
        if r["right"] == "no_novelty_agent_aligned":
            m = r["metrics"]["revealed_coverage"]
            lo, hi = m["map_bootstrap_95ci"]
            text.append(f"- {r['preset']}: {100*m['mean_difference']:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}].")
    text += ["", "A positive or negative result must be reported as measured; the protocol makes no assumed superiority claim.",
             "The old 30000 comparison and seed-overlap disclosure remain archived. These new maps are now exposed; further selection needs separate maps.",
             "Raw episode/run variation and all predeclared contrasts are in summary.json; fixed-budget curves retain information after death."]
    (OUT / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-python", default=str(ROOT.parent / ".cpu-venv/Scripts/python.exe"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / ".pipeline.lock"), timeout=0):
        path = OUT / "protocol.json"
        if path.exists():
            plan = json.loads(path.read_text())
            verify_protected(plan)
        else:
            plan = protocol()
            save(path, plan)
        if args.prepare_only:
            print(json.dumps({"protocol": str(path), "preflight": plan["preflight"]}), flush=True)
            return
        state = {"phase": "starting", "started_utc": datetime.now(UTC).isoformat(), "pid": os.getpid(),
                 "training": {}, "evaluation": {}, "protocol_sha256": sha(path)}
        if (OUT / "status.json").exists():
            previous = json.loads((OUT / "status.json").read_text())
            state["training"] = previous.get("training", {})
            state["evaluation"] = previous.get("evaluation", {})
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA training runtime required")
            run_training(state, load_config(ROOT / "configs/aligned-no-bonuses.yaml"))
            verify_protected(plan)
            records = freeze_checkpoints(plan)
            evaluate(state, records, Path(args.cpu_python))
            assert all(sha(ROOT / r["checkpoint"]) == r["sha256"] for r in records)
            state["phase"] = "analysis"
            save(OUT / "status.json", state)
            analyze(plan)
            verify_protected(plan)
            state.update(phase="complete", completed_utc=datetime.now(UTC).isoformat())
        except Exception as exc:
            state.update(phase="failed", error=repr(exc))
            save(OUT / "status.json", state)
            raise
        save(OUT / "status.json", state)


if __name__ == "__main__":
    main()
