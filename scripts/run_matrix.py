"""Run the full experiment matrix and export evaluation/report artifacts.

Each child run has its own durable log. Completed runs are skipped; interrupted
runs resume from latest.zip. No test data is read until every run is complete.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from exploration.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def command(args, logfile):
    print("RUN", " ".join(map(str, args)), flush=True)
    with logfile.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"Command failed ({process.returncode}); inspect {logfile}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--variants", nargs="+", default=["main", "no_memory", "no_novelty"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--tag", default="full")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--test-count", type=int, default=100)
    args = parser.parse_args(argv)
    config = load_config(ROOT / args.config)
    if config["memory_frame"] != "start_aligned":
        parser.error(
            "run_matrix.py supports only start_aligned experiments. "
            "For agent_aligned, run each seed directly with python -m exploration train "
            "--config configs/memory-aligned.yaml --seed <seed>. "
            "Aligned experiments must not automatically evaluate held-out seeds or export the submission."
        )
    # Validate before creating status/run folders, inspecting completed runs,
    # launching children, or touching held-out results and submission artifacts.
    status_path = ROOT / "results" / f"{args.tag}-status.json"
    status_path.parent.mkdir(exist_ok=True)
    checkpoints = []
    status = {"tag": args.tag, "target_per_run": args.timesteps, "runs": [], "phase": "training"}

    def save():
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    for variant in args.variants:
        for seed in args.seeds:
            output = Path("runs") / (f"{variant}-{seed}" if args.tag == "full" else f"{args.tag}-{variant}-{seed}")
            folder = ROOT / output
            folder.mkdir(parents=True, exist_ok=True)
            final = folder / "final.json"
            complete = final.exists() and json.loads(final.read_text(encoding="utf-8"))["training_steps"] >= args.timesteps
            entry = {"variant": variant, "seed": seed, "directory": str(output), "state": "complete" if complete else "running"}
            status["runs"].append(entry)
            save()
            if not complete:
                argv = ["-m", "exploration", "train", "--config", args.config, "--variant", variant,
                        "--seed", str(seed), "--timesteps", str(args.timesteps), "--device", args.device,
                        "--output", str(output)]
                latest = folder / "latest.zip"
                if latest.exists():
                    argv += ["--resume", str(latest)]
                elif (folder / "config.yaml").exists():
                    raise RuntimeError(f"{folder} contains an interrupted run without a checkpoint; inspect it before restarting")
                for attempt in range(3):
                    try:
                        command(argv, folder / "console.log")
                        break
                    except RuntimeError as exc:
                        tail = (folder / "console.log").read_text(encoding="utf-8", errors="replace")[-12000:]
                        transient_cuda = "CUDA error:" in tail or "CUDA out of memory" in tail
                        if attempt < 2 and transient_cuda and latest.exists():
                            if "--resume" in argv:
                                i = argv.index("--resume")
                                del argv[i:i + 2]
                            argv += ["--resume", str(latest)]
                            entry.setdefault("recoveries", []).append({"reason": "CUDA runtime failure", "checkpoint": str(latest)})
                            save()
                            continue
                        entry["state"] = "failed"
                        entry["error"] = str(exc)
                        status["phase"] = "failed"
                        save()
                        raise
                entry["state"] = "complete"
                save()
            checkpoints.append(str(output / "best.zip"))
    status["phase"] = "held_out_evaluation"
    save()
    out = Path("results") / args.tag
    command(["-m", "exploration", "evaluate", "--agents", "random", "frontier", "ppo",
             "--checkpoints", *checkpoints, "--split", "test", "--count", str(args.test_count),
             "--output", str(out)], ROOT / "results" / f"{args.tag}-evaluation.log")
    if not args.skip_artifacts:
        status["phase"] = "artifacts"
        save()
        command(["scripts/export_submission.py", "--results", str(out), "--checkpoint", checkpoints[0]],
                ROOT / "results" / f"{args.tag}-artifacts.log")
    status["phase"] = "complete"
    save()


if __name__ == "__main__":
    main()
