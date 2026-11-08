"""Additional training-partition diagnostic, separate from the fresh test."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
OUT = ROOT / "results/memory-alignment-comparison"


def evaluate(seed):
    folder = OUT / "training-partition" / f"main_agent_aligned-{seed}"
    if (folder / "summary.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-u", "-m", "exploration", "evaluate", "--config", str(OUT / "evaluation-config.yaml"),
            "--agents", "ppo", "--checkpoints", f"runs/main_agent_aligned-{seed}/best.zip",
            "--split", "train", "--count", "50", "--output", str(folder)]
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=0x08000000)
    print(f"Training-partition diagnostic completed: {seed}", flush=True)


if __name__ == "__main__":
    if json.loads((OUT / "status.json").read_text())["phase"] != "complete":
        raise RuntimeError("Finish primary evaluation first")
    directory = OUT / "training-partition"
    directory.mkdir(exist_ok=True)
    protocol = {"map_seeds": [0, 49], "presets": ["easy", "medium", "hard"], "training_seeds": [11, 22, 33],
                "purpose": "diagnostic sample from training partition; not guaranteed encountered by every trained run",
                "new_aligned_episodes": 450, "reference": "matching rows from results/full/train-diagnostic"}
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(evaluate, (11, 22, 33)))
