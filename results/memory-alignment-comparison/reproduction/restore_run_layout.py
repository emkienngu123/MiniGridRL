"""Restore selected models/history from committed artifacts; never train."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
OUT = ROOT / "results/memory-alignment-comparison"


def restore(source, destination):
    if destination.exists():
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return 1


def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    restored = 0
    for item in protocol["checkpoint_manifest"]:
        name = f"{item['family']}-{item['seed']}"
        source_dir = ROOT / "checkpoints"
        if item["family"] == "main_agent_aligned":
            source_dir /= "aligned"
        archive = source_dir / f"{name}.zip"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Curated checkpoint checksum mismatch: {archive}")
        folder = ROOT / "runs" / name
        destination = folder / "best.zip"
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Existing run contains another policy; refusing overwrite: {destination}")
        restored += restore(archive, destination)
        history = OUT / "training" / name
        restored += restore(history / "best.json", folder / "best.json")
        existing = json.loads((folder / "best.json").read_text(encoding="utf-8"))
        if existing["sha256"] != item["sha256"] or existing["seed"] != item["seed"]:
            raise ValueError(f"Existing run metadata mismatch: {folder}")
        for source in history.iterdir():
            if source.name.endswith("-progress.csv"):
                path = folder / source.name.removesuffix("-progress.csv") / "progress.csv"
            else:
                path = folder / source.name
            restored += restore(source, path)
    print(f"Verified 12 selected checkpoint archives; restored {restored} missing run-layout files. No training performed.")


if __name__ == "__main__":
    main()
