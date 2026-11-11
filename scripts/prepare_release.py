"""Refresh submission documents in an existing CPU build and assemble release assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="dist/final/MiniGridExplorerAligned")
    parser.add_argument("--submission", default="submission/final")
    args = parser.parse_args()
    folder = (ROOT / args.build).resolve()
    submission = (ROOT / args.submission).resolve()
    for path in (folder, submission):
        if not path.is_relative_to(ROOT) or not path.is_dir():
            raise ValueError(f"Expected an existing directory within the repository: {path}")
    info = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
    expected = json.loads((submission / "manifest.json").read_text(encoding="utf-8"))
    if info["cuda_runtime"] is not None or info["memory_frame"] != "agent_aligned":
        raise ValueError("Final build must use CPU PyTorch and aligned memory")
    model = folder / "_internal/checkpoints/demo.zip"
    if sha256(model) != expected["checkpoint_sha256"]:
        raise ValueError("Bundled weights differ from the validation-selected demonstration model")
    assets = {}
    for name in ("technical-report.pdf", "demo.mp4", "START-HERE.md"):
        source = submission / name
        assets[name] = sha256(source)
        shutil.copy2(source, folder / name)
    if assets["technical-report.pdf"] != expected["report_sha256"]:
        raise ValueError("Report does not match the submission manifest")
    if assets["demo.mp4"] != expected["video_sha256"]:
        raise ValueError("Demo does not match the submission manifest")
    (folder / "SUBMISSION-ASSETS.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "README.md", folder / "README.md")
    output = folder.parent
    archive = Path(shutil.make_archive(str(output / "MiniGridExplorer-Windows-CPU"),
                                      "zip", output, folder.name))
    paths = [archive]
    for name in ("technical-report.pdf", "demo.mp4"):
        destination = output / name
        shutil.copy2(submission / name, destination)
        paths.append(destination)
    manifest = {"assets": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths},
                "checkpoint_sha256": expected["checkpoint_sha256"],
                "memory_frame": info["memory_frame"], "training_seed": info["training_seed"],
                "selected_steps": info["selected_steps"],
                "confirmation_episodes": 5700, "report_pages": 5, "demo_seconds": 240}
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in paths), encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
