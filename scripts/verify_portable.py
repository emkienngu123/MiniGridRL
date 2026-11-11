"""Extract and smoke-test the final Windows ZIP without Python or GPU environment settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="dist/final/MiniGridExplorer-Windows-CPU.zip")
    parser.add_argument("--extract", default="tmp/final portable extraction")
    parser.add_argument("--output", default="submission/final/portable-verification.json")
    args = parser.parse_args()
    archive = (ROOT / args.zip).resolve()
    target = (ROOT / args.extract).resolve()
    output = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT) or target.exists() or not output.is_relative_to(ROOT):
        raise ValueError("Use a new extraction directory and output within this repository")
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError(f"Archive contains an unsafe path: {name}")
        zipped.extractall(target)
    folders = list(target.iterdir())
    if len(folders) != 1 or not folders[0].is_dir():
        raise ValueError("Expected one portable application directory")
    folder = folders[0]
    info = json.loads((folder / "build-info.json").read_text())
    assert info["cuda_runtime"] is None and info["torch"].endswith("+cpu")
    assert info["memory_frame"] == "agent_aligned" and info["training_seed"] == 33
    model = folder / "_internal/checkpoints/demo.zip"
    assert sha256(model) == info["checkpoint_sha256"]
    assets = json.loads((folder / "SUBMISSION-ASSETS.json").read_text())
    for name, digest in assets.items():
        assert sha256(folder / name) == digest == sha256(ROOT / "submission/final" / name)
    env = {key: value for key, value in os.environ.items()
           if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    windows = Path(env.get("SYSTEMROOT", r"C:\Windows"))
    env.update(PATH=f"{windows / 'System32'};{windows}", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([str(folder / f"{info['name']}.exe"), "smoke", "--checkpoint", str(model)],
                            cwd=folder, env=env, text=True, capture_output=True, timeout=180, check=True)
    # Preserve complete stdout: smoke includes a diagnostic preamble before episode JSON.
    record = {"status": "passed", "archive": archive.name, "archive_sha256": sha256(archive),
              "archive_bytes": archive.stat().st_size, "checkpoint_sha256": info["checkpoint_sha256"],
              "torch": info["torch"], "memory_frame": info["memory_frame"],
              "training_seed": info["training_seed"], "selected_steps": info["selected_steps"],
              "bundled_assets_verified": assets, "path_contains_spaces": " " in str(target),
              "python_environment_removed": True, "path_windows_only": True,
              "cuda_devices_hidden": True, "smoke_exit_code": result.returncode,
              "smoke_stdout": result.stdout,
              "scope": "Fresh extraction and isolated environment on the development Windows host; not a separate physical machine"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "smoke_stdout"}, indent=2))


if __name__ == "__main__":
    main()
