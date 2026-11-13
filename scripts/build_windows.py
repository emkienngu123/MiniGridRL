from __future__ import annotations

import _tkinter
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Build a portable Windows CPU-inference folder")
    parser.add_argument("--checkpoint", default="checkpoints/demo.zip")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--name", default="MiniGridExplorer")
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--work-dir", default="build")
    parser.add_argument("--submission-dir", help="Include the public demo and start guide")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Build name must contain only letters, numbers, underscores, or hyphens")
    dist_dir, work_dir = (Path(value).resolve() for value in (args.dist_dir, args.work_dir))
    for path in (dist_dir, work_dir):
        if path == ROOT or ROOT not in path.parents:
            raise ValueError("Build output/work directories must be inside the repository")
    submission = Path(args.submission_dir).resolve() if args.submission_dir else None
    if submission and (ROOT not in submission.parents or not submission.is_dir()):
        raise ValueError("Submission inputs must be an existing subdirectory of the repository")
    checkpoint = Path(args.checkpoint).resolve()
    metadata = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not metadata.is_file():
        raise ValueError("Checkpoint .zip and .json are required")
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError("Build with a separate CPU-only PyTorch environment; see README")
    from exploration.policy import load_checkpoint

    torch.set_num_threads(1)
    model, checkpoint_info = load_checkpoint(checkpoint, "cpu")
    del model
    tcl_root = Path(sys.base_prefix) / "tcl"
    tcl = tcl_root / f"tcl{_tkinter.TCL_VERSION}"
    tk = tcl_root / f"tk{_tkinter.TK_VERSION}"
    if not (tcl / "init.tcl").is_file() or not (tk / "tk.tcl").is_file():
        raise RuntimeError("A complete Windows Python installation with matching Tcl/Tk scripts is required")
    stage = work_dir / "resources" / "checkpoints"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, stage / "demo.zip")
    shutil.copy2(metadata, stage / "demo.json")
    build_env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(work_dir / "pyinstaller-cache"))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
                    "--name", args.name, "--paths", str(ROOT),
                    "--distpath", str(dist_dir), "--workpath", str(work_dir / "pyinstaller"),
                    "--specpath", str(work_dir / "specs"),
                    "--additional-hooks-dir", str(ROOT / "scripts" / "pyinstaller_hooks"),
                    "--hidden-import", "tkinter", "--hidden-import", "_tkinter",
                    "--add-data", f"{tcl};_tcl_data", "--add-data", f"{tk};_tk_data",
                    "--collect-all", "stable_baselines3", "--collect-all", "minigrid",
                    "--collect-submodules", "exploration", "--hidden-import", "PIL._tkinter_finder",
                    "--add-data", f"{stage};checkpoints",
                    "--exclude-module", "tensorboard", "--exclude-module", "matplotlib.tests",
                    "--exclude-module", "pytest", "--exclude-module", "reportlab",
                    "--exclude-module", "pypdfium2", "--exclude-module", "IPython",
                    str(ROOT / "scripts" / "frozen_entry.py")], cwd=ROOT, check=True, env=build_env)
    folder = dist_dir / args.name
    executable = folder / f"{args.name}.exe"
    if (tcl_root / "tcl8").is_dir():
        shutil.copytree(tcl_root / "tcl8", folder / "_internal" / "tcl8", dirs_exist_ok=True)
    if not (folder / "_internal" / "_tcl_data" / "init.tcl").is_file() or not (folder / "_internal" / "_tk_data" / "tk.tcl").is_file():
        raise RuntimeError("Portable build is missing Tcl/Tk runtime scripts")
    (folder / "Launch UI.bat").write_text(
        f'@echo off\ncd /d "%~dp0"\n"{args.name}.exe" ui\n', encoding="ascii")
    (folder / "Smoke test.bat").write_text(
        f'@echo off\ncd /d "%~dp0"\n"{args.name}.exe" smoke --checkpoint "_internal\\checkpoints\\demo.zip"\npause\n', encoding="ascii")
    build_info = {
        "name": args.name, "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "checkpoint_source": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else checkpoint.name,
        "checkpoint_sha256": checkpoint_info["sha256"], "memory_frame": checkpoint_info["memory_frame"],
        "training_seed": checkpoint_info["seed"], "selected_steps": checkpoint_info["training_steps"],
        "validation_score": checkpoint_info.get("validation_score"),
        "scientific_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (ROOT / "exploration").glob("*.py")},
        "dependencies": {key: importlib.metadata.version(key) for key in
                         ("stable-baselines3", "minigrid", "gymnasium", "numpy", "pillow", "pyinstaller")},
    }
    (folder / "build-info.json").write_text(json.dumps(build_info, indent=2), encoding="utf-8")
    (folder / "PORTABLE-START-HERE.txt").write_text(
        f"{args.name}: Windows CPU inference\n\n"
        "Extract the entire ZIP before launching. Keep the _internal folder beside the EXE.\n"
        "Double-click Launch UI.bat (or the EXE) to open the interactive explorer.\n"
        "Double-click Smoke test.bat for a headless saved-model check.\n"
        "Python, a GPU, retraining, and internet access are not required.\n\n"
        f"Default PPO: {checkpoint_info['memory_frame']}, training seed {checkpoint_info['seed']}, "
        f"selected at {checkpoint_info['training_steps']:,} transitions.\n"
        "Selection uses validation AUC, not demonstration or test-map performance.\n"
        "Use seed/difficulty/agent controls, Generate / reset map, Start, Pause, and Single step.\n"
        "Evaluator view is ground truth for inspection only; the policy uses partial observations.\n"
        "The history display remains start-relative; the PPO map input is aligned internally.\n"
        "See build-info.json for model identity and third-party-licenses for attribution.\n",
        encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", folder / "LICENSE-MiniGrid.txt")
    if (Path(sys.base_prefix) / "LICENSE.txt").exists():
        shutil.copy2(Path(sys.base_prefix) / "LICENSE.txt", folder / "LICENSE-Python.txt")
    shutil.copy2(ROOT / "ATTRIBUTION.md", folder / "ATTRIBUTION.md")
    for name in ("README.md", "requirements-cpu.lock.txt"):
        shutil.copy2(ROOT / name, folder / name)
    licenses = folder / "third-party-licenses"
    licenses.mkdir(exist_ok=True)
    for distribution in importlib.metadata.distributions():
        for file in distribution.files or []:
            if any(word in file.name.lower() for word in ("license", "copying", "notice")) and str(file).endswith((".txt", ".md", "LICENSE", "COPYING", "NOTICE", ".rst")):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    destination = licenses / distribution.metadata["Name"] / str(file).replace("..", "_")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    if submission:
        stale = folder / "technical-report.pdf"
        assert stale.resolve().is_relative_to(ROOT)
        stale.unlink(missing_ok=True)
        assets = {}
        for name in ("demo.mp4", "START-HERE.md"):
            source = submission / name
            if not source.is_file():
                raise ValueError(f"Missing final submission input: {source}")
            shutil.copy2(source, folder / name)
            assets[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (folder / "SUBMISSION-ASSETS.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
    # This is the same headless CPU smoke the evaluator can run without Python.
    result = subprocess.run([str(executable), "smoke", "--checkpoint",
                             str(folder / "_internal" / "checkpoints" / "demo.zip")],
                            cwd=folder, check=True, text=True, capture_output=True)
    (folder / "smoke-result.json").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if not args.no_zip:
        archive = shutil.make_archive(str(dist_dir / f"{args.name}-Windows-CPU"), "zip", dist_dir, folder.name)
        digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        Path(archive).with_suffix(".zip.sha256").write_text(f"{digest}  {Path(archive).name}\n", encoding="ascii")
        print(archive, flush=True)


if __name__ == "__main__":
    main()
