from __future__ import annotations

import _tkinter
import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Build a portable Windows CPU-inference folder")
    parser.add_argument("--checkpoint", default="checkpoints/demo.zip")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    metadata = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not metadata.is_file():
        raise ValueError("Checkpoint .zip and .json are required")
    import torch
    if torch.version.cuda is not None:
        raise RuntimeError("Build with a separate CPU-only PyTorch environment; see README")
    tcl_root = Path(sys.base_prefix) / "tcl"
    tcl = tcl_root / f"tcl{_tkinter.TCL_VERSION}"
    tk = tcl_root / f"tk{_tkinter.TK_VERSION}"
    if not (tcl / "init.tcl").is_file() or not (tk / "tk.tcl").is_file():
        raise RuntimeError("A complete Windows Python installation with matching Tcl/Tk scripts is required")
    stage = ROOT / "build" / "resources" / "checkpoints"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, stage / "demo.zip")
    shutil.copy2(metadata, stage / "demo.json")
    build_env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(ROOT / "build" / "pyinstaller-cache"))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
                    "--name", "MiniGridExplorer", "--paths", str(ROOT),
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
    folder = ROOT / "dist" / "MiniGridExplorer"
    if (tcl_root / "tcl8").is_dir():
        shutil.copytree(tcl_root / "tcl8", folder / "_internal" / "tcl8", dirs_exist_ok=True)
    if not (folder / "_internal" / "_tcl_data" / "init.tcl").is_file() or not (folder / "_internal" / "_tk_data" / "tk.tcl").is_file():
        raise RuntimeError("Portable build is missing Tcl/Tk runtime scripts")
    (folder / "Launch UI.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\nMiniGridExplorer.exe ui\r\n', encoding="ascii")
    (folder / "Smoke test.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\nMiniGridExplorer.exe smoke --checkpoint "_internal\\checkpoints\\demo.zip"\r\npause\r\n', encoding="ascii")
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
    # This is the same headless CPU smoke the evaluator can run without Python.
    subprocess.run([str(folder / "MiniGridExplorer.exe"), "smoke", "--checkpoint",
                    str(folder / "_internal" / "checkpoints" / "demo.zip")], cwd=folder, check=True)
    if not args.no_zip:
        archive = shutil.make_archive(str(ROOT / "dist" / "MiniGridExplorer-Windows-CPU"), "zip", ROOT / "dist", folder.name)
        print(archive, flush=True)


if __name__ == "__main__":
    main()
