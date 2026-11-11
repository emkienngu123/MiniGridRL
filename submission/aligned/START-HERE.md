# Current aligned-policy submission artifacts

- **Video:** `demo.mp4`, 4 minutes, 1280 x 720 at 12 fps, with explanatory captions and no audio.
- **Report:** `technical-report.pdf`, the current five-page LaTeX report, including implementation reasoning, seed-overlap disclosure, sensitivity results, and current delivery status.
- **Windows CPU package:** `../../dist/aligned/MiniGridExplorerAligned-Windows-CPU.zip`.
- **Verification:** `artifact-checks.json`, `portable-verification.json`, and `ui-verification.json`.
- **Identity and hashes:** `release-manifest.json`, `SHA256SUMS.txt`, and `demo-provenance.json`.

The default policy in the new demo and package is `main_agent_aligned-33.zip`, selected at 3,750,000 transitions by the highest validation AUC among aligned seeds 11/22/33 (0.889863). The checkpoint SHA-256 is `abf8a85a5ea4d7a25b1ed93363da5a36fc37d37ea504c2d3e6f4e126506d91ab`.

## Run the Windows package

Extract the **complete** ZIP and keep the EXE beside its `_internal` folder. Open `Launch UI.bat` or `MiniGridExplorerAligned.exe`. `Smoke test.bat` checks saved-model loading and CPU inference. Python, CUDA, network access, and retraining are unnecessary. A package-specific `PORTABLE-START-HERE.txt` is included inside the ZIP.

The extracted executable was smoke-tested with Python/virtual-environment variables removed, PATH restricted to Windows system directories, and an extraction directory containing spaces. The packaged UI was also launched and exercised on this Windows host. This is an extraction/environment rehearsal, not a claim that another physical computer was tested.

## Video chapters

| Time | Content |
|---|---|
| 0:00 | Task, representation, validation-selected model |
| 0:12 | Actual packaged UI controls and map/policy selection |
| 0:44 | Procedural Easy/Medium/Hard maps |
| 0:56 | Aligned PPO on Easy, validation map 10000 |
| 1:20 | Aligned PPO on Medium, validation map 10000 |
| 1:48 | Aligned PPO on Hard, validation map 10000 |
| 2:12 | Frontier on the same Medium map |
| 2:30 | Aligned seed-33 threat contact, selected Hard map 30000 |
| 2:48 | Recorded 4,800-episode aggregate comparison |
| 3:10 | Prior seed exposure and the 70-map exclusion sensitivity |
| 3:24 | Explicitly labeled loop failure from a different aligned run, seed 11 |
| 3:40 | Portable launch and reproduction instructions |

The five principal demonstration episodes use the bundled seed-33 model or frontier. The loop example deliberately uses seed 11 and is labeled as a separate trained run. Examples illustrate behavior; aggregate claims come from the archived study, not these selected clips. The entire episode is compressed into each replay segment with its final outcome held for two seconds. `replays/*/replay.json` retains full action traces, model hashes, presets, and seeds. `ui-recording.mp4` is the real packaged-application recording; unused black capture padding is cropped during presentation.

## Reproduce the artifacts

From the repository root with the pinned CPU environment:

```powershell
$env:PYTHONPATH=(Get-Location).Path
..\.cpu-venv\Scripts\python.exe scripts/build_windows.py --checkpoint checkpoints/aligned/main_agent_aligned-33.zip --name MiniGridExplorerAligned --dist-dir dist/aligned --work-dir build/aligned
..\.cpu-venv\Scripts\python.exe scripts/export_aligned_demo.py --ui-video submission/aligned/ui-recording.mp4
..\.cpu-venv\Scripts\python.exe scripts/verify_artifacts.py --submission submission/aligned
```

This process does not train policies or rewrite the frozen original study. The parent `submission/` files and original `dist/MiniGridExplorer-Windows-CPU.zip` remain historical artifacts. The main README is being reviewed separately and has not been edited during this refresh. Files are prepared locally; publishing the ZIP as a GitHub release asset is a separate distribution step.
