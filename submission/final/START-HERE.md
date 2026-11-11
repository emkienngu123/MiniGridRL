# MiniGrid exploration agent - final submission

The current release is [v1.0.0](https://github.com/emkienngu123/MiniGridRL/releases/tag/v1.0.0).

## Review in order

1. Read `technical-report.pdf` (five pages), covering all assignment topics and the reasoning behind the design.
2. Watch `demo.mp4` (four minutes, captioned, no audio). It shows the actual packaged UI, seeded exploration/discovery, threat contact, frontier behavior, aggregate results, reward ablation, and a labeled historical failure.
3. Download/extract `MiniGridExplorer-Windows-CPU.zip` from the release. Keep the entire executable folder together and run `Launch UI.bat`; `Smoke test.bat` checks CPU inference. No Python installation, GPU, network connection, or training is required.

The Windows package also contains this guide, the report, and the video. Its default saved model is **aligned PPO, seed 33**, selected at 3.75M transitions by validation AUC 0.889863. Model SHA-256: `abf8a85a5ea4d7a25b1ed93363da5a36fc37d37ea504c2d3e6f4e126506d91ab`. The model was not selected on demonstration or confirmation performance.

## Primary evidence

The confirmation protocol reserved seeds **40000-40099**, all three presets, before the new reward-ablation training/testing. Five PPO conditions x three trained seeds, three random policy seeds, and deterministic frontier give **5,700 episodes**. All models were selected on Easy/Medium validation; Hard is excluded from training/selection. Training comprised 15 runs, approximately 75M transitions in total.

Equal-preset coverage: aligned PPO 90.8%, mapless PPO 78.6%, original map 68.9%, aligned without exploration bonuses 59.6%, frontier 91.5%, random 43.9%. The reward-removal comparison reduces aligned coverage by 31.2 percentage points. Paired-map reward gains are positive on all presets. Frontier has higher coverage AUC, while aligned PPO survives more often than frontier; mapless PPO retains higher Hard survival. Exact weights and network dimensions remain engineering choices, not claimed optima.

Raw episodes/curves, frozen protocol and model hashes, validation/training records, and paired intervals are under `results/aligned-reward-followup/`. `verification.json` independently checks all 5,700 episodes, 188,100 curve rows, bootstrap intervals, and 15 model archives. The older 20000/30000 evaluation ranges informed development and remain disclosed; earlier artifacts are preserved in `submission/` and `submission/aligned/`. The confirmation maps are now observed; further model selection needs another reserved set.

## Reproduce without training

Clone [emkienngu123/MiniGridRL](https://github.com/emkienngu123/MiniGridRL), create a Python 3.11 environment, and install `requirements-cpu.lock.txt`. From the source repository root:

```powershell
python -m exploration smoke --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
python scripts/reproduce_confirmation.py --verify-only
python scripts/reproduce_confirmation.py --count 1 --output results/rehearsal
python scripts/reproduce_confirmation.py --count 100 --output results/confirmation-reproduced
```

Choose a new output folder for each reproduction. The short rehearsal evaluates 57 episodes, covering all models/baselines/presets; full reproduction checks all non-timing metrics against the 5,700 archived episodes. Timing varies with hardware/load. No ignored `runs/` directory is needed. Selected original/aligned/reward-ablation checkpoints are committed, with metadata and hashes.

The portable ZIP is verified after extraction in a path containing spaces, with Python variables removed and PATH restricted to Windows directories. The actual packaged UI controls are checked on the Windows host. This is a clean extraction/environment rehearsal, not a test on another physical machine. Read the release's verification and checksum assets for the final evidence.

## Video chapters

0:00 task/model; 0:12 packaged UI; 0:44 map generation; 0:56 Easy; 1:20 Medium; 1:48 Hard; 2:12 frontier; 2:30 threat contact; 2:48 confirmation results; 3:10 aligned reward ablation; 3:24 historical loop failure (different trained seed 11); 3:40 portable launch.

Qualitative examples use fixed validation map 10000 and explicitly selected historical contact/failure maps. They are illustrative, not the primary dataset. Complete action traces, checkpoint hashes, and clip provenance are retained. The displayed history is start-relative; PPO aligns its input internally. Ground-truth evaluator views are never policy inputs.

Farama Foundation MiniGrid is the upstream environment foundation. Its history, Apache-2.0 license, citation, and third-party attribution remain in the repository and portable package.
