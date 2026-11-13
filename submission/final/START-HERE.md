# MiniGrid Explorer - getting started

Download the app from the [latest release](https://github.com/emkienngu123/MiniGridRL/releases/latest), extract the complete ZIP, and open **Launch UI.bat**. Keep the executable and `_internal` folder together. **Smoke test.bat** checks saved-model loading and CPU inference. Python installation, a GPU, and retraining are not required.

Watch `demo.mp4` for the UI controls, exploration, POI discovery, a survived close pass, baseline behavior, and evaluation results. The technical report is supplied separately; it is not bundled with the app or the current public source.

The app loads the aligned PPO model trained with seed 33, selected at 3.75M transitions using Easy/Medium validation AUC. Model metadata and hashes are included. Demonstration maps do not select the model.

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

0:00 task/model; 0:12 packaged UI; 0:44 map generation; 0:56 Easy; 1:20 Medium; 1:48 Hard (near-median map 10039); 2:00 close pass (within the Hard replay); 2:24 frontier; 2:42 threat contact; 3:00 results; 3:22 exploration rewards; 3:36 failure (same checkpoint, map 10000); 3:44 limitations; 3:52 portable launch.

Easy, Medium, and frontier use fixed validation-numbered map 10000. The main Hard episode is map **10039**, selected for near-median coverage and a survived close pass after auditing all 50 Hard maps 10000-10049 with the already-fixed seed-33 checkpoint. Its coverage is **89.9%**, versus a **89.8%** median; it finds all five POIs and survives. Map 10010 is equally near the median, so this is not a unique closest-map result. Hard examples do not select the checkpoint or training parameters. The 50 audit episodes are qualitative-selection evidence, separate from the 5,700 primary episodes.

The 36-second main Hard segment slows the close pass at step 258 and marks it with gold rings in the evaluator view. Its own turning loops at steps 402-909 and 1181-1536 are explicitly labeled and fast-forwarded; the full trajectory remains in the source replay. The caption reports proximity and survival without attributing intentional avoidance. The trace has four post-step orthogonal adjacencies (258, 260, 1068, 1070); it does not establish eight separate passes. `hard-example-selection/` contains the population metrics, selection rule, and selected trajectory.

The original seed-33/map-10000 turning loop now appears in the failure section; its 36.9% coverage and 4/5 POIs are unchanged. Contact map 30000 remains a selected illustration. The older seed-11/map-30029 replay is retained on disk but is not shown in this cut. All clips illustrate behavior rather than supply aggregate evidence. Source-step timing is in `storyboard.json`, and `hard-replay-diagnostics.json` documents the original loop. The displayed history remains start-relative; PPO aligns its input internally. Evaluator positions and proximity highlights never enter policy inputs.

Farama Foundation MiniGrid is the upstream environment foundation. Its history, Apache-2.0 license, citation, and third-party attribution remain in the repository and portable package.
