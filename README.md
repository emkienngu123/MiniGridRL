# MiniGrid Explorer

A reproducible reinforcement-learning exploration project prepared by **Kiên Nguyễn** for the **Atherlabs Reinforcement Learning Engineer Technical Test**, built directly on the official Farama MiniGrid repository. A PPO policy explores procedural rooms and corridors using a 7 x 7 partial view and a map reconstructed only from its own observations. A moving patrol makes survival part of the task.

This project extends **[Farama Foundation's MiniGrid](https://github.com/Farama-Foundation/Minigrid)**, starting from upstream commit [`7f585379`](https://github.com/Farama-Foundation/Minigrid/commit/7f5853792ea7f04f75614661006d87fcf949f26d). The upstream Git history, [Apache 2.0 license](LICENSE), and [citation metadata](CITATION.cff) are retained. The original README is in [README.minigrid.md](README.minigrid.md); assignment code is under `exploration/`. See [attribution and dependency licenses](ATTRIBUTION.md).

The final submission is available in [GitHub release v1.0.1](https://github.com/emkienngu123/MiniGridRL/releases/tag/v1.0.1): a [five-page technical report](submission/final/technical-report.pdf), [four-minute demo](submission/final/demo.mp4), and a [portable Windows CPU build](https://github.com/emkienngu123/MiniGridRL/releases/download/v1.0.1/MiniGridExplorer-Windows-CPU.zip). The primary study contains **5,700 confirmation episodes on predeclared seeds 40000-40099**, including the completed aligned reward ablation. [Start here](submission/final/START-HERE.md) for the evaluator workflow; [edit/rebuild the LaTeX report](report/README.md) or inspect the [design rationale audit](report/design-rationale-audit.md). Historical studies and artifacts are preserved separately.

![Interactive inspector](docs/ui.png)

## Try the saved policy

The default release model is **aligned PPO, training seed 33**, selected at 3.75 million transitions using the highest validation AUC (0.889863) among the aligned full-reward runs. Selection did not use confirmation or demonstration scores. The source includes all three full-reward aligned policies in [checkpoints/aligned](checkpoints/aligned) and all three aligned reward-ablation policies in [checkpoints/aligned-no-bonuses](checkpoints/aligned-no-bonuses).

**Windows, without Python:** download the [CPU ZIP](https://github.com/emkienngu123/MiniGridRL/releases/download/v1.0.1/MiniGridExplorer-Windows-CPU.zip), extract the complete archive, and open **Launch UI.bat**. **Smoke test.bat** verifies saved-policy loading and CPU inference. Keep the EXE beside `_internal/`. The archive includes its own start guide, report, demo, model metadata, and third-party licenses. No GPU, internet connection, or retraining is required to run it.

**Source checkout:** after installing the CPU requirements below, use the explicit current checkpoint:

```powershell
python -m exploration smoke --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
python -m exploration ui --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
```

Source UI commands without a checkpoint can still find the preserved historical `checkpoints/demo.zip`; the examples above and the release build explicitly use the current aligned policy. Checkpoint metadata records the observation version, memory frame, selected training step, configuration, dependency versions, and SHA-256.

## Install from source

Use **Python 3.11** (verified with 3.11.15). Run commands from this repository's root. A complete pinned CPU runtime, including test and artifact tools, is supplied:

```powershell
git clone https://github.com/emkienngu123/MiniGridRL.git
cd MiniGridRL
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock.txt
.\.venv\Scripts\python.exe -m exploration smoke --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
.\.venv\Scripts\python.exe -m exploration ui --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
```

Subsequent `python` commands assume this virtual environment is active; otherwise use `.\.venv\Scripts\python.exe` explicitly.

Verified core versions: **minigrid 3.2.0** (this repository), **gymnasium 1.4.0**, **stable-baselines3 2.9.2**, **torch 2.11.0** (+cpu or CUDA 12.8), and **numpy 2.4.6**. The lock file's `-e .` installs this checkout as the `minigrid` package, including the exploration code. Use that source installation; a separate PyPI MiniGrid install can replace it and hide this project's extensions.

For GPU training, use `requirements-cuda.lock.txt` instead. It pins PyTorch 2.11.0 with CUDA 12.8; the NVIDIA driver must support that runtime. The CPU package is built in a separate environment to exclude CUDA libraries. Other dependency versions are pinned in the corresponding lock file. Linux source execution is supported by the libraries; the delivered native artifact is Windows-specific.

## Checkpoint locations

Every saved model has a `.zip` and matching `.json` sidecar. The curated models are committed; training directories are local only.

| Location | Purpose |
|---|---|
| [checkpoints/aligned/](checkpoints/aligned) | Full-reward aligned PPO, seeds 11/22/33; seed 33 is the release default |
| [checkpoints/aligned-no-bonuses/](checkpoints/aligned-no-bonuses) | Aligned PPO without seen/visit bonuses, seeds 11/22/33 |
| `checkpoints/main-{11,22,33}.zip` | Original start-relative map policies |
| `checkpoints/no_memory-{11,22,33}.zip` | No-map policies |
| `checkpoints/no_novelty-{11,22,33}.zip` | Original-map policies without seen/visit bonuses |
| [checkpoints/demo.zip](checkpoints/demo.zip) | Historical default retained for compatibility |
| `runs/` | Ignored local training outputs: best/latest/final weights, logs, and validation history |

## Environment and agent interface

```python
from exploration.memory import make_env

env = make_env("easy")
observation, info = env.reset(seed=123)
observation, reward, terminated, truncated, info = env.step(2)
env.close()
```

Environment settings can also be overridden explicitly:

```python
env = make_env("medium", size=25, poi_count=4, max_steps=1200, monster_interval=3)
observation, info = env.reset(seed=123)
env.close()
```

Actions are `0=left`, `1=right`, `2=forward`, `3=wait`. Turn/forward transitions, the grid, wall visibility, observation encoding, and RGB rendering use MiniGrid. No doors, pickup, or combat mechanics are required.

| Preset | Size | POIs | Action budget | Monster period |
|---|---:|---:|---:|---:|
| Easy | 15 x 15 | 2 | 512 | 3 |
| Medium | 21 x 21 | 3 | 1,024 | 2 |
| Hard | 27 x 27 | 5 | 1,536 | 2 |

Map generation carves a connected depth-first maze, rectangular rooms, and sparse extra links. It checks connectivity and reachable placements. Separate seeded streams generate terrain and monster motion. Fixed seed plus actions reproduces an episode. The monster starts at least six traversable steps away, moves orthogonally, and avoids reversing when alternatives exist. It can occupy POIs without destroying them. Agent movement and fatal contact are processed before a scheduled monster move and a second contact check. Death terminates; exhausting the budget truncates. Discovering all POIs does not end exploration.

The raw environment returns MiniGrid's symbolic image, heading, mission, and four feedback values: movement succeeded, wall contact, monster contact, and remaining-budget fraction. The policy wrapper returns:

| Key | Shape | Contents |
|---|---|---|
| `local` | 5 x 7 x 7 | One-hot unseen, empty, wall, POI floor, monster |
| `memory` | 7 x 55 x 55 | Observed mask, walls, floor, discovered POIs, clipped occupancy count, currently visible monsters, agent |
| `scalars` | 8 | Heading one-hot and four feedback values |

All arrays are float32 in [0,1]. Memory coordinates are relative to the start. Mapping consumes only observations, actions, and movement feedback; its class has no environment reference. Occupancy counts include stationary actions and are clipped at ten for policy input. Threat occupancy is cleared on every observation. Static POI discoveries remain remembered. Supported odd grid sizes are 9 through 27. POIs must be 1 through grid size; budgets and monster periods must be positive.

`info` contains only observable reward components. Ground-truth denominators and evaluator statistics are accessed by the evaluator, never inserted into observations. Agents implement `reset(seed)` and `act(observation)` and receive no environment handle.

## Reward and architecture

```text
reward = 0.02 * new_visible_floor
       + 0.01 * first_occupied_cells
       + 1.00 * new_visible_POIs
       - 0.001
       - 0.02 * wall_contact
       - 1.00 * monster_contact
```

Each discovery is credited once per episode using observable history. Reset initializes discovery state without issuing rewards. No reward clipping or running normalization is used. PPO retains its standard advantage normalization. Tests cover repeated turns, waits, revisits, re-sightings, and hidden-map modifications. Legitimately revealing a previously unseen region by turning can earn reward once. Aggressive discovery versus survival and stationary/short-loop behavior remain measurable failure modes; return alone is not a success metric.

Exploration depends on both immediate perception and which regions were already explored. We use MiniGrid's symbolic view for nearby geometry and currently visible threats, plus a map reconstructed from observable movement and sightings for spatial history. The policy uses a local CNN (16/32 channels, 64 output features), a map CNN (32/64/64 stride-two channels, 128 output features), and eight scalar features. Separate actor and critic heads each have two 128-unit layers. The policy is feed-forward; history is represented by the auditable observation-derived map. The [architecture rationale](docs/architecture-rationale.md) connects each choice to the task, separates experimental findings from engineering defaults, and provides a report-ready explanation. Two branches and the exact widths have not been established as optimal; our controlled evidence is strongest for the aligned map readout.

With `memory_frame: agent_aligned`, the feature extractor centers the agent and rotates its remembered map so forward points upward before applying the existing CNN. The default `start_aligned` mode preserves previous policy behavior. Both modes use the same observation interface, stored history, and network architecture. The [memory alignment guide](docs/memory-alignment-review.md) explains geometry, compatibility, and verification. Train aligned policies individually using `configs/memory-aligned.yaml`; the matrix runner rejects this frame to protect the original results and submission from automatic evaluation and export.

The final confirmation uses 15 selected PPO checkpoints (five conditions x three training seeds), random movement, and frontier exploration on identical reserved maps and budgets:

| Agent | Coverage | Visited | Coverage AUC | POIs | Survival |
|---|---:|---:|---:|---:|---:|
| Random | 43.9% | 11.7% | 0.386 | 45.4% | 29.6% |
| Frontier | 91.5% | 58.5% | 0.869 | 92.1% | 33.7% |
| PPO, original map | 68.9% | 37.4% | 0.617 | 68.0% | 74.7% |
| PPO, no map | 78.6% | 48.2% | 0.701 | 79.4% | 75.0% |
| PPO, no bonuses, original map | 42.0% | 9.7% | 0.377 | 41.8% | 60.1% |
| PPO, no bonuses, aligned map | 59.6% | 19.1% | 0.547 | 59.2% | 56.8% |
| **PPO, aligned map** | **90.8%** | **65.6%** | **0.825** | **91.3%** | **72.1%** |

Values weight Easy, Medium, and Hard equally. Removing seen/visit bonuses reduces aligned coverage by 31.2 percentage points. Full-minus-no-bonus gains are +16.2 / +35.8 / +41.6 points on Easy / Medium / Hard, with positive paired-map 95% intervals. Alignment also improves coverage over original-map and mapless PPO. Frontier has higher coverage AUC; its per-preset endpoint coverage intervals against aligned PPO include zero. Aligned Hard survival is 62.7%, versus 70.7% mapless and 25.0% frontier, so exploration gains do not imply uniformly better safety.

See the [confirmation report](results/aligned-reward-followup/report.md), [raw measurements and paired intervals](results/aligned-reward-followup/summary.json), and [verification](results/aligned-reward-followup/verification.json). Earlier seeds 20000-20099 were exposed during investigation; old-model results on 30000-30029 also informed diagnosis. The older [alignment comparison](results/memory-alignment-comparison/report.md) remains archived with its original bytes; the final report supersedes its freshness claim. Seeds 40000-40099 were reserved before the new ablation/evaluation, with validation-only checkpoint selection and an audit of prior recorded episodes. These confirmation maps are now observed; subsequent model development needs new confirmation maps.

## Training

Train the headline aligned model and its matched reward ablation from fresh weights:

```powershell
python -m exploration train --config configs/memory-aligned.yaml --seed 33 --device cuda
python -m exploration train --config configs/aligned-no-bonuses.yaml --seed 33 --device cuda
```

Repeat with seeds 11 and 22 for the other runs. These commands train for five million transitions each and write separate `runs/main_agent_aligned-33` and `runs/no_novelty_agent_aligned-33` directories. To resume an interrupted aligned run:

```powershell
python -m exploration train --config configs/memory-aligned.yaml --seed 33 --device cuda --resume runs/main_agent_aligned-33/latest.zip
```

Do not reuse an existing output folder for a new run. `--resume` treats the configured timesteps as a total target. PPO rounds to a complete rollout. Resume preserves optimizer weights and step count but restarts environment sampling; it is not bitwise continuation. Incompatible observation versions, action/observation spaces, missing sidecars, and invalid checkpoint checksums are rejected.

Default PPO configuration: 16 subprocess environments, 256 steps/environment, batch 256, four epochs, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, clipping 0.2, entropy coefficient 0.01, target KL 0.03, value coefficient 0.5, gradient norm bound 0.5. Training samples Easy and Medium with equal probability. The YAML supports all environment-independent PPO parameters, seed partitions, reward weights, device, worker backend, output folder, and evaluation cadence. UI/`make_env` expose environment parameter overrides.

### Historical matrix workflow

The original nine-run start-relative-map matrix is retained for historical reproduction. It is separate from the aligned model commands and final confirmation workflow:

```powershell
python -m exploration matrix
python scripts/run_matrix.py --device cuda
```

This trains `main`, `no_memory`, and `no_novelty` for training seeds 11, 22, and 33 to five million transitions each. No-memory zeros the policy's memory branch while retaining internal reward bookkeeping. No-novelty zeros seen-floor and first-visit rewards. Validation occurs every 250,000 transitions plus the final rollout. The selected checkpoint maximizes validation coverage AUC, then POI discovery, then survival. Validation RNG state is restored so evaluation does not perturb training randomness.

The historical runner records progress in `results/full-status.json`; its per-run output is under `runs/<variant>-<seed>/`. Each folder contains `config.yaml`, `console.log`, `logs/progress.csv`, TensorBoard events, `training_episodes.csv`, `validation.csv`, validation episode results, `best/latest/final.zip` plus JSON sidecars, and `runtime.json`. Inspect TensorBoard with `tensorboard --logdir runs`.

Run `python scripts/status.py` for a fresh per-run transition count. Training directories are protected by process locks. CUDA subprocess failures retry from the latest saved checkpoint at most twice; the status file records failure if recovery is exhausted.

In that historical workflow, the matrix runner performs its original held-out evaluation and artifact export after the requested runs complete. Those outputs are not the final aligned confirmation artifacts. It does not tune on held-out results. `--skip-artifacts` stops after evaluation. Use `--tag` with a different value to keep a smaller diagnostic matrix separate from the full study.

## Reproduce the final confirmation without training

The selected models and frozen evidence are committed, so the final comparison does not require `runs/`, a GPU, or retraining:

```powershell
python scripts/reproduce_confirmation.py --verify-only
python scripts/reproduce_confirmation.py --count 100 --output results/confirmation-reproduced
```

The second command uses CPU workers and checks every non-timing episode metric against the archived 5,700 episodes. Use `--count 1` for a 57-episode rehearsal. Choose a new output directory each time. Measured inference latency varies with hardware/load and is excluded from exact reproduction checks.

The additional aligned no-bonus runs use [configs/aligned-no-bonuses.yaml](configs/aligned-no-bonuses.yaml), seeds 11/22/33, and the same five-million-transition budget. Their archived histories and frozen protocol are under `results/aligned-reward-followup/`; [follow-up documentation](docs/aligned-reward-followup.md) explains the experiment. The full study has 15 runs, approximately 75 million transitions. Exact network widths, reward weights, and PPO settings are engineering choices with stated reasoning, not claimed optima.

## Evaluation and metrics

Inclusive ranges: training 0-9,999; validation 10,000-10,049; original test 20,000-20,099; historical alignment comparison 30,000-30,099; final reserved confirmation 40,000-40,099. The original test and part of the alignment set informed development; the final protocol addresses that exposure. Training run seeds and environment map seeds serve different roles. The random baseline uses three independent policy seeds; the deterministic frontier baseline runs once per map. All agents receive identical information and budgets. Frontier BFS uses observed floor only, fixed tie-breaking, and visible-threat avoidance; it is a fair strong baseline, not a full-map oracle.

```powershell
python -m exploration evaluate --agents random frontier --split validation --count 10 --output results/baselines
python -m exploration evaluate --config results/aligned-reward-followup/confirmation-config.yaml --agents random frontier ppo --checkpoints checkpoints/aligned/main_agent_aligned-11.zip checkpoints/aligned/main_agent_aligned-22.zip checkpoints/aligned/main_agent_aligned-33.zip --split test --output results/aligned-example
python -m exploration evaluate --agents ppo --checkpoints checkpoints/aligned/main_agent_aligned-33.zip --split train --count 100 --output results/train-diagnostic
python -m exploration replay --agent ppo --checkpoint checkpoints/aligned/main_agent_aligned-33.zip --seed 10000 --preset medium --output results/replay
```

The final confirmation uses all 100 reserved seeds on Easy, Medium, and Hard. Hard is larger than any training map. For aligned policies, Hard also tests generalization to explored cells farther from the agent than those typically present on training maps. A train/test diagnostic uses map seeds from the training partition with the same evaluator; training reward is not substituted for a generalization metric.

| Metric | Definition |
|---|---|
| Revealed coverage | Unique observed traversable cells / true reachable traversable cells |
| Visited coverage | Unique occupied cells / true reachable traversable cells |
| Coverage AUC | Trapezoidal coverage over the full action budget, divided by budget |
| POI discovery | Unique visible POIs, and fraction of generated POIs |
| Survival/completion | Reaches the configured action budget without death |
| Threat interactions | Fatal monster contact count and termination reason |
| Wall contacts | Blocked forward-action count and count / actions |
| Movement efficiency | New occupied cells after reset / actions |
| Repeated-state rate | Fraction returning to a (relative position, heading) seen within 16 actions |
| Loop rate | Repeated state within 16 actions with no new floor or POI discovery |
| Latency | Action selection plus wrapper observation/memory preprocessing; mean, median, p95 |

Coverage/discovery curves hold final values after death through the configured budget, avoiding survivor bias. Engine simulation and rendering are excluded from inference latency. The first action is included. All saved policies are evaluated on CPU with one PyTorch thread; runtime/platform provenance is in the summary.

Episode rows contain their own latency mean, median, and p95. The summary's `inference_step_mean_ms` weights episode means by executed actions to obtain the overall per-step mean; ordinary metric means and deviations describe episodes.

Outputs: `episodes.csv` (every episode), `curves.csv` (every 32 actions plus final budget), `summary.json` (means, episode standard deviations, individual run means, run-mean standard deviations, paired-map bootstrap intervals, seed and checkpoint provenance). Paired comparisons average runs per map before resampling common maps; these intervals quantify map variation, not complete uncertainty across training seeds. A single deterministic run has no invented between-run standard deviation.

The historical matrix exporter evaluated 100 training-partition seeds and wrote `generalization.json`: test-minus-training-partition gaps and Hard-minus-Medium gaps for coverage, AUC, discoveries, and survival. This diagnostic does not imply that every sampled training-partition map was encountered by every training run. Curated training histories and selected checkpoints are exported alongside the results; `checkpoints/index.json` separates selected steps from the completed training budget.

The research goal is improvement over random exploration with meaningful discovery and survival. Results against the stronger frontier baseline are reported honestly. A higher training reward or an attractive selected video alone does not establish learning.

## UI, replay, and submission artifacts

The Tkinter UI exposes seed entry/randomization; preset, size, POI count, action limit and patrol interval; baseline/checkpoint selection; start, pause, reset, selected-agent single-step; speed; and live coverage, visited coverage, POIs, contacts, and loops. Parameter and agent changes apply on reset. The evaluator ground-truth view is clearly labeled and never fed to the policy. All policy inference is CPU-based.

Replay exports `episode.mp4`, `replay.json` with every action and checkpoint checksum, and `final.png`. Playback skips four simulation actions per video frame by default; the complete trace is retained. The two synchronized panels show ground truth and observation-derived memory.

Current handoff files are in [submission/final](submission/final/START-HERE.md); previous `submission/` and `submission/aligned/` outputs remain historical. Rebuild the current report, captioned demo, and Windows CPU package from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File report/build.ps1 -Python python
python scripts/export_aligned_demo.py --ui-video submission/aligned/ui-recording.mp4
python scripts/build_windows.py --checkpoint checkpoints/aligned/main_agent_aligned-33.zip --name MiniGridExplorerAligned --dist-dir dist/final --work-dir build/final --submission-dir submission/final
python scripts/prepare_release.py
python scripts/verify_artifacts.py --submission submission/final
```

Build the EXE using an activated environment with the pinned **CPU-only** runtime. PyInstaller outputs an executable folder, ZIP, checksum, bundled checkpoint/metadata, licenses, report, demo, and start guide. The build runs the saved-model smoke test. The published binary was additionally checked after ZIP extraction with Python environment variables removed and PATH restricted to Windows system directories; its UI was exercised on the Windows host. This is an environment/extraction rehearsal, not a claim that another physical computer was tested.

The video is four minutes, 1280 x 720 at 12 fps, with explanatory captions and no audio. It includes actual packaged-UI controls, exploration/discovery, a near-median Hard example (map 10039: 89.9% coverage, 5/5 POIs, survival), and a slowed one-cell monster pass. The original map-10000 turning loop appears as a failure of the same checkpoint. The clip-selection rule and 50-map audit are in [hard-example-selection](submission/final/hard-example-selection/README.md). Baseline and exploration-reward comparisons use the full recorded study; long loops are labeled and fast-forwarded. Replay action traces and model hashes accompany the clips. The primary results come from the reserved confirmation study, not selected demonstrations. [Farama attribution and licenses](ATTRIBUTION.md) are retained in both source and package.

## Tests and checks

```powershell
python -m pytest tests_exploration -q
python -m ruff check exploration tests_exploration scripts
python -m exploration smoke --checkpoint checkpoints/aligned/main_agent_aligned-33.zip
```

The focused suite checks reset/trajectory reproducibility, connected maps and safe placements, native movement/odometry, occlusion, monster timing/walls, both fatal-contact orders, truncation, reward math and one-time discovery, hidden-information invariance, memory reset and threat clearing, fair baseline determinism, seed partitions, AUC after death, run/episode variation, paired comparisons, and checkpoint round trips. Upstream tests remain in `tests/`.

## Limitations and attribution

Exact compass/contact odometry simplifies localization. Only one generator family and patrol rule are implemented. Fixed memory supports map sizes through 27. A deterministic policy may loop despite stochastic training; loop rates and failure replays expose that. Selected checkpoints may precede the final five-million-transition budget; completed-run counts and selected checkpoint steps are recorded separately.

Observed failures and tradeoffs include:

- **Turning loops:** the release checkpoint on Hard map 10000 reveals only 36.9% of floor despite surviving the full budget. See its [action trace](submission/final/replays/aligned-hard/replay.json).
- **Early threat contact:** on Hard map 30000, the same policy dies at step 122, with 58.7% coverage and 3/5 POIs. See the [contact replay](submission/final/replays/aligned-contact/replay.json).
- **Hard survival tradeoff:** aligned PPO survives 62.7% of episodes versus 70.7% for mapless PPO, despite higher coverage.
- **Exploration speed:** frontier has higher coverage AUC on every preset; learned exploration does not uniformly outperform the classical baseline.

See [ATTRIBUTION.md](ATTRIBUTION.md) for upstream revision, library licenses, and research references. Future work includes uncertain odometry, recurrent-policy comparisons, richer map distributions, and tracking beliefs about hidden threats.

## Upstream citation

MiniGrid is the environment foundation; its authors and maintainers deserve credit separately from this project's exploration environment, memory representation, and experiments. The citation recommended in the upstream README is:

```bibtex
@inproceedings{MinigridMiniworld23,
  author = {Maxime Chevalier{-}Boisvert and Bolun Dai and Mark Towers and Rodrigo Perez{-}Vicente and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title = {Minigrid {\&} Miniworld: Modular {\&} Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  booktitle = {Advances in Neural Information Processing Systems 36, New Orleans, LA, USA},
  month = {December},
  year = {2023}
}
```
