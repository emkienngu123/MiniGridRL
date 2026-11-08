# Memory alignment: implementation review

Status: alignment implementation reviewed and verified by the reviewer. After execution was authorized, the implementing assistant passed the five matrix-guard tests and the full CPU-selected suite (115 passed, 33 CUDA cases deselected), then completed separate 8,192-transition CPU and capped-GPU training pilots. On November 7, 2026, after the user freed the GPU, all 148 tests passed including CUDA cases and all three full-length aligned runs completed (5,001,216 transitions each; approximately 3 hours 42 minutes total). The subsequently requested full comparison completed 4,800 fresh-test episodes, 450 additional training-partition diagnostic episodes, and 96 isolated CPU-latency episodes. See [the complete results and analysis](../results/memory-alignment-comparison/report.md). Training state is recorded in `results/memory-aligned-status.json`. The implementation branch is `feature/memory-alignment`, based on `investigation/memory`.

Changes are left uncommitted for IDE review. Run `git diff` for the complete implementation diff, including configuration, review notes, and tests.

## Review order and behavior

1. `exploration/policy.py`: `align_memory` and its use immediately before the memory CNN; checkpoint serialization and compatibility checks.
2. `exploration/config.py` and the training CLI: frame selection, labels, defaults, and output paths.
3. Agent/resume integration, evaluation/replay provenance, and the UI's policy-frame label.
4. `tests_exploration/test_memory_alignment.py`: prepared geometry, serialization, configuration, and regression tests.

The feature extractor consumes the unchanged observation dictionary. Its local CNN still produces 64 features; its memory CNN still produces 128; eight scalar features are concatenated into 200 features. No trainable parameters or layers are added. Both actor and critic receive features derived from the selected frame.

The transformation has no simulator reference. It uses the marker in map channel 6 for position and the heading one-hot in the scalar input. All seven map channels are gathered using the same integer coordinates on the input tensor's device. The result is a separate tensor; source observations, canonical memory, rewards, and evaluator tracking remain untouched. Currently invisible monsters remain absent because the memory tracker already clears that channel on each observation.

Nonzero normal observations have exactly one agent marker. All-zero observations are the supported no-memory ablation and remain zero under alignment. Arbitrary nonzero marker-removed probes are outside this contract: the earlier investigation's marker-removal probe must not be interpreted as a localization diagnostic for this new, marker-dependent transform.

## Coordinate example

Arrays use `[batch, channel, y, x]`. Heading indices are `0=east`, `1=south`, `2=west`, `3=north`. For output column `u` and row `v`, let `forward_offset = 27-v`, `right_offset = u-27`, `F = DIRECTIONS[heading]`, and `R = DIRECTIONS[(heading+1)%4]`. Read:

```text
source = agent_position + forward_offset * F + right_offset * R
```

For an agent at source `(10,20)` facing east:

| Source cell | Relationship | Output cell |
|---|---|---|
| `(10,20)` | Agent | `(27,27)` |
| `(11,20)` | Forward | `(27,26)` |
| `(10,21)` | Right | `(28,27)` |
| `(9,20)` | Behind | `(27,28)` |
| `(10,19)` | Left | `(26,27)` |

Out-of-bounds samples are unknown (all channels zero), with no wrapping or interpolation. A fixed 55-cell output has radius 27. Terrain within the supported 27-cell-wide maps fits at any legal agent location; deliberately larger synthetic source distributions may be clipped by that output window. The tests distinguish these cases. Heading scalars retain their existing absolute-heading meaning, and the local 7x7 view is unchanged.

## Configuration and compatibility

- `memory_frame` accepts `start_aligned` (default) and `agent_aligned`; `train --memory-frame` overrides YAML.
- SB3 serializes the frame through `features_extractor_kwargs`. The extractor constructor defaults to `start_aligned`, so archives without that argument retain their old behavior.
- Sidecars record the frame at the top level and inside configuration. Missing legacy fields mean `start_aligned`. Loading checks agreement between those fields, serialized constructor options, and actor/critic extractors. Saving rejects mismatched configuration before writing the archive.
- External observation shapes/keys and `OBSERVATION_VERSION` are unchanged. Equal tensor shapes do not make different input-frame semantics interchangeable.
- Resume checks seed, variant, and frame before constructing training environments. A legacy policy cannot be resumed as an aligned policy. A fresh aligned model is required for the planned experiment.
- Aligned policies are labeled `main_agent_aligned` (or the equivalent variant prefix). Automatic run names include that label, while explicit `--output` takes precedence. Legacy names such as `main` and `runs/main-11` remain valid.
- The UI continues to display canonical history and states the PPO input frame. Baselines continue consuming canonical observations. Episode/curve rows, evaluation provenance, and replay traces record the frame.

`configs/memory-aligned.yaml` copies the main experiment settings, changing only the frame and output folder. Train its seeds individually. The matrix runner explicitly rejects non-start-aligned configurations before creating status files, inspecting run folders, or launching any child process, regardless of tag or `--skip-artifacts`. This prevents it from reusing completed legacy runs and overwriting held-out results or rebuilding the submission. The retained original test partition must not be reused for selecting the redesign; a fresh held-out manifest is needed for a later final comparison.

## Verification status

Before review, the implementing assistant ran Python AST syntax parsing and Ruff only. The reviewer subsequently reported:

- All 101 alignment tests passed; the full project suite passed with 143 tests; Ruff was clean. Pytest used a reviewer-selected scratch directory because the default temporary directory had previously been denied.
- An independent real-environment audit ran 30 random-action episodes across Easy, Medium, and Hard: 8,557 states and 226,735 visible-cell comparisons, with zero mismatches between the current view and aligned memory.
- Agent markers remained centered, all four headings were covered, and no explored cells were lost. The largest observed offset was 22 cells, within the radius of 27.
- Legacy checkpoint behavior, resume rejection, isolated seed output naming, and matched configuration were verified.

These counts and geometry audit results are reviewer-reported. They describe the alignment implementation before the follow-up matrix guard. After the user authorized execution, the implementing assistant ran the five new guard cases successfully, then the CPU-selected full suite: 115 passed, 33 CUDA cases deselected to avoid an unnecessary competing CUDA workload. The guard cases check rejection with default/custom tags and either artifact setting, no filesystem changes or child dispatch on rejection, and continued legacy dispatch.

Alignment tests cover all four headings and multiple positions; channel consistency and padding; translation/rotation equivalence; supported map extents; CPU/CUDA batching and zero maps; unchanged inputs and parameter/state-dictionary structure; untrained checkpoint round trips and legacy defaults; mismatch rejection; labels/output paths; resume checks; and reward/baseline/reset regressions. CUDA cases skip when CUDA is unavailable.

The untrained checkpoint tests construct policies in temporary directories and never call `learn()`. Provenance tests perform one-step untrained-policy inference for evaluation and replay (with video encoding mocked). The existing full project suite includes short training tests, so permission to run focused alignment tests does not imply permission to run the entire suite.

## Experiment interpretation

This is an alignment experiment using the original 459,008-parameter map encoder. It isolates the map coordinate frame; it does not implement compact local/global inputs or reduce encoder capacity. A positive result would motivate testing a smaller encoder as a separate change. A negative result would weaken the frame-alignment hypothesis for this architecture and training budget, without proving that other memory designs cannot help.

Hard remains an extrapolation test. The reviewer noted explored-cell offsets around 19 cells on Medium and 22 or more on Hard; farther spatial regions of the input are an additional generalization axis beyond training map size. Report Medium and Hard separately when evaluating an approved trained model.

## Full training launch

Separate resource pilots are recorded in `results/memory-alignment-resource-fit/`: CPU and CUDA each completed 8,192 transitions with 16 dummy-vector environments, the original PPO batch/rollout settings, and one validation map per preset. CPU measured 264 transitions/second; GPU measured 754. PyTorch peaked at 172 MiB allocated and 298 MiB reserved under a 768 MiB allocator cap. System GPU memory rose by about 0.7 GiB during the GPU pilot, with at least 1,872 MiB free in one-second samples. Competing workloads affect these system-wide measurements. The allocator cap excludes CUDA context/driver allocations and does not cap compute utilization.

The pilots validate execution and current resource fit, not learning quality or a full-run performance guarantee. Full validation uses 50 maps per preset instead of one. On November 7, the RTX 4090 had about 22 GiB free and 3% utilization before launch. The full runs use the original 16 subprocess environments and original 5-million-transition target, without the pilot's allocator cap, in seed order 11, 22, 33. The training-only coordinator is `tmp/run_aligned_training.py`; it dispatches the equivalent direct CLI commands below, records source hashes and snapshots under `runs/memory-aligned-launcher/`, and writes live status to `results/memory-aligned-status.json`. It stops on errors, rejects incompatible resume artifacts, and never invokes held-out evaluation or submission export. Per-run logs are in `runs/main_agent_aligned-<seed>/console.log`. Do not start these commands again while the coordinator is running, and do not use `run_matrix.py` for aligned experiments.

```powershell
# Direct command equivalent; already dispatched by the training-only coordinator.
..\.venv\Scripts\python.exe -m exploration train --config configs/memory-aligned.yaml --device cuda

# Example seed override, using a distinct automatic output folder.
..\.venv\Scripts\python.exe -m exploration train --config configs/memory-aligned.yaml --seed 22 --device cuda
..\.venv\Scripts\python.exe -m exploration train --config configs/memory-aligned.yaml --seed 33 --device cuda
```

Original checkpoints, results, submission artifacts, and release builds are preserved. A later approved experiment should compare matched budgets/seeds and record both greedy and sampled behavior. Compact maps, encoder changes, action-selection changes, and reward changes are outside this implementation.
