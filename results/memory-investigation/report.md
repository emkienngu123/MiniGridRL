# Memory investigation

This is a post-benchmark investigation using frozen checkpoints and validation seeds 10000-10009. It does not change the submitted benchmark, models, or report.

## Confirmed findings

- Mapping audit: 5,218 states and 1,526,629 known-cell checks across all four headings. Errors: {'pose': 0, 'floor': 0, 'wall': 0, 'poi': 0, 'agent_marker': 0, 'current_monster': 0}.
- A supplementary frontier-trajectory audit checks 1,185 states and 34,914 currently visible cells. There are no missing visible cells/POIs or forgotten sampled observations. Full results: `mapping-audit.json`.
- The no_memory variant still maintains observation history for reward calculation. The original ablation removes the map from policy inputs, not history from the entire system.
- The trained memory policies depend on the map: zeroing it changes greedy actions in 22.9% of 864 sampled states. Mean coverage falls from 68.2% to 56.6%. Keeping only current-view information gives 61.2%.
- Sampling actions improves the main policies' mean coverage from 68.2% to 80.7%, with loop rate falling from 47.3% to 24.1%. The memory-disabled policies also improve under sampling, so sampling does not explain the entire gap.
- Removing the agent-position marker changes 0.0% of greedy actions; average probability total variation is 0.0047. Pose can be redundant with other channels, so this is weak marker reliance, not proof of no localization.
- A non-clipping four-cell translation of every map channel changes 13.5% of main-policy decisions. Joint map/heading rotation changes 20.5%, versus 4.7% for memory-disabled policies on the same local observations.
- The map encoder has 459,008 parameters out of 649,925 total (70.6%). Every one of those parameters remains exactly at initialization in all three no_memory checkpoints. Their memory-to-actor and memory-to-critic columns also remain unchanged. Thus the nominally identical architecture has substantially less active learning capacity in the no_memory condition.
- Average known canvas fraction is 9.7%. Actor tanh saturation above |preactivation|=3 affects only 0.7% of units, so these probes do not support a simple activation-explosion explanation.

## Frozen-checkpoint rollout results

Means pool Easy/Medium/Hard equally. Learned-policy rows each use three training seeds; these are validation diagnostics, not final test scores.

| Family | Policy input | Action choice | Episodes | Coverage | Loop rate | Survival |
|---|---|---|---:|---:|---:|---:|
| main | full | greedy | 90 | 68.2% | 47.3% | 73.3% |
| main | full | stochastic | 90 | 80.7% | 24.1% | 78.9% |
| main | zero_memory | greedy | 90 | 56.6% | 73.2% | 72.2% |
| main | no_history | greedy | 90 | 61.2% | 66.0% | 74.4% |
| main | no_visits | greedy | 90 | 65.8% | 45.0% | 68.9% |
| no_memory | full | greedy | 90 | 78.3% | 35.0% | 87.8% |
| no_memory | full | stochastic | 90 | 85.0% | 19.0% | 80.0% |
| frontier | full | greedy | 30 | 94.6% | 18.3% | 40.0% |
| frontier | no_history | greedy | 30 | 24.5% | 98.7% | 26.7% |

## Interpretation and limits

The evidence supports a representation/optimization problem plus greedy-policy looping, rather than corrupted or wholly ignored memory. The local view is agent-relative, while the large map is anchored at the episode start and aligned to world directions. A three-stage strided CNN followed by flattening must learn localization, frame alignment, history interpretation, and navigation from PPO rewards. No explicit agent-centered crop or frontier-goal representation is supplied.

This is a supported mechanism hypothesis, not an isolated causal proof. Masked and translated maps are out-of-distribution interventions. Agent position is also partially recoverable from visit patterns. The stochastic comparison has one action RNG stream per model/map; more sampling seeds would strengthen it. Frontier without history is a positive control for its map-dependent algorithm, not a proposed competitive stateless baseline.

The main model's weaker results are not explained solely by the resumed seed-33 run: the original held-out comparison still favors no_memory when comparing only seeds 11 and 22. Main validation AUC also regresses substantially late in all three runs, whereas no_memory is more stable; simply extending this training setup is not established as a solution.

## Next controlled experiment

1. Encode a smaller agent-centered, heading-aligned memory patch, including visit counts and known/unknown cells. This removes the need to infer the agent's location from a single pixel on a world-aligned 55x55 map.
2. Use a smaller map encoder and distinguish input-information effects from active-parameter-count effects with a capacity-matched local-only control.
3. Evaluate both greedy and sampled actions under a declared validation protocol. Select the mode on validation and reserve a fresh held-out set for the revised system.
4. Consider warm-starting from the successful local policy, initially zeroing only the map-to-head connection weights so adding memory preserves initial behavior while allowing new map connections to learn.

No revised policy has been trained by this investigation. These changes require matched-budget, multi-seed retraining before claiming an improvement.

## Reproduction and sources

```powershell
..\.venv\Scripts\python.exe scripts/investigate_memory.py --count 10
..\.venv\Scripts\python.exe scripts/investigate_memory.py --count 10 --mapping-only
```

- Raw evidence: `results/memory-investigation/diagnostics.json` and `episodes.csv`; aggregates: `summary.json`.
- Original benchmark summary SHA-256: `6de3e48a4ef54fa95595dc2ad649c8bdce53cdcb50622563b56a5276b2a67b70`; unchanged: `True`.
- Source: `exploration/memory.py`, `policy.py`, `training.py`; diagnostic transformations have independent visibility/geometry tests.
- SB3 documents shared PPO feature extractors: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html . Shared extraction is confirmed in the loaded models; it is not proven to be the cause here.
- Architectural context: Active Neural SLAM uses modular mapping, global/local policies, and analytic planning: https://arxiv.org/abs/2004.05155 . This motivates a structured map interface, not a claim that it will automatically improve this task.
