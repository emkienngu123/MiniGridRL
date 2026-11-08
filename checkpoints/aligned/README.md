# Aligned-memory checkpoints

These are the validation-selected policies from training seeds 11, 22, and 33. All runs completed 5,001,216 environment transitions. Selected checkpoints occur earlier according to validation coverage AUC, then POI discovery, then survival; the fresh test results were not used for selection.

`index.json` records selected steps, completed budgets, validation scores, and archive SHA-256 checksums. Each archive has a required JSON sidecar, with `variant: main` and `memory_frame: agent_aligned`. These files require the current source implementation; the historical Windows build predates the alignment option.

From the repository root, after installing dependencies:

```powershell
python -m exploration smoke --checkpoint checkpoints/aligned/main_agent_aligned-11.zip
python -m exploration ui --checkpoint checkpoints/aligned/main_agent_aligned-11.zip
```

See [the full comparison](../../results/memory-alignment-comparison/report.md). The original start-aligned and no-map/reward-ablation models remain in the parent checkpoint directory.
