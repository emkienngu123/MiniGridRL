# Hard demonstration selection audit

The saved full-reward aligned PPO checkpoint (training seed 33, selected at 3.75M transitions on Easy/Medium validation) was frozen before this audit. All evaluation here runs on CPU. It does not change checkpoints, hyperparameters, rewards, or the primary study.

`episodes.csv` contains all 50 Hard episodes on seeds 10000-10049, including deaths. Their median revealed coverage is 0.8982869827441782 (89.8%). Map 10039 reaches 0.898936170212766 (89.9%), discovers 5/5 POIs, and survives all 1,536 actions. Maps 10010 and 10039 are equally close to the median. The reviewer proposed 10039; it is used as a near-median illustration that also shows a survived close pass. This is a disclosed qualitative choice, not an unbiased random sample or the best episode.

The monster is orthogonally adjacent after steps 258, 260, 1068, and 1070. These are four adjacent observations, not a verified count of eight separate encounters. The video slows the first event and says the monster passes within one cell and the episode survives. Evaluator-only rings identify the agent and monster; no intentional avoidance is inferred. `selected-trace.json` retains every action and post-step pose. The rendered replay uses one frame per action so a short encounter cannot be lost to the older four-action sampling.

The primary 5,700 episodes and reported aggregate results remain untouched. Hard was not used to select the trained model; this later audit selects a demonstration only. The old map-10000 loop is retained in the failure section with the same checkpoint.

To reproduce all 50 CPU episodes from the repository root, choose a new output directory:

```powershell
python scripts/select_hard_demo.py --workers 4 --output tmp/hard-demo-audit-reproduced
```

Compare the output metrics and selection with this archive. The exporter checks that the new replay's complete action sequence and metrics match this audit.

Near-median final coverage does not imply smooth control: this episode also turns in place during steps 402-909 and 1181-1536. Those intervals are labeled and fast-forwarded in the main video. The unedited one-frame-per-action source replay and full trace retain them.
