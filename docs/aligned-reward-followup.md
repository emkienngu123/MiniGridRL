# Aligned reward ablation and reserved-map confirmation

**Completed:** all three runs finished 5,001,216 transitions; all 5,700 reserved-map episodes and independent checks passed. Full-reward aligned coverage is 90.8% versus 59.6% without bonuses. [Final handoff](../submission/final/START-HERE.md), [analysis](../results/aligned-reward-followup/report.md), and [CPU reproduction without training](../scripts/reproduce_confirmation.py).

Authorized on 11 November 2026 after stopping the idle `datamate-vllm-vllm-1` container. The source README, frozen comparison, historical artifacts, and current aligned demo/build are not rewritten by this pipeline.

## Scientific change

`configs/aligned-no-bonuses.yaml` matches `configs/memory-aligned.yaml` except for the variant and output directory. `variant: no_novelty` disables only seen-floor and first-visit rewards in the wrapper. It retains the same local/map/scalar observations, observation-derived history, aligned readout, network, POI reward, step/contact penalties, environment mixture, and PPO settings. A preflight checked identical observations/transitions and the exact reward-term difference.

Three independent runs use seeds 11, 22, and 33, each targeting 5,000,000 transitions with the existing rollout rounding. They run sequentially on CUDA under:

```text
runs/no_novelty_agent_aligned-11/
runs/no_novelty_agent_aligned-22/
runs/no_novelty_agent_aligned-33/
```

Each seed selects its checkpoint solely on Easy/Medium validation AUC, then POI fraction, then survival. This fills the missing reward ablation; it does not assume the full-reward policy will win.

## Reserved confirmation protocol

The frozen protocol is `results/aligned-reward-followup/protocol.json`. It reserves map seeds **40000-40099**, all three presets, before new training or evaluation. An audit of 21,165 recorded episode rows found no use of those seeds; it cannot audit unrecorded external/manual evaluations.

After all three runs complete, the runner freezes selected checkpoint hashes for the five PPO conditions (original map, mapless, original no bonuses, aligned full reward, aligned no bonuses), each with training seeds 11/22/33. Random uses policy seeds 11/22/33; frontier is deterministic. The comparison contains **5,700 episodes** on CPU, with four concurrent processes and one PyTorch thread per process. Hard remains excluded from training/selection. No confirmation-based tuning or checkpoint replacement is permitted.

The primary reward contrast is aligned full reward minus aligned no bonuses. Other predeclared contrasts compare aligned full reward with original map, mapless, frontier, and random. Coverage AUC, discovery, and survival are retained; paired-map intervals condition on trained runs, with episode and run variation separate. Ten thousand bootstrap draws use RNG seed 20261111. Results of either sign must be reported honestly.

## Status and recovery

Read progress without touching the training process:

```powershell
Get-Content results/aligned-reward-followup/status.json
Get-Content runs/no_novelty_agent_aligned-11/console.log -Tail 35
```

The pipeline's durable logs are in `../.runtime/aligned-followup.log` and `../.runtime/aligned-followup-error.log`. Its process and child PIDs are recorded in `status.json`. Training CSVs, TensorBoard logs, validation records, and checkpoints remain in each run directory.

If a run fails, inspect its log before restarting. Running the same command again resumes only a matching saved run, with identical scientific configuration. A process lock prevents duplicate pipeline runs. Interrupted sampling is not bitwise continuation; resume provenance must be disclosed. No automatic parameter changes are made.

```powershell
..\.venv\Scripts\python.exe scripts/run_aligned_reward_followup.py
```

The pipeline automatically proceeds from training to checkpoint freezing, confirmation evaluation, and aggregation. On successful completion it writes `episodes.csv`, `curves.csv`, `summary.json`, and `report.md` in `results/aligned-reward-followup/`. Until status is `complete`, these are pending experiments, not established results. The final LaTeX report and demo now incorporate the reviewed confirmation measurements; package selection remains the validation-selected full-reward seed-33 model. Historical submissions remain unchanged.
