# Design rationale and remaining evidence gaps

This audit accompanies the final five-page report and the completed 5700-episode confirmation. The aligned reward ablation is now complete: full reward reaches 90.8% coverage versus 59.6% without seen/visit bonuses. It distinguishes a reason for making a choice from evidence that its exact value is necessary or optimal. It does not retrofit later measurements as original design intent.

## Decisions now explained in the report

| Choice | Reason explained | Evidence / qualification |
|---|---|---|
| MiniGrid extension | Reuse grid, visibility, movement, rendering, and Gymnasium behavior rather than implement another simulator | Engineering rationale; upstream is credited and licensed |
| Connected maze, rooms, extra openings | Dead ends require backtracking; rooms and cycles create alternative routes and loop opportunities; connectivity avoids impossible tasks | Task-based rationale; no generator ablation |
| Six-step spawn separation | Reduce immediate-contact starts | Exact threshold is heuristic; no guaranteed survival interval |
| Non-reversing random patrol | Add stochastic timing decisions without a chasing rule | No experiment isolates the non-reversal rule |
| Increasing sizes, POIs, budgets, patrol pressure | Increase spatial/task demand, allow more travel, and make Easy timing gentler | Exact settings were fixed in the initial plan; no claim they were chosen from measured floor counts |
| Continue after POI discovery | Preserve exploration of undiscovered terrain | Task-based rationale |
| Native 7x7 view, symbolic one-hot input | Preserve partial observability and nearby geometry; avoid artificial ordinal relationships between object IDs | No field-of-view or RGB comparison |
| Wait, heading, movement/contact feedback, remaining time | Permit patrol timing; support odometry; expose observable outcomes and finite-horizon context | Exact feature necessity not ablated |
| Explicit map and relative frame | Give history-dependent exploration an auditable representation; align stored geometry with relative actions | Strongest controlled evidence: aligned versus original-map retraining; probes are supporting diagnostics |
| Seven channels and clearing old threats | Distinguish unknown/observed terrain, task progress, revisits, pose, and current visibility | Reasoned representation; no channel-by-channel or threat-history ablation |
| Fixed canvas and bounded visits | Accommodate supported map sizes and bound numeric inputs | Canvas has a geometric rationale; visit saturation at ten is heuristic |
| Seen/visited/POI bonuses and step/contact costs | Supply feedback before sparse discoveries, reward information and movement, retain task objective, discourage waste and contact | Completed no-bonus comparisons support shaping in both frames; exact coefficients remain heuristic |
| One-time credits and unpaid reset initialization | Prevent repeated sightings, revisits, or initialization from generating repeated bonuses | Implemented accounting/tests; random performance is not treated as an anti-exploit proof |
| Dual CNNs, scalar fusion, shared encoders, separate heads | Preserve local detail, compress wider history, process nonspatial features directly, avoid duplicated perception, separate action/value outputs | Practical architectural rationale; comparative necessity remains unmeasured |
| PPO/SB3 | Discrete actions, dictionary/custom encoders, mature parallel rollout/update/checkpoint pipeline | No algorithm comparison establishes superiority |
| Replay-memory discussion | Explain a later feasibility advantage of short on-policy storage | 85,712 bytes per float32 observation; 1M observation/next-observation pairs about 171 GB before other fields; not the original reason DQN was rejected |
| Easy/Medium training; Hard withheld | Learn across two scales and deliberately test extrapolation | Hard changes several factors jointly |
| Fixed budgets and three training seeds | Match interaction cost and expose run variation | Neither convergence nor a statistical power target was established |
| PPO update controls | Entropy supports diversity; clip/KL/gradient controls limit updates; GAE trades variance and bias | Exact settings come from the initial plan; several differ from SB3 defaults |
| Validation selection by AUC | Reward early information gain and endpoint coverage while keeping selection separate from comparison maps | Tie-breaks: discovery, then survival |
| Random/frontier baselines | Establish a minimal floor and a strong classical coverage reference under observable-information restrictions | Frontier is not an upper bound; safety and coverage remain separate |
| Fixed-budget curves and paired maps | Avoid survivor-only averages and reduce map-difficulty confounding | Intervals condition on saved models; run SD is separate |
| CPU inference measurement | Isolate agent/preprocessing cost from simulation/rendering | Hardware-dependent, descriptive timing |

## What still lacks quantitative justification

These choices have a practical category-level reason, but no evidence for the exact setting. They should remain labeled as heuristics or initial engineering settings, rather than be given invented optimality explanations.

| Remaining choice | What is missing | Smallest useful follow-up |
|---|---|---|
| Exact reward coefficients, especially death versus discovery | The aligned reward ablation is complete, but no weight sweep or causal attribution of the safety tradeoff is available | Separate death/novelty weight comparisons, with the completed no-bonus comparison as context |
| CNN widths, 64/128 feature split, three stride-two map stages, shared encoders, ReLU/Tanh | No capacity search, single-encoder comparison, or shared/separate-feature comparison | One matched-budget encoder-size or sharing ablation at a time |
| Gamma 0.99, lambda 0.95, clip/entropy/KL and update sizes | No parameter search or evidence these values explain stalls; gamma's 100-step discount scale is not a hard horizon | Validation-only discount or entropy ablation with other settings fixed |
| Visit saturation at ten | Bounded inputs are motivated, but ten is not distinguished from alternatives | Fixed-policy input probe for diagnosis, followed by matched retraining if warranted |
| Exact room/opening settings, six-step spawn distance, patrol periods, POI counts, episode budgets | No difficulty calibration, threat fairness bound, or optimized floor-count ratio | Generate descriptive difficulty statistics; change one setting for a controlled robustness study |
| Five-million-transition run budget, three seeds, 250k validation interval, 50 validation maps | No convergence, power, or selection-frequency justification for these exact values | More seeds, learning-curve/convergence analysis, and validation sampling sensitivity |
| 16-state loop window; frontier danger radius/tie-breaking | Reproducible conventions, not optimized navigation/diagnostic choices | Window/radius sensitivity; do not select settings on the current comparison set |

## What is motivated but still unvalidated

- Explicit maps versus recurrent or learned memory; symbolic versus RGB inputs; exact view size; local branch and individual scalar/channel contributions.
- PPO versus compressed/bounded-replay DQN or another algorithm.
- Current-visible-only monsters versus predictive threat memory, and sampled versus greedy aligned action selection.
- Whether the proposed reasons for survival loss, stalls, or Hard degradation are causal. Existing observations motivate hypotheses but do not settle them.

Additional experiments strengthen research claims, but every implementation constant need not be optimized to satisfy a report requiring reasoning. The defensible report explains the practical purpose, identifies arbitrary exact values, and limits conclusions to measured comparisons.

## Seed exposure and evidence provenance

The user disclosed that an earlier review used original-map and mapless results on seeds 30000-30029 to motivate the memory investigation. The archived comparison originally described all seeds 30000-30099 as fresh; that wording is superseded by the explicit disclosure in the revised report. The archived files remain byte-for-byte unchanged to preserve their checksums.

`prepare_assets.py` computes the 30030-30099 exclusion sensitivity from existing episode rows only. `generated/seed-overlap-sensitivity.json` records the disclosure source, included/excluded seeds, macro means, and 10,000-resample paired-map intervals (NumPy RNG seed 20261107). It neither runs models nor selects checkpoints. The exclusion is post-hoc, not a new preregistered confirmation set. Further model development needs independent maps.
