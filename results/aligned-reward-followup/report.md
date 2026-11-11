# Aligned memory: reward ablation and reserved-map comparison

Confirmation maps 40000-40099, frozen validation-selected policies; no confirmation tuning.

| Agent | Coverage % | Visited % | AUC | POIs % | Survival % | Loops % |
|---|---:|---:|---:|---:|---:|---:|
| random | 43.9 | 11.7 | 0.386 | 45.4 | 29.6 | 68.4 |
| frontier | 91.5 | 58.5 | 0.869 | 92.1 | 33.7 | 20.5 |
| main | 68.9 | 37.4 | 0.617 | 68.0 | 74.7 | 46.4 |
| no_memory | 78.6 | 48.2 | 0.701 | 79.4 | 75.0 | 31.7 |
| no_novelty | 42.0 | 9.7 | 0.377 | 41.8 | 60.1 | 84.4 |
| main_agent_aligned | 90.8 | 65.6 | 0.825 | 91.3 | 72.1 | 41.0 |
| no_novelty_agent_aligned | 59.6 | 19.1 | 0.547 | 59.2 | 56.8 | 75.4 |

## Predeclared aligned reward contrast

Full reward minus no seen/visit bonuses; coverage differences in percentage points. 95% paired-map intervals average training runs within each map.
- easy: +16.2 pp [+13.5, +19.1].
- medium: +35.8 pp [+32.2, +39.6].
- hard: +41.6 pp [+37.8, +45.3].

A positive or negative result must be reported as measured; the protocol makes no assumed superiority claim.
The old 30000 comparison and seed-overlap disclosure remain archived. These new maps are now exposed; further selection needs separate maps.
Raw episode/run variation and all predeclared contrasts are in summary.json; fixed-budget curves retain information after death.
