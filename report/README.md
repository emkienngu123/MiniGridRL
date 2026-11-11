# Editable technical report

`technical-report.tex` is the report source; `references.bib` contains the MiniGrid, PPO, and Stable-Baselines3 references. The final PDF is [`../output/pdf/technical-report.pdf`](../output/pdf/technical-report.pdf), also delivered under `submission/final/technical-report.pdf`. The historical `submission/technical-report.pdf` is preserved.

## Write and preview in VS Code

1. Restart VS Code after installing TinyTeX so its new PATH is available.
2. Open `report.code-workspace` using **File > Open Workspace from File**.
3. Open `technical-report.tex`. **Ctrl+Alt+B** compiles; **Ctrl+Alt+V** opens the PDF preview. Saving rebuilds the PDF. Review compiler diagnostics in **Output > LaTeX Workshop**.

The workspace settings route LaTeX output and SyncTeX files to `output/pdf/`. They do not change global VS Code settings. Tables and figures are already generated and checked in, so normal text edits require only LaTeX, not Python or training.

## Rebuild from recorded evidence

From the repository root, using a Python environment with NumPy and Matplotlib:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File report/build.ps1 -Python ..\.cpu-venv\Scripts\python.exe
```

If the environment lives elsewhere, replace the `-Python` path. To compile only the text, from `report/`:

```powershell
latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -outdir=../output/pdf technical-report.tex
```

`prepare_assets.py` verifies the confirmation archive, checks all 5,700 episode means, and creates current tables, numeric macros, and Hard curves under `generated/`. It does not run agents or train. `generated/provenance.json` records source hashes. The archive's independent `verification.json` additionally verifies all 188,100 curve rows, fixed-budget endpoints, paired bootstrap intervals, and 15 selected models. Older generated validation and 70-map sensitivity assets remain historical context and are not the primary results.

## Specification coverage

| Required topic | Report location |
|---|---|
| Problem formulation; MiniGrid environment | Section 1 |
| Observation/action spaces | Section 2 |
| Reward design and scales | Section 3 |
| RL algorithm; policy architecture and motivation | Section 4 |
| Training setup; baselines and ablations | Section 5 |
| Evaluation methodology and seed disclosure | Section 6 |
| Results, uncertainty, sensitivity, generalization | Section 7 |
| Failure cases | Section 8 |
| Compute, reproducibility, delivery status | Section 9 |
| Limitations; future improvements | Section 10 |

The primary report uses reserved confirmation seeds 40000-40099 and includes the completed aligned reward ablation. Earlier seeds 20000-20099 and old-model results on 30000-30029 informed development; the old comparison and its exclusion sensitivity remain disclosed as historical evidence. Selected models were frozen before confirmation testing. The report separates run SD from episode variation and paired-map intervals, and reports the frontier AUC and safety tradeoffs honestly.

[The design rationale audit](design-rationale-audit.md) distinguishes motivated decisions from exact constants still lacking optimization or comparative evidence.
