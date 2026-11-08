# Reproducing this comparison

Use the pinned source-checkout Python environment and the checkpoint paths/hashes in `../protocol.json`. Run from the repository root:

```powershell
..\.venv\Scripts\python.exe results/memory-alignment-comparison/reproduction/restore_run_layout.py
..\.venv\Scripts\python.exe results/memory-alignment-comparison/reproduction/evaluate_aligned_comparison.py
..\.venv\Scripts\python.exe results/memory-alignment-comparison/reproduction/finish_aligned_comparison.py
..\.venv\Scripts\python.exe results/memory-alignment-comparison/reproduction/verify_aligned_curves.py
```

The restore command reconstructs the ignored `runs/` layout from committed selected checkpoints and archived training metadata without overwriting existing files or training anything. It does not restore final/latest optimizer snapshots for continuing training. The evaluator uses frozen models on fresh test seeds 30000-30099; matching complete parts are reused. The finishing pipeline aggregates results, evaluates the aligned policies on training-partition seeds 0-49, profiles CPU latency in isolation, draws a fixed illustrative map, and writes the report. None of these commands trains policies or exports the original submission. Existing results and protected original-artifact hashes must match the frozen protocol. In a newly created local virtual environment, replace `..\.venv\Scripts\python.exe` with `.\.venv\Scripts\python.exe` or `python`.

The observed primary results are deterministic under the saved policies and pinned environment. Latency and wall-clock measurements vary with hardware and load. Map-bootstrap intervals describe map variation conditional on the three trained policies; between-run standard deviations are separate. Further model development should use validation and reserve another fresh confirmation set, since these test results have now been observed.
