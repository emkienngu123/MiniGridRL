"""Finish diagnostics and reporting after the primary frozen evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
OUT = ROOT / "results/memory-alignment-comparison"


def main():
    while True:
        try:
            state = json.loads((OUT / "status.json").read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if state["phase"] == "failed":
            raise RuntimeError(state)
        if state["phase"] == "complete":
            break
        time.sleep(3)
    for step in ("analyze_aligned_comparison", "evaluate_aligned_train_partition", "profile_aligned_comparison",
                 "qualitative_aligned_comparison", "write_aligned_comparison_report", "verify_aligned_curves"):
        (OUT / "analysis-status.json").write_text(json.dumps({"phase": step}), encoding="utf-8")
        with (OUT / f"{step}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parent / f"{step}.py")], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=0x08000000)
    (OUT / "analysis-status.json").write_text(json.dumps({"phase": "complete"}), encoding="utf-8")


if __name__ == "__main__":
    main()
