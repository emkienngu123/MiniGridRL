import csv
from pathlib import Path

project = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
root = project / "results/memory-alignment-comparison"


def key(row):
    return tuple(row[k] for k in ("agent", "preset", "seed", "training_seed", "policy_seed"))


with (root / "episodes.csv").open(encoding="utf-8", newline="") as stream:
    expected = {key(row): row for row in csv.DictReader(stream)}
last = {}
count = 0
with (root / "curves.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream):
        k = key(row)
        coverage, discovery = float(row["coverage"]), float(row["poi_fraction"])
        if k in last:
            assert coverage + 1e-12 >= last[k][0]
            assert discovery + 1e-12 >= last[k][1]
        last[k] = (coverage, discovery, float(row["budget_fraction"]))
        count += 1
assert count == 158400 and len(last) == 4800
for k, values in last.items():
    assert abs(values[0] - float(expected[k]["revealed_coverage"])) < 1e-12
    assert abs(values[1] - float(expected[k]["poi_fraction"])) < 1e-12
    assert values[2] == 1
print("Verified 158400 curve rows: monotonic discoveries; fixed-budget endpoints match all 4800 episodes")
