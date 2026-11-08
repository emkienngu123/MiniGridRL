"""Aggregate the predeclared fresh test; no fitting, selection, or tuning."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
sys.path.insert(0, str(ROOT))
from exploration.evaluation import summarize, write_csv  # noqa: E402

OUT = ROOT / "results/memory-alignment-comparison"
ORDER = ("random", "frontier", "no_novelty", "no_memory", "main", "main_agent_aligned")
NAMES = {"random": "Random", "frontier": "Frontier", "no_novelty": "No shaping (old frame)",
         "no_memory": "PPO without map", "main": "PPO original map", "main_agent_aligned": "PPO aligned map"}
COLORS = {"random": "#8a97a8", "frontier": "#279c87", "no_novelty": "#cf9840",
          "no_memory": "#a875b3", "main": "#4777c2", "main_agent_aligned": "#da6046"}
PRESETS = ("easy", "medium", "hard")
METRICS = ("revealed_coverage", "visited_coverage", "coverage_auc", "poi_discovered", "poi_fraction",
           "survived", "monster_collisions", "wall_collisions", "wall_collision_rate",
           "movement_efficiency", "repeated_state_rate", "loop_rate", "reward")


def read_rows(path):
    text = {"agent", "preset", "termination", "memory_frame"}
    integers = {"seed", "training_seed", "policy_seed", "step", "steps", "episode_limit"}
    with path.open(encoding="utf-8", newline="") as stream:
        return [{k: v if k in text else (None if v == "" else int(v) if k in integers else float(v))
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["agent"], row["preset"]].append(row)
    return result


def stats(records, metric):
    values = np.array([r[metric] for r in records])
    runs = defaultdict(list)
    for row in records:
        runs[row["training_seed"] if row["training_seed"] is not None else row["policy_seed"]].append(row[metric])
    means = {str(k): float(np.mean(v)) for k, v in runs.items()}
    return {"mean": float(values.mean()), "episode_std": float(values.std(ddof=1)),
            "run_means": means, "run_mean_std": float(np.std(list(means.values()), ddof=1)) if len(means) > 1 else None,
            "p10": float(np.percentile(values, 10)), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90))}


def paired(rows, selected_seeds=None):
    metrics = ("revealed_coverage", "coverage_auc", "poi_fraction", "survived", "visited_coverage",
               "loop_rate", "wall_collision_rate", "movement_efficiency")
    records = [r for r in rows if selected_seeds is None or r["training_seed"] in selected_seeds or r["training_seed"] is None]
    groups = grouped(records)
    result = []
    rng = np.random.default_rng(20261107)
    for preset in PRESETS:
        indices = rng.integers(0, 100, size=(10000, 100))
        for baseline in ORDER[:-1]:
            arrays = []
            for family in ("main_agent_aligned", baseline):
                by_map = defaultdict(list)
                for row in groups[family, preset]:
                    by_map[row["seed"]].append([row[m] for m in metrics])
                assert sorted(by_map) == list(range(30000, 30100))
                arrays.append(np.array([np.mean(by_map[s], axis=0) for s in sorted(by_map)]))
            differences = arrays[0] - arrays[1]
            entry = {"preset": preset, "baseline": baseline, "paired_maps": 100,
                     "training_seeds": selected_seeds or [11, 22, 33], "metrics": {}}
            for i, metric in enumerate(metrics):
                samples = differences[:, i][indices].mean(axis=1)
                entry["metrics"][metric] = {"mean_difference": float(differences[:, i].mean()),
                                            "map_bootstrap_95ci": list(map(float, np.percentile(samples, [2.5, 97.5])))}
            result.append(entry)
    return result


def history():
    result = {}
    for family in ORDER[2:]:
        per_run = []
        for seed in (11, 22, 33):
            with (ROOT / f"runs/{family}-{seed}/validation.csv").open(encoding="utf-8", newline="") as stream:
                values = {int(r["step"]): float(r["coverage_auc"]) for r in csv.DictReader(stream)}
            best = json.loads((ROOT / f"runs/{family}-{seed}/best.json").read_text(encoding="utf-8"))
            runtime = json.loads((ROOT / f"runs/{family}-{seed}/runtime.json").read_text(encoding="utf-8"))
            per_run.append({"seed": seed, "values": values, "selected_step": best["training_steps"],
                            "selected_auc": best["validation_score"][0], "final_auc": values[max(values)], "runtime": runtime})
        steps = sorted(set.intersection(*[set(r["values"]) for r in per_run]))
        steps = [s for s in steps if s <= 5000000 and s % 250000 == 0]
        result[family] = {"runs": per_run, "steps": steps,
                          "mean": [float(np.mean([r["values"][s] for r in per_run])) for s in steps],
                          "std": [float(np.std([r["values"][s] for r in per_run], ddof=1)) for s in steps]}
    return result


def plots(groups, curves, histories, comparisons):
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    short = ["Random", "Frontier", "No shaping", "No map", "Old map", "Aligned"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey="row")
    for col, preset in enumerate(PRESETS):
        for row, (metric, label) in enumerate((("revealed_coverage", "Revealed coverage"), ("coverage_auc", "Coverage AUC"), ("survived", "Survival"))):
            means = [stats(groups[a, preset], metric)["mean"] for a in ORDER]
            sd = [stats(groups[a, preset], metric)["run_mean_std"] or 0 for a in ORDER]
            ax = axes[row, col]
            ax.bar(range(6), means, color=[COLORS[a] for a in ORDER], yerr=sd, capsize=3)
            ax.set(ylim=(0, 1.06), xticks=range(6), xticklabels=short)
            ax.tick_params(axis="x", labelrotation=35, labelsize=9)
            if row == 0:
                ax.set_title(preset.title())
            if col == 0:
                ax.set_ylabel(label)
    fig.suptitle("Fresh held-out seeds 30000–30099; bars = mean, error bars = run-mean SD", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(directory / "outcomes.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for family in ORDER[2:]:
        h = histories[family]
        x, y, sd = np.array(h["steps"]) / 1e6, np.array(h["mean"]), np.array(h["std"])
        axes[0].plot(x, y, label=NAMES[family], color=COLORS[family])
        axes[0].fill_between(x, y - sd, y + sd, color=COLORS[family], alpha=0.13)
    axes[0].set(xlabel="Training transitions (millions)", ylabel="Validation coverage AUC", ylim=(0, 1), title="Mean ± run SD; Easy and Medium")
    axes[0].legend(fontsize=8, frameon=False)
    for i, family in enumerate(ORDER[2:]):
        h = histories[family]
        selected = [r["selected_auc"] for r in h["runs"]]
        final = [r["final_auc"] for r in h["runs"]]
        axes[1].bar(i - .18, np.mean(selected), width=.35, color=COLORS[family], label="Selected" if i == 0 else None)
        axes[1].bar(i + .18, np.mean(final), width=.35, color=COLORS[family], alpha=.4, label="Final" if i == 0 else None)
    axes[1].set(xticks=range(4), xticklabels=["No shaping", "No map", "Old map", "Aligned"], ylim=(0, 1), ylabel="Validation coverage AUC", title="Selected versus final checkpoint")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "learning.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row", sharex=True)
    grouped_curves = defaultdict(lambda: defaultdict(list))
    for r in curves:
        for metric in ("coverage", "poi_fraction"):
            grouped_curves[r["agent"], r["preset"], metric][r["budget_fraction"]].append(r[metric])
    for col, preset in enumerate(PRESETS):
        for row, metric in enumerate(("coverage", "poi_fraction")):
            for family in ORDER:
                values = grouped_curves[family, preset, metric]
                xs = sorted(values)
                axes[row, col].plot(xs, [np.mean(values[x]) for x in xs], color=COLORS[family], label=NAMES[family])
            axes[row, col].set(ylim=(0, 1.02), xlim=(0, 1))
            if row == 0:
                axes[row, col].set_title(preset.title())
            else:
                axes[row, col].set_xlabel("Fraction of full action budget")
            if col == 0:
                axes[row, col].set_ylabel("Revealed coverage" if row == 0 else "POI discovery fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Fixed-budget curves retain final coverage/discovery after death")
    fig.tight_layout(rect=(0, .1, 1, .96))
    fig.savefig(directory / "coverage-discovery.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, metric, title in zip(axes, ("revealed_coverage", "coverage_auc", "survived"), ("Coverage gain (pp)", "AUC gain × 100", "Survival gain (pp)")):
        for i, preset in enumerate(PRESETS):
            records = [next(r for r in comparisons if r["preset"] == preset and r["baseline"] == base) for base in ORDER[:-1]]
            means = np.array([r["metrics"][metric]["mean_difference"] for r in records]) * 100
            ci = np.array([r["metrics"][metric]["map_bootstrap_95ci"] for r in records]) * 100
            ax.errorbar(means, np.arange(5) + (i - 1) * .2, xerr=np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means])),
                        fmt="o", capsize=3, label=preset.title())
        ax.axvline(0, color="#666", lw=.8)
        ax.set(title=title, yticks=range(5), yticklabels=[NAMES[a] for a in ORDER[:-1]])
        ax.grid(axis="x", alpha=.15)
    axes[-1].legend(frameon=False)
    fig.suptitle("Aligned minus comparator: paired-map mean and 95% bootstrap interval")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / "paired-differences.png", dpi=160)
    plt.close(fig)


def main():
    status = json.loads((OUT / "status.json").read_text())
    if status["phase"] != "complete":
        raise ValueError("Evaluation has not completed")
    protocol = json.loads((OUT / "protocol.json").read_text())
    rows, curves = [], []
    for part in sorted((OUT / "parts").iterdir()):
        rows.extend(read_rows(part / "episodes.csv"))
        curves.extend(read_rows(part / "curves.csv"))
    assert len(rows) == 4800
    groups = grouped(rows)
    assert len(groups) == 18
    for (family, preset), records in groups.items():
        assert len(records) == (100 if family == "frontier" else 300)
        assert {r["seed"] for r in records} == set(range(30000, 30100))
        keys = [(r["seed"], r["training_seed"], r["policy_seed"]) for r in records]
        assert len(set(keys)) == len(keys)
    for path, expected in protocol["original_artifact_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    write_csv(OUT / "episodes.csv", rows)
    write_csv(OUT / "curves.csv", curves)
    detail = {f"{a}/{p}": {m: stats(groups[a, p], m) for m in METRICS} for a in ORDER for p in PRESETS}
    aggregate = []
    for a in ORDER:
        for p in PRESETS:
            for m in METRICS:
                s = detail[f"{a}/{p}"][m]
                aggregate.append({"agent": a, "preset": p, "metric": m, "episodes": len(groups[a, p]),
                                  "mean": s["mean"], "episode_std": s["episode_std"], "run_mean_std": s["run_mean_std"],
                                  "p10": s["p10"], "p50": s["p50"], "p90": s["p90"]})
    write_csv(OUT / "aggregate-metrics.csv", aggregate)
    comparisons = paired(rows)
    histories = history()
    overall = {a: {m: float(np.mean([detail[f"{a}/{p}"][m]["mean"] for p in PRESETS])) for m in METRICS} for a in ORDER}
    gap = {a: {m: detail[f"{a}/hard"][m]["mean"] - detail[f"{a}/medium"][m]["mean"] for m in METRICS} for a in ORDER}
    report = {"scope": "fresh held-out evaluation; frozen validation-selected models", "episodes": len(rows),
              "protocol": protocol, "summary": summarize(rows), "metrics": detail, "overall_equal_preset_mean": overall,
              "paired_comparisons": comparisons, "sensitivity_training_seeds_11_22": paired(rows, [11, 22]),
              "validation_history": histories, "hard_minus_medium": gap, "original_artifacts_unchanged": True,
              "uncertainty_note": "Map-bootstrap intervals condition on the three trained policies; run SD is reported separately. No multiple-comparison correction.",
              "latency_note": "Raw episode timings were collected under four-way evaluation concurrency. Use isolated-latency.json for reported latency."}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plots(groups, curves, histories, comparisons)
    print(json.dumps({"overall": overall, "fresh_coverage": {a: {p: detail[f'{a}/{p}']['revealed_coverage']['mean'] for p in PRESETS} for a in ORDER}}, indent=2))


if __name__ == "__main__":
    main()
