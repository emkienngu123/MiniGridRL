"""Build current-report assets from verified confirmation records; no rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
RESULTS = ROOT / "results/aligned-reward-followup"
OUTPUT = REPORT / "generated"
AGENTS = (
    ("random", "Random", "#83909e"),
    ("frontier", "Frontier", "#7e4c9f"),
    ("no_novelty", "PPO, no bonuses (original)", "#ac7b1a"),
    ("no_memory", "PPO, no map", "#b95825"),
    ("main", "PPO, original map", "#427ca6"),
    ("no_novelty_agent_aligned", "PPO, no bonuses (aligned)", "#bc5b8c"),
    ("main_agent_aligned", "PPO, aligned map", "#008478"),
)


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    hashes = {}
    for name in (
        "summary.json",
        "episodes.csv",
        "curves.csv",
        "protocol.json",
        "frozen-checkpoints.json",
    ):
        digest = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        assert digest == manifest["files"][name], name
        hashes[name] = digest
    s = json.loads((RESULTS / "summary.json").read_text())
    assert s["episodes"] == 5700
    with (RESULTS / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5700 and {int(r["seed"]) for r in rows} == set(
        range(40000, 40100)
    )
    for agent, _, _ in AGENTS:
        for preset in ("easy", "medium", "hard"):
            records = [r for r in rows if r["agent"] == agent and r["preset"] == preset]
            assert len(records) == (100 if agent == "frontier" else 300)
            for metric in (
                "revealed_coverage",
                "visited_coverage",
                "coverage_auc",
                "poi_fraction",
                "survived",
                "loop_rate",
            ):
                assert np.isclose(
                    np.mean([float(r[metric]) for r in records]),
                    s["summary"][f"{agent}/{preset}"]["metrics"][metric]["mean"],
                    atol=1e-12,
                    rtol=0,
                )
    numbers = ["% Generated from verified confirmation seeds 40000-40099."]
    for macro, agent in (
        ("AlignedCoverage", "main_agent_aligned"),
        ("OriginalCoverage", "main"),
        ("MaplessCoverage", "no_memory"),
        ("FrontierCoverage", "frontier"),
        ("NoBonusCoverage", "no_novelty_agent_aligned"),
    ):
        numbers.append(
            f"\\newcommand{{\\{macro}}}{{{100 * s['overall'][agent]['revealed_coverage']:.1f}}}"
        )
    numbers += [
        r"\newcommand{\PrimaryEpisodes}{5,700}",
        r"\newcommand{\TotalParameters}{649,925}",
        r"\newcommand{\MapParameters}{459,008}",
    ]
    probes = json.loads(
        (ROOT / "results/memory-investigation/summary.json").read_text()
    )["probes"]["main"]
    for macro, key in (
        ("TranslationFlips", "translate"),
        ("RotationFlips", "rotate90"),
    ):
        value = 100 * probes["interventions"][key]["greedy_action_flip_fraction"]
        numbers.append(f"\\newcommand{{\\{macro}}}{{{value:.1f}}}")
    contrasts = []
    for preset in ("easy", "medium", "hard"):
        entry = next(
            r
            for r in s["paired_comparisons"]
            if r["preset"] == preset and r["right"] == "no_novelty_agent_aligned"
        )
        value = entry["metrics"]["revealed_coverage"]
        lo, hi = value["map_bootstrap_95ci"]
        contrasts.append(
            f"{preset.title()} +{100 * value['mean_difference']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        )
    numbers.append("\\newcommand{\\RewardGains}{" + "; ".join(contrasts) + "}")
    (OUTPUT / "numbers.tex").write_text("\n".join(numbers) + "\n", encoding="utf-8")

    coverage = [
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Agent & Easy & Medium & Hard\\\midrule",
    ]
    overall = [
        r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        r"Agent & Coverage & Visited & AUC & POIs & Survival & Loops\\\midrule",
    ]
    for agent, label, _ in AGENTS:
        cells = []
        for preset in ("easy", "medium", "hard"):
            m = s["summary"][f"{agent}/{preset}"]["metrics"]["revealed_coverage"]
            value = f"{100 * m['mean']:.1f}"
            if m["run_mean_std"] is not None:
                value += f" $\\pm$ {100 * m['run_mean_std']:.1f}"
            cells.append(value)
        prefix = r"\rowcolor{accent!8}" if agent == "main_agent_aligned" else ""
        coverage.append(prefix + label + " & " + " & ".join(cells) + r"\\")
        m = s["overall"][agent]
        values = [
            f"{m[key] * (1 if key == 'coverage_auc' else 100):.{3 if key == 'coverage_auc' else 1}f}"
            for key in (
                "revealed_coverage",
                "visited_coverage",
                "coverage_auc",
                "poi_fraction",
                "survived",
                "loop_rate",
            )
        ]
        overall.append(prefix + label + " & " + " & ".join(values) + r"\\")
    for name, lines in (
        ("coverage-table.tex", coverage),
        ("overall-table.tex", overall),
    ):
        (OUTPUT / name).write_text(
            "\n".join(lines + [r"\bottomrule\end{tabular}"]) + "\n", encoding="utf-8"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#17324d",
            "text.color": "#17324d",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "savefig.facecolor": "white",
        }
    )
    curves = defaultdict(list)
    with (RESULTS / "curves.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["preset"] == "hard":
                for metric in ("coverage", "poi_fraction"):
                    curves[row["agent"], metric, float(row["budget_fraction"])].append(
                        float(row[metric])
                    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.95))
    for ax, metric, title in zip(
        axes,
        ("coverage", "poi_fraction"),
        ("Revealed traversable cells", "POIs discovered"),
    ):
        for agent, label, color in AGENTS:
            xs = sorted(x for a, m, x in curves if a == agent and m == metric)
            ys = [100 * np.mean(curves[agent, metric, x]) for x in xs]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.7 if agent == "main_agent_aligned" else 1.15,
                label=label.replace("PPO, ", ""),
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 103),
            xlabel="Fraction of action budget",
            ylabel="Percent",
            title=title,
        )
        ax.set_yticks([0, 25, 50, 75, 100])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False, fontsize=7.2)
    fig.tight_layout(rect=(0, 0.25, 1, 1), pad=0.2, w_pad=1)
    fig.savefig(OUTPUT / "hard-curves.png", dpi=240)
    plt.close(fig)
    provenance = {
        "study": "aligned-reward-followup",
        "maps": [40000, 40099],
        "episodes": 5700,
        "source_sha256": hashes,
        "verification": "Independent full archive verification in results/aligned-reward-followup/verification.json",
        "historical_note": "Previous generated seed-overlap sensitivity remains historical; it is not the primary confirmation.",
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        "Verified 5,700 confirmation episodes; generated current tables, macros, and Hard curves."
    )


if __name__ == "__main__":
    main()
