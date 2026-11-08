"""Build readable tables and an evidence index from completed measurements."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "exploration/policy.py").is_file())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_aligned_comparison import METRICS, NAMES, ORDER, OUT, PRESETS, grouped, read_rows, stats  # noqa: E402


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    report = json.loads((OUT / "summary.json").read_text())
    detail, overall = report["metrics"], report["overall_equal_preset_mean"]
    latency = json.loads((OUT / "isolated-latency.json").read_text())
    fresh = read_rows(OUT / "episodes.csv")
    training = [r for r in read_rows(ROOT / "results/full/train-diagnostic/episodes.csv") if r["seed"] < 50]
    for seed in (11, 22, 33):
        training.extend(read_rows(OUT / f"training-partition/main_agent_aligned-{seed}/episodes.csv"))
    assert len(training) == 2400
    fields = list(dict.fromkeys(key for row in training for key in row))
    normalized_training = [{key: row.get(key, "start_aligned" if key == "memory_frame" else None) for key in fields} for row in training]
    from exploration.evaluation import write_csv
    write_csv(OUT / "training-partition/episodes.csv", normalized_training)
    reference_manifest = json.loads((ROOT / "results/full/train-diagnostic/summary.json").read_text())
    expected_hashes = {(r["family"], r["seed"]): r["sha256"] for r in report["protocol"]["checkpoint_manifest"]}
    for agent in reference_manifest["agents"]:
        if agent.get("training_seed") is not None:
            assert agent["sha256"] == expected_hashes[agent["agent"], agent["training_seed"]]
    train_groups = grouped(training)
    generalization = {}
    for family in ORDER:
        for preset in PRESETS:
            assert len(train_groups[family, preset]) == (50 if family == "frontier" else 150)
            key = f"{family}/{preset}"
            generalization[key] = {m: {"train_partition_mean": stats(train_groups[family, preset], m)["mean"],
                                       "fresh_test_mean": detail[key][m]["mean"],
                                       "test_minus_train": detail[key][m]["mean"] - stats(train_groups[family, preset], m)["mean"]}
                                   for m in ("revealed_coverage", "coverage_auc", "poi_fraction", "survived")}
    (OUT / "generalization.json").write_text(json.dumps({"training_map_seeds": [0, 49], "fresh_test_seeds": [30000, 30099],
        "note": "Descriptive difference of independent map samples; training partition does not guarantee each map was encountered. Hard maps were never trained on; for Hard, training partition refers only to seed numbers.",
        "metrics": generalization, "hard_minus_medium": report["hard_minus_medium"]}, indent=2), encoding="utf-8")
    aligned = [r for r in fresh if r["agent"] == "main_agent_aligned"]
    failure = {"lowest_coverage": sorted(aligned, key=lambda r: r["revealed_coverage"])[:10],
               "survived_below_70_percent": sorted([r for r in aligned if r["survived"] and r["revealed_coverage"] < .7], key=lambda r: r["revealed_coverage"])[:10],
               "highest_loops_below_95_percent": sorted([r for r in aligned if r["revealed_coverage"] < .95], key=lambda r: -r["loop_rate"])[:10]}
    (OUT / "failure-cases.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    selected_rows = []
    optimization = []
    for item in report["protocol"]["checkpoint_manifest"]:
        family, seed = item["family"], item["seed"]
        folder = ROOT / f"runs/{family}-{seed}"
        selected_rows.append({"agent": family, "training_seed": seed, "selected_steps": item["selected_steps"],
                              "validation_auc": item["validation_score"][0], "sha256": item["sha256"]})
        destination = OUT / "training" / f"{family}-{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("validation.csv", "config.yaml", "runtime.json", "best.json", "final.json"):
            shutil.copyfile(folder / name, destination / name)
        logs = []
        for path in folder.glob("logs*/progress.csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                logs.extend(csv.DictReader(stream))
            shutil.copyfile(path, destination / f"{path.parent.name}-progress.csv")
        last = max(logs, key=lambda r: float(r.get("time/total_timesteps") or 0))
        optimization.append({"agent": family, "training_seed": seed, "last_logged_steps": int(float(last["time/total_timesteps"])),
                             "policy_entropy": -float(last["train/entropy_loss"]), "approx_kl": float(last["train/approx_kl"]),
                             "value_loss": float(last["train/value_loss"]), "explained_variance": float(last["train/explained_variance"])})
    write_csv(OUT / "selected-checkpoints.csv", selected_rows)
    write_csv(OUT / "optimization-diagnostics.csv", optimization)
    histories = report["validation_history"]
    lines = ["# Memory alignment: full results and comparison", "",
             "Primary results use fresh held-out map seeds **30000–30099**, all three presets, and frozen validation-selected checkpoints. The original test set (20000–20099) informed the earlier investigation and is not reused as the primary confirmation set.", "",
             "## Protocol and evidence", "",
             "- **4,800 primary episodes**: each PPO variant uses three training seeds × 100 maps × three presets; random uses three policy seeds; frontier runs once per map.",
             "- Same partial observations, budgets, monster rules, and map seeds for all agents. PPO and frontier use greedy/deterministic actions. No new training or tuning was performed on these fresh test results.",
             "- Checkpoints were selected before testing by validation coverage AUC, then discovery, then survival. Each PPO training run completed approximately five million transitions.",
             "- Fixed-budget curves preserve coverage/discovery after death. Primary evaluation ran four CPU processes, each with one PyTorch thread; isolated latency was measured separately in one process.",
             "- Uncertainty separates episode variation, variation between three trained runs, and paired-map bootstrap intervals. Intervals condition on the trained models and are not multiplicity-corrected.",
             "- A further **450 new aligned episodes** assess 50 training-partition seeds; legacy diagnostics are filtered to those same seeds. These maps are not guaranteed to have been encountered during training.", "",
             "## Primary revealed coverage", "",
             "Means ± SD of the three run means (percentage points). Random variation is across policy seeds; frontier has one deterministic run per map.", "",
             "| Agent | Easy | Medium | Hard |", "|---|---:|---:|---:|"]
    for family in ORDER:
        cells = []
        for preset in PRESETS:
            s = detail[f"{family}/{preset}"]["revealed_coverage"]
            cells.append(pct(s["mean"]) + (f" ± {100*s['run_mean_std']:.1f}" if s["run_mean_std"] is not None else ""))
        lines.append(f"| {NAMES[family]} | " + " | ".join(cells) + " |")
    lines += ["", "## Overall comparison", "", "Easy, Medium, and Hard receive equal weight. POI values are fractions of each map's configured count.", "",
              "| Agent | Coverage | Visited | AUC | POIs | Survival | Loop rate | New visited / 100 actions |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family in ORDER:
        r = overall[family]
        lines.append(f"| {NAMES[family]} | {pct(r['revealed_coverage'])} | {pct(r['visited_coverage'])} | {r['coverage_auc']:.3f} | {pct(r['poi_fraction'])} | {pct(r['survived'])} | {pct(r['loop_rate'])} | {100*r['movement_efficiency']:.2f} |")
    lines += ["", "## Complete per-preset metrics", ""]
    for preset in PRESETS:
        lines += [f"### {preset.title()}", "", "| Agent | Visited | AUC | POIs found | POI fraction | Survival | Wall contacts / 100 actions | Loops | Repeated states |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for family in ORDER:
            d = detail[f"{family}/{preset}"]
            lines.append(f"| {NAMES[family]} | {pct(d['visited_coverage']['mean'])} | {d['coverage_auc']['mean']:.3f} | {d['poi_discovered']['mean']:.2f} | {pct(d['poi_fraction']['mean'])} | {pct(d['survived']['mean'])} | {100*d['wall_collision_rate']['mean']:.2f} | {pct(d['loop_rate']['mean'])} | {pct(d['repeated_state_rate']['mean'])} |")
        lines.append("")
    lines += ["## Paired comparisons", "", "Aligned minus comparator; mean differences and 95% map-bootstrap intervals. Coverage and survival use percentage points. AUC remains on its 0–1 scale.", "",
              "| Preset | Comparator | Coverage difference [95% CI] | AUC difference [95% CI] | Survival difference [95% CI] |", "|---|---|---:|---:|---:|"]
    for comparison in report["paired_comparisons"]:
        cells = []
        for metric, scale, digits in (("revealed_coverage", 100, 1), ("coverage_auc", 1, 3), ("survived", 100, 1)):
            s = comparison["metrics"][metric]
            lo, hi = s["map_bootstrap_95ci"]
            cells.append(f"{scale*s['mean_difference']:+.{digits}f} [{scale*lo:+.{digits}f}, {scale*hi:+.{digits}f}]")
        lines.append(f"| {comparison['preset'].title()} | {NAMES[comparison['baseline']]} | " + " | ".join(cells) + " |")
    lines += ["", "The sensitivity comparison using only matched training seeds 11 and 22 is stored in `summary.json`; it excludes the original memory run that resumed after a CUDA failure.", "",
              "## Learning, selection, and stability", "", "| Policy | Mean selected validation AUC | Mean final validation AUC | Selected training steps (seeds 11 / 22 / 33) |", "|---|---:|---:|---|"]
    for family in ORDER[2:]:
        runs = histories[family]["runs"]
        lines.append(f"| {NAMES[family]} | {np.mean([r['selected_auc'] for r in runs]):.3f} | {np.mean([r['final_auc'] for r in runs]):.3f} | " + " / ".join(f"{r['selected_step']/1e6:g}M" for r in runs) + " |")
    lines += ["", "Last logged entropy, KL, critic loss, and explained variance are available in `optimization-diagnostics.csv`. They describe each policy's own training distribution and are auxiliary diagnostics, not substitutes for test performance.", "",
              "## Generalization", "", "| Agent | Fresh test minus training-partition coverage: Easy / Medium / Hard (pp) | Hard minus Medium coverage (pp) |", "|---|---|---:|"]
    for family in ORDER:
        gaps = " / ".join(f"{100*generalization[f'{family}/{p}']['revealed_coverage']['test_minus_train']:+.1f}" for p in PRESETS)
        lines.append(f"| {NAMES[family]} | {gaps} | {100*report['hard_minus_medium'][family]['revealed_coverage']:+.1f} |")
    lines += ["", "Hard is a joint extrapolation in map size, POI count, action budget, and agent-relative spatial extent. A Hard drop alone does not identify which factor causes it. For Hard, the training-partition diagnostic refers only to seed numbers: Hard maps were never used for training.", "",
              "## Isolated CPU inference latency", "", "Includes action selection plus observation/memory preprocessing for the observation used. Excludes environment physics, rendering, and model loading. Two shared validation maps per preset, all three policy/training seeds; 96 episodes total. First action included.", "",
              "| Agent | Actions sampled | Mean (ms) | Median (ms) | p95 (ms) |", "|---|---:|---:|---:|---:|"]
    for family in ORDER:
        l = latency["summary"][family]
        lines.append(f"| {NAMES[family]} | {l['actions']} | {l['total_ms_mean']:.3f} | {l['total_ms_p50']:.3f} | {l['total_ms_p95']:.3f} |")
    lines += ["", "All PPO variants have the same trainable parameter count; alignment adds an integer-coordinate readout before the map CNN. The no-map variant's zero-input branch is inactive, so its effective optimization problem differs. Parameter counts are recorded in `isolated-latency.json`.", "",
              "## Interpretation limits", "",
              "- The original-map versus aligned-map comparison isolates the intended representation change while retaining architecture, rewards, interaction budget, and selection rule. Three seeds provide limited evidence about training-seed uncertainty.",
              "- The no-shaping model uses the original map frame. It is a legacy reward ablation, not an aligned-map reward ablation; it cannot isolate the reward contribution in the redesigned system.",
              "- A high loop rate can include revisiting after most of a map is already revealed. Interpret it jointly with coverage curves and incomplete-coverage failure cases.",
              "- Coverage retained after death credits information gathered before contact. Survival is therefore reported separately; frontier coverage alone does not establish safer behavior.",
              "- Training wall times came from different sessions/load conditions. They are observed costs, not a controlled claim that alignment accelerates training.",
              "- Historical submission artifacts were checksum-verified unchanged. No model was selected or tuned using these new test results.", "",
              "## Figures", "", "![Outcomes](figures/outcomes.png)", "", "![Learning](figures/learning.png)", "", "![Coverage and discovery](figures/coverage-discovery.png)", "", "![Paired differences](figures/paired-differences.png)", "", "![Illustrative paths](figures/illustrative-paths.png)", "",
              "## Files and reproducibility", "",
              "- `protocol.json`: frozen checkpoint hashes, map split, inference settings, and original-artifact checksums.",
              "- `episodes.csv`, `curves.csv`: all primary raw measurements; `aggregate-metrics.csv`: complete metric means, episode SDs, run SDs, and percentiles.",
              "- `summary.json`: paired intervals, run-level results, learning histories, and seed-11/22 sensitivity.",
              "- `selected-checkpoints.csv`, `training/`: selected steps, validation histories, configuration, training logs, and recorded runtimes.",
              "- `generalization.json`, `training-partition/`: matched training-partition diagnostic.",
              "- `isolated-latency.json`, `isolated-latency-samples.csv`: independent CPU latency measurements.",
              "- `failure-cases.json`, `qualitative/`: failures and fixed-map action traces; the illustrated map is Hard seed 30000 with training seed 11, not a claim of typical behavior.", ""]
    a, old, local, frontier = [overall[k] for k in ("main_agent_aligned", "main", "no_memory", "frontier")]
    hard_local = next(r for r in report["paired_comparisons"] if r["preset"] == "hard" and r["baseline"] == "no_memory")
    survival_ci = hard_local["metrics"]["survived"]["map_bootstrap_95ci"]
    sensitivity = {r["preset"]: r["metrics"]["revealed_coverage"]["mean_difference"] for r in report["sensitivity_training_seeds_11_22"] if r["baseline"] == "main"}
    val = np.mean([r["selected_auc"] for r in histories["main_agent_aligned"]["runs"]])
    em = np.mean([detail[f"main_agent_aligned/{p}"]["coverage_auc"]["mean"] for p in ("easy", "medium")])
    analysis = ["## Findings and interpretation", "",
        f"**The alignment redesign makes memory useful for this PPO agent.** Mean coverage rises from {pct(old['revealed_coverage'])} with the original map to {pct(a['revealed_coverage'])} with the aligned map, a gain of {100*(a['revealed_coverage']-old['revealed_coverage']):.1f} percentage points. It also exceeds the mapless PPO by {100*(a['revealed_coverage']-local['revealed_coverage']):.1f} points. This is a representation change using the same {latency['parameters']['main_agent_aligned']['total']:,}-parameter network and observation-derived information, not additional hidden-map access or a recurrent policy.", "",
        f"The gain includes actual movement: visited coverage is {pct(a['visited_coverage'])}, versus {pct(local['visited_coverage'])} without map input and {pct(old['visited_coverage'])} with the original map. POI discovery rises to {pct(a['poi_fraction'])}, versus {pct(local['poi_fraction'])} and {pct(old['poi_fraction'])}, respectively. Because POIs are placed across traversable terrain, discovery gains largely track better coverage; these results do not establish a specialized hidden-POI search strategy.", "",
        "**The gains are consistent across presets and robust to the resumed reference run.** Paired-map intervals for coverage and AUC improvements over both original-map and mapless PPO remain above zero on all three presets. Excluding the original seed-33 run that resumed after a CUDA failure still gives coverage gains over original memory of " + " / ".join(f"{100*sensitivity[p]:+.1f}" for p in PRESETS) + " percentage points (Easy / Medium / Hard). These intervals condition on the trained runs; only three training seeds were tested.", "",
        f"**Frontier remains the strongest overall exploration reference.** It reaches {pct(frontier['revealed_coverage'])} mean coverage and {frontier['coverage_auc']:.3f} AUC, versus {pct(a['revealed_coverage'])} and {a['coverage_auc']:.3f} for aligned PPO. Aligned PPO has a modest Easy endpoint-coverage advantage, while Easy AUC is effectively tied. Medium endpoint uncertainty overlaps zero; Medium AUC and both Hard exploration metrics favor frontier. Thus the redesign improves the learned policy substantially without establishing overall superiority to the deterministic explorer.", "",
        f"**Survival is a tradeoff.** Aligned PPO survives {pct(a['survived'])} of episodes versus {pct(frontier['survived'])} for frontier, but mapless PPO survives {pct(local['survived'])}. On Hard, aligned survival is {pct(detail['main_agent_aligned/hard']['survived']['mean'])} versus {pct(detail['no_memory/hard']['survived']['mean'])} without map input: {100*hard_local['metrics']['survived']['mean_difference']:+.1f} pp, with a paired-map interval [{100*survival_ci[0]:+.1f}, {100*survival_ci[1]:+.1f}]. It would be incorrect to describe alignment as improving safety across all PPO comparisons.", "",
        f"**Loops and blocked movement remain weaknesses.** Overall loop rate falls from {pct(old['loop_rate'])} to {pct(a['loop_rate'])}, but remains above mapless PPO's {pct(local['loop_rate'])}. On Hard it is {pct(detail['main_agent_aligned/hard']['loop_rate']['mean'])}, versus {pct(detail['main/hard']['loop_rate']['mean'])} for original memory. Wall contacts occur on {pct(a['wall_collision_rate'])} of all actions, versus {pct(local['wall_collision_rate'])} without map input. More forward movement changes exposure, so this per-all-actions rate does not isolate wall-avoidance accuracy per forward attempt.", "",
        "Rare severe stalls remain: on Hard map 30029, training seed 11, aligned PPO survived the entire budget but revealed only 10.0% of the map and recorded a 99.9% loop rate. This is a post-hoc failure example, not typical performance. It shows why survival, mean coverage, and qualitative failures must be reported together.", "",
        f"**Training is more stable, while Hard remains an extrapolation challenge.** Mean selected/final validation AUC is {val:.3f}/{np.mean([r['final_auc'] for r in histories['main_agent_aligned']['runs']]):.3f}, compared with {np.mean([r['selected_auc'] for r in histories['main']['runs']]):.3f}/{np.mean([r['final_auc'] for r in histories['main']['runs']]):.3f} for original memory. Validation covers Easy and Medium only: the comparable fresh-test Easy/Medium AUC is {em:.3f}; the {a['coverage_auc']:.3f} three-preset score also includes Hard. Hard coverage drops {abs(100*report['hard_minus_medium']['main_agent_aligned']['revealed_coverage']):.1f} pp from Medium, versus {abs(100*report['hard_minus_medium']['main']['revealed_coverage']):.1f} pp for the original map. The remaining gap cannot be attributed to spatial extrapolation alone without another controlled experiment.", "",
        f"**The reward ablation is informative but incomplete.** The legacy no-seen/visit-bonus policy averages {pct(overall['no_novelty']['revealed_coverage'])} coverage, close to random's {pct(overall['random']['revealed_coverage'])}, and loops on {pct(overall['no_novelty']['loop_rate'])} of actions. It retains POI and survival-related rewards. Its comparison with original-map PPO supports exploration shaping in that frame; an aligned no-bonus model has not been trained, so the interaction with alignment remains unmeasured.", "",
        f"**Alignment has a measurable CPU cost without extra trainable capacity.** Isolated mean inference plus input preprocessing is {latency['summary']['main_agent_aligned']['total_ms_mean']:.3f} ms, versus {latency['summary']['main']['total_ms_mean']:.3f} ms for original memory and {latency['summary']['no_memory']['total_ms_mean']:.3f} ms without map input. The map encoder remains {latency['parameters']['main_agent_aligned']['map_encoder']:,} parameters. A smaller encoder is a justified next isolated experiment, not an improvement already demonstrated here.", "",
        "**Practical conclusion:** retain aligned memory as the strongest learned exploration baseline tested here. Keep frontier as the stronger aggregate exploration reference, and retain mapless PPO as a survival reference. Further work should test a compact aligned representation separately and address incomplete-coverage loops and threat interactions. Sampled action selection has not been evaluated for the redesigned model in this comparison. These test maps are now observed; further model selection should use validation and reserve a new confirmation set.", ""]
    lines[4:4] = analysis
    scripts = ("evaluate_aligned_comparison", "analyze_aligned_comparison", "evaluate_aligned_train_partition",
               "profile_aligned_comparison", "qualitative_aligned_comparison", "write_aligned_comparison_report", "finish_aligned_comparison", "verify_aligned_curves")
    reproduction = OUT / "reproduction"
    reproduction.mkdir(exist_ok=True)
    for name in scripts:
        source = Path(__file__).resolve().parent / f"{name}.py"
        destination = reproduction / source.name
        if source != destination:
            shutil.copyfile(source, destination)
    lines += ["Reproduction scripts are archived in `reproduction/` and locate the project root from their checkout location. Run the primary evaluator first, then the finishing pipeline. Existing completed parts are reused only when the frozen checkpoint/protocol manifest matches. Raw episode timings from parallel workers are not used as isolated latency estimates.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    index = {"primary_episodes": 4800, "new_train_partition_episodes": 450, "latency_episodes": 96,
             "original_artifacts_unchanged": True, "report": "report.md", "sha256": {}}
    for name in ("protocol.json", "episodes.csv", "curves.csv", "summary.json", "aggregate-metrics.csv", "generalization.json", "isolated-latency.json", "report.md"):
        index["sha256"][name] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    index["reproduction_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reproduction.glob("*.py")}
    (OUT / "manifest.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
