"""Generate plots, a five-page evidence-based report, replays, and a 3-minute demo.

No synthetic scores or invented conclusions: every reported number comes from
the supplied evaluation CSV/JSON. Pilot outputs are prominently labeled.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import yaml
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.rendering import font

COLORS = {"random": "#8a97a8", "frontier": "#38a58c", "main": "#377ad3",
          "no_memory": "#b37ac2", "no_novelty": "#d59347"}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plots(report, curves, output, checkpoint):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summary = report["summary"]
    agents = sorted({key.split("/")[0] for key in summary}, key=lambda a: ["random", "frontier", "main", "no_memory", "no_novelty"].index(a))
    presets = report["presets"]
    x = np.arange(len(presets))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.25))
    width = 0.8 / len(agents)
    for index, agent in enumerate(agents):
        for ax, metric, title in zip(axes, ("revealed_coverage", "survived"), ("Revealed floor coverage", "Survival to action limit")):
            values = [summary[f"{agent}/{p}"]["metrics"][metric]["mean"] for p in presets]
            ax.bar(x + (index - (len(agents) - 1) / 2) * width, values, width, label=agent, color=COLORS[agent])
            ax.set(title=title, xticks=x, xticklabels=presets, ylim=(0, 1.06))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(presets), figsize=(10, 2.9), squeeze=False)
    for ax, preset in zip(axes[0], presets):
        for agent in agents:
            selected = [r for r in curves if r["agent"] == agent and r["preset"] == preset]
            steps = sorted({float(r["budget_fraction"]) for r in selected})
            means = [np.mean([float(r["coverage"]) for r in selected if float(r["budget_fraction"]) == s]) for s in steps]
            ax.plot(steps, means, label=agent, color=COLORS[agent])
        ax.set(title=preset.title(), xlabel="Fraction of action budget", ylim=(0, 1.03), xlim=(0, 1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0, 0].set_ylabel("Mean revealed coverage")
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "coverage-curves.png", dpi=180)
    plt.close(fig)
    # Show training/validation history for the actual packaged checkpoint's run.
    checkpoint = Path(checkpoint)
    validation = checkpoint.parent / "validation.csv"
    if validation.exists():
        rows = read_csv(validation)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot([int(r["step"]) for r in rows], [float(r["coverage_auc"]) for r in rows], "o-", color=COLORS["main"])
        ax.set(xlabel="Training transitions", ylabel="Validation coverage AUC", ylim=(0, 1))
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output / "learning.png", dpi=180)
        plt.close(fig)


def report_pdf(report, checkpoint, output, complete):
    metadata = json.loads(Path(checkpoint).with_suffix(".json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontName="Helvetica", fontSize=9, leading=13,
                              spaceAfter=7, alignment=TA_LEFT))
    styles["Title"].fontSize = 25
    styles["Title"].leading = 29
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading2"].fontSize = 11
    story = []

    def p(text):
        story.append(Paragraph(text, styles["BodySmall"]))

    def heading(text):
        story.append(Paragraph(text, styles["Heading1"]))

    def table(rows, widths=None, padding=6):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#193756")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#eef3f8"), colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 9))

    status = "FULL EXPERIMENT MATRIX" if complete else "ENGINEERING PILOT - FINAL EXPERIMENTS INCOMPLETE"
    story.append(Paragraph("Learning to explore<br/>with partial observations", styles["Title"]))
    p("MiniGrid exploration agent | Technical report | Python 3.11")
    p(f"<b>{status}</b>")
    heading("1. Problem and information boundary")
    p("We model exploration as a partially observed decision process. The agent must reveal traversable terrain, discover points of interest (POIs), and survive a moving patrol within an action budget. The environment is an extension of Farama's MiniGrid, rather than a replacement simulator. Its native grid, visibility, turn/forward transitions, and renderer remain the foundation.")
    p("The policy receives a wall-occluded 7 x 7 symbolic view, heading, movement/contact feedback, remaining time, and a map reconstructed from those signals. The mapping module has no environment reference. Absolute map coordinates, hidden POIs, hidden patrol positions, and ground-truth coverage denominators never enter policy inputs or rewards.")
    p("A fixed 55 x 55 memory canvas is centered at the initial position. Known walls/floor, discovered POIs, clipped occupancy counts, the agent marker, and currently visible monsters form seven channels. The current view is five one-hot channels; eight scalars encode heading and feedback. Monster occupancy is cleared every observation, avoiding fictitious current positions from stale sightings.")
    heading("2. Environment design")
    table([["Preset", "Size", "POIs", "Action budget", "Monster period"],
           ["Easy", "15 x 15", "2", "512", "3 actions"],
           ["Medium", "21 x 21", "3", "1,024", "2 actions"],
           ["Hard", "27 x 27", "5", "1,536", "2 actions"]])
    p("A seeded depth-first maze supplies corridors and dead ends. Rectangular rooms and sparse additional connections add open areas, junctions, and cycles. Connectivity and reachable placements are checked before simulation. One red monster starts at graph distance at least six, walks orthogonally, avoids immediate reversal when alternatives exist, and respects walls. Agent motion precedes collision checking, then the scheduled monster move and a second check. Contact is fatal; the ordinary budget limit is a truncation. Map and patrol RNG streams are independent.")

    story.append(PageBreak())
    heading("3. Rewards and policy")
    p("The four actions are left, right, forward, and wait. The reward is computed from observation history, with each newly revealed floor cell, visited cell, and POI counted once per episode. Initial observations initialize history but pay no reward.")
    table([["Component", "Weight", "Observable event"],
           ["New floor revealed", "+0.02", "Previously unknown traversable tile seen"],
           ["New cell visited", "+0.01", "First occupancy inferred through odometry"],
           ["New POI discovered", "+1.00", "First visible green floor marker"],
           ["Every action", "-0.001", "Action taken"],
           ["Wall contact", "-0.02", "Contact feedback"],
           ["Monster contact", "-1.00", "Fatal contact feedback"]], [145, 55, 300])
    p("No reward clipping or running reward normalization is used. Turning can legitimately reveal a new area once; repeated turns, revisits, and re-sightings cannot repeatedly earn discovery reward. The time cost discourages camping. Remaining risks include prioritizing an immediately rewarding discovery over survival and learning short cycles; therefore outcomes and loop metrics are reported independently of training return.")
    p("Stable-Baselines3 PPO uses separate encoders for the local view (16/32-channel convolutions, then 64 features) and reconstructed map (32/64/64 stride-two convolutions, then 128 features). Their outputs plus eight scalars feed separate two-layer 128-unit actor and critic heads. This approach makes memory auditable and avoids adding recurrent-state management to a short engineering assignment. It assumes exact movement feedback and a known maximum map size.")
    p("Default training: 16 environments; 256 steps per rollout per environment; minibatches of 256; four epochs; learning rate 0.0003; gamma 0.99; GAE lambda 0.95; clipping 0.2; entropy coefficient 0.01; target KL 0.03; value coefficient 0.5; gradient norm bound 0.5. Training samples an equal-probability mixture of Easy and Medium maps. Advantage normalization is PPO's default. KL, entropy, losses, coverage, discoveries, survival, and loops are logged.")
    p("The planned matrix trains main, no-memory, and no-novelty variants for seeds 11, 22, and 33, each to five million transitions. PPO rounds budgets to complete rollouts. The no-memory variant zeros the policy's map branch while retaining internal discovery accounting; no-novelty removes the floor-reveal and first-visit bonuses. Everything else is held fixed.")
    p(f"Packaged checkpoint: variant <b>{metadata['variant']}</b>, training seed <b>{metadata['seed']}</b>, selected step <b>{metadata['training_steps']:,}</b>. Runtime dependencies: MiniGrid {metadata['dependencies']['minigrid']}; SB3 {metadata['dependencies']['stable-baselines3']}; PyTorch {metadata['dependencies']['torch']}.")
    hardware_file = ROOT / "results" / "hardware.json"
    if hardware_file.exists():
        hardware = json.loads(hardware_file.read_text(encoding="utf-8"))
        p(f"Recorded training host: {hardware['gpu_name']} ({hardware['gpu_memory_bytes'] / 2**30:.1f} GiB), {hardware['logical_cpu_cores']} logical CPU cores, NVIDIA driver {hardware['nvidia_driver']}. PyTorch uses one CPU thread per training/inference process; policy evaluation uses CPU. See results/hardware.json and per-run runtime.json for provenance and measured throughput.")

    story.append(PageBreak())
    heading("4. Evaluation methodology")
    p("Map-seed partitions are training 0-9,999, validation 10,000-10,049, and held-out test 20,000-20,099. Training RNG seeds are distinct from map seeds. A fixed seed and action sequence reproduces the trajectory. Checkpoints are selected on validation mean coverage AUC, with POI discovery and then survival as lexicographic tie-breakers. Test outcomes do not select checkpoints.")
    p("The random baseline uniformly samples all four actions, with policy seeds 11, 22, and 33. The deterministic frontier baseline uses BFS only through observed floor, turns to inspect frontier cells, uses fixed tie-breaking, and treats visible monsters and adjacent cells as unsafe. Both baselines use the same seeds, budgets, and information restrictions as PPO. The frontier agent is an intentionally strong reference, not an oracle.")
    table([["Metric", "Definition"],
           ["Revealed / visited coverage", "Unique observed / occupied floor divided by true reachable floor"],
           ["Coverage AUC", "Trapezoidal coverage over the full action budget, normalized by budget"],
           ["POI discovery", "Unique observed POIs divided by total generated POIs"],
           ["Survival / completion", "Reaches the configured action limit without fatal contact"],
           ["Movement efficiency", "New occupied cells after reset divided by actions"],
           ["Repeated-state / loop rate", "Recent (position, heading) repeat / repeat with no discovery, window 16"],
           ["Threat / wall interactions", "Fatal contacts and wall-contact count/rate"],
           ["Inference latency", "Action selection plus observation/memory preprocessing; engine excluded"]], [135, 365])
    p("The evaluator alone knows the true terrain and total POI count. After death, final coverage is held constant through the remaining budget when computing AUC and curves, so early deaths cannot improve averages by excluding later timesteps. Latency includes the first action and is reported as mean, median, and p95. CSVs preserve per-episode results and termination reasons.")
    p("Episode standard deviation and standard deviation of training-run means are separate. Paired bootstrap intervals average runs per map and resample shared maps; they quantify map variation, not full training uncertainty. Hard maps test extrapolation to a size absent from training. Training-partition diagnostic evaluation, when exported, uses the same evaluation command with --split train; no train/test gap is inferred from training returns.")
    p(f"This report's actual evaluation: <b>{report['split']}</b>; {len(report['seeds'])} map seeds per preset ({report['seeds'][0]}-{report['seeds'][-1]}); {len(report['summary'])} agent/preset groups. Raw sources: episodes.csv, curves.csv, summary.json in the evaluation output directory.")
    if not complete:
        p("<b>Evidence limitation:</b> This is a runnable engineering pilot. It does not establish the planned three-run, five-million-transition result or a reliable generalization claim. Full-matrix automation is supplied separately; preliminary scores must not be presented as final assignment findings.")

    story.append(PageBreak())
    heading("5. Measured results")
    rows = [["Agent / preset", "Episodes", "Coverage %", "AUC", "POI", "Survival"]]
    for key, value in sorted(report["summary"].items()):
        m = value["metrics"]
        rows.append([key, str(value["episodes"]), f"{100*m['revealed_coverage']['mean']:.1f} +/- {100*m['revealed_coverage']['episode_std']:.1f}",
                     f"{m['coverage_auc']['mean']:.3f}", f"{m['poi_fraction']['mean']:.1%}", f"{m['survived']['mean']:.1%}"])
    table(rows, [150, 55, 73, 60, 72, 90], padding=4)
    story.append(ReportImage(str(output / "figures" / "outcomes.png"), width=6.75 * inch, height=2.19 * inch))
    p("Coverage is mean +/- episode standard deviation in percentage points. Bars show means. Run means, run-mean variation, wall contacts, loops, latency, and paired comparison intervals are in summary.json. No claim that PPO beats the frontier baseline is assumed.")
    for comparison in report.get("paired_comparisons", []):
        if comparison["agent"] == "main" and comparison["baseline"] == "random":
            lo, hi = comparison["map_bootstrap_95ci"]
            p(f"{comparison['preset'].title()}: main minus random coverage AUC = {comparison['coverage_auc_difference']:+.3f}; paired-map 95% interval [{lo:+.3f}, {hi:+.3f}] across {comparison['paired_maps']} maps.")
    for variant in ("no_memory", "no_novelty"):
        comparable = [preset for preset in ("easy", "medium") if f"main/{preset}" in report["summary"] and f"{variant}/{preset}" in report["summary"]]
        if comparable:
            difference = np.mean([report["summary"][f"main/{preset}"]["metrics"]["coverage_auc"]["mean"] - report["summary"][f"{variant}/{preset}"]["metrics"]["coverage_auc"]["mean"] for preset in comparable])
            p(f"Ablation: main minus {variant} mean coverage AUC across training-sized presets = {difference:+.3f}. Run variation is retained in the raw summary.")

    story.append(PageBreak())
    heading("6. Behavior, limitations, and reproducibility")
    story.append(ReportImage(str(output / "figures" / "coverage-curves.png"), width=6.75 * inch, height=1.96 * inch))
    if (output / "figures" / "learning.png").exists():
        story.append(ReportImage(str(output / "figures" / "learning.png"), width=6.75 * inch, height=1.69 * inch))
    failures = report.get("failure_examples", [])
    if failures:
        case = failures[0]
        p(f"Failure inspection: main seed {case['training_seed']}, {case['preset']} map {case['seed']}, reached {float(case['revealed_coverage']):.1%} coverage with {float(case['loop_rate']):.1%} looping and termination '{case['termination']}'. Full action traces, evaluator/memory panels, final frames, and separately ranked low-coverage, high-loop, and fatal cases support inspection. Replay metadata records skipped video frames and every action.")
    else:
        p("Seeded qualitative artifacts include evaluator/memory panels, full action traces, final frames, and separately ranked low-coverage, high-loop, and fatal cases. Replay metadata records skipped video frames and every action.")
    p("Known limitations: perfect compass/contact odometry simplifies localization; only one procedural generator family and one patrol rule are studied; frontier avoidance is reactive; mortality makes exploration/survival a genuine trade-off; deterministic PPO inference may expose cycles that stochastic training hides. The fixed map canvas supports sizes through 27 only. CPU latency is machine-dependent. Checkpoint resume restarts environment sampling and is not bitwise continuation. Different map seeds do not guarantee different geometry, although generator diversity is tested.")
    p("The interactive Tkinter inspector exposes seeds, presets, size, POIs, horizon, patrol interval, agent/checkpoint selection, playback speed, and start/pause/reset/single-step controls. Its evaluator view is clearly labeled. A PyInstaller Windows CPU build includes the model, metadata, and Python runtime; smoke mode runs without a GPU or retraining.")
    p("Reproduce: install the locked CPU or CUDA dependencies, run pytest tests_exploration, train with configs/main.yaml, evaluate saved checkpoints using the committed seed partition, then export this report and demo. scripts/run_matrix.py resumes interrupted runs and evaluates the held-out partition only after every requested training run completes. Logs, configs, checkpoint checksums, runtime metadata, and raw episode metrics are retained.")
    gaps = report.get("generalization", {})
    if gaps:
        size_gap = gaps["hard_minus_medium_test"].get("main", {}).get("coverage_auc")
        partition_gaps = [value["coverage_auc"] for key, value in gaps["test_minus_training_partition"].items() if key in {"main/easy", "main/medium"}]
        if size_gap is not None and partition_gaps:
            p(f"Generalization: main test minus training-partition coverage AUC averages {np.mean(partition_gaps):+.3f} on Easy/Medium; Hard minus Medium test AUC is {size_gap:+.3f}. Exact coverage, POI, and survival gaps are in generalization.json. Training-partition samples are not guaranteed to have been encountered by every run.")
    p("Future work: uncertain odometry, beliefs about hidden threats, recurrent-policy comparisons, richer map families, and more independent training seeds. These are extensions, not evidence supplied here.")
    p("References and attribution: Farama Foundation MiniGrid (Apache 2.0), github.com/Farama-Foundation/Minigrid; Stable-Baselines3 (MIT), stable-baselines3.readthedocs.io; PyTorch (BSD-style); Gymnasium (MIT). See ATTRIBUTION.md, upstream CITATION.cff, and bundled third-party license files. PPO: Schulman et al., Proximal Policy Optimization Algorithms (2017), arXiv:1707.06347.")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#65758b"))
        canvas.drawString(42, 24, "MiniGrid exploration | " + ("Full matrix" if complete else "Preliminary engineering pilot"))
        canvas.drawRightString(570, 24, str(doc.page))

    SimpleDocTemplate(str(output / "technical-report.pdf"), pagesize=(612, 792),
                      rightMargin=42, leftMargin=42, topMargin=36, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)


def slide(title, lines, image=None, footer=""):
    canvas = Image.new("RGB", (1280, 720), "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 30), "MINIGRID / EXPLORER", font=font(19, True), fill="#74d4c1")
    draw.text((48, 76), title, font=font(34, True), fill="#f2f5fb")
    y = 145
    for line in lines:
        draw.text((48, y), line, font=font(22), fill="#bdcbe0")
        y += 39
    if image:
        picture = Image.open(image).convert("RGB")
        picture.thumbnail((1184, 720 - y - 65), Image.Resampling.LANCZOS)
        canvas.paste(picture, ((1280 - picture.width) // 2, y + 10))
    draw.text((48, 678), footer, font=font(17), fill="#93a4bd")
    return np.asarray(canvas)


def demo(output, complete):
    fps = 12
    writer = imageio.get_writer(output / "demo.mp4", fps=fps, codec="libx264", quality=7,
                                macro_block_size=1, ffmpeg_log_level="error")
    status = "Full benchmark" if complete else "Engineering pilot: full experiment matrix pending"

    def still(frame, seconds):
        for _ in range(seconds * fps):
            writer.append_data(frame)

    def clip(path, seconds):
        reader = imageio.get_reader(path)
        try:
            length = reader.count_frames()
            # Show an entire seeded episode within its storyboard segment.
            for index in np.linspace(0, max(0, length - 1), seconds * fps).astype(int):
                writer.append_data(reader.get_data(int(index)))
        finally:
            reader.close()

    try:
        still(slide("Learning to explore an unknown map", [status,
                    "MiniGrid + PPO + memory built only from observations",
                    "Coverage, discoveries, survival, efficiency, and generalization"],
                    footer="3-minute silent demo / explanatory captions / seeded replay footage"), 15)
        ui_image = ROOT / "docs" / "ui.png"
        still(slide("Interactive Python inspection", ["Set seed and difficulty; select baseline or checkpoint.",
                    "Start, pause, reset, single-step, and inspect both views."],
                    image=ui_image if ui_image.exists() else output / "replays" / "frontier-medium" / "final.png",
                    footer="Evaluator view is ground truth. Agent memory is restricted to observations."), 20)
        clip(output / "replays" / "frontier-medium" / "episode.mp4", 30)
        clip(output / "replays" / "ppo-easy" / "episode.mp4", 30)
        clip(output / "replays" / "ppo-medium" / "episode.mp4", 30)
        clip(output / "replays" / "threat-case" / "episode.mp4", 20)
        still(slide("Measured outcomes", [status, "Same maps, action budgets, and information restrictions for every agent."],
                    image=output / "figures" / "outcomes.png", footer="Raw metrics and variation: episodes.csv and summary.json"), 20)
        still(slide("Reproduce and inspect", ["smoke: load saved policy and run on CPU, without retraining",
                    "evaluate: fixed seed manifests, raw episode CSVs, paired comparisons",
                    "replay: complete action traces and synchronized map/memory video",
                    "Known limits: perfect odometry, one patrol rule, bounded map sizes"],
                    footer="See README.md and the five-page technical report."), 15)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="submission")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Skip the default portable CPU build")
    args = parser.parse_args()
    import torch

    from exploration.replay import replay
    torch.set_num_threads(1)
    result_dir = Path(args.results)
    report = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(result_dir / "episodes.csv")
    curves = read_csv(result_dir / "curves.csv")
    output = Path(args.output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    run_seeds = {(r["agent"], r["training_seed"]) for r in rows if r["training_seed"]}
    trained = [a for a in report.get("agents", []) if a.get("training_seed") is not None]
    complete = (report["split"] == "test" and len(report["seeds"]) == 100 and len(run_seeds) == 9
                and len(trained) == 9 and all(a.get("completed_training_steps", 0) >= 5_000_000 for a in trained))
    if complete:
        diagnostic = result_dir / "train-diagnostic"
        if not (diagnostic / "summary.json").exists():
            evaluation_config = result_dir / "evaluation-config.yaml"
            evaluation_config.write_text(yaml.safe_dump(trained[0]["training_config"]), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "exploration", "evaluate", "--config", str(evaluation_config),
                            "--agents", "random", "frontier", "ppo", "--checkpoints",
                            *[a["checkpoint"] for a in trained], "--split", "train", "--count", "100",
                            "--presets", *report["presets"], "--output", str(diagnostic)], check=True)
        train_report = json.loads((diagnostic / "summary.json").read_text(encoding="utf-8"))
        generalization = {"note": "Training-partition seeds are a diagnostic sample, not a guarantee that every particular map was encountered by every run.",
                          "test_minus_training_partition": {}, "hard_minus_medium_test": {}}
        for key, group in report["summary"].items():
            if key in train_report["summary"]:
                generalization["test_minus_training_partition"][key] = {
                    metric: group["metrics"][metric]["mean"] - train_report["summary"][key]["metrics"][metric]["mean"]
                    for metric in ("revealed_coverage", "coverage_auc", "poi_fraction", "survived")}
        for key in report["summary"]:
            agent, preset = key.split("/")
            if preset == "hard" and f"{agent}/medium" in report["summary"]:
                generalization["hard_minus_medium_test"][agent] = {
                    metric: report["summary"][key]["metrics"][metric]["mean"] - report["summary"][f"{agent}/medium"]["metrics"][metric]["mean"]
                    for metric in ("revealed_coverage", "coverage_auc", "poi_fraction", "survived")}
        (result_dir / "generalization.json").write_text(json.dumps(generalization, indent=2), encoding="utf-8")
        report["generalization"] = generalization
    report["failure_examples"] = sorted([r for r in rows if r["agent"] == "main"], key=lambda r: float(r["revealed_coverage"]))[:3]
    plots(report, curves, output / "figures", args.checkpoint)
    report_pdf(report, args.checkpoint, output, complete)
    selected = sorted([r for r in rows if r["agent"] not in {"random", "frontier"}], key=lambda r: float(r["coverage_auc"]))
    failures = {"lowest_coverage": selected[:3],
                "highest_loop_rate": sorted(selected, key=lambda r: -float(r["loop_rate"]))[:3],
                "fatal_contacts": [r for r in selected if r["termination"] == "monster"][:3]}
    (output / "failure-cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    checkpoint = Path(args.checkpoint)
    target = ROOT / "checkpoints"
    target.mkdir(exist_ok=True)
    if checkpoint.resolve() != (target / "demo.zip").resolve():
        shutil.copy2(checkpoint, target / "demo.zip")
        shutil.copy2(checkpoint.with_suffix(".json"), target / "demo.json")
    curated = []
    for item in trained:
        source = Path(item["checkpoint"])
        stem = f"{item['agent']}-{item['training_seed']}"
        shutil.copy2(source, target / f"{stem}.zip")
        meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        meta["completed_training_steps"] = item["completed_training_steps"]
        (target / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        curated.append({"checkpoint": f"{stem}.zip", "variant": item["agent"], "seed": item["training_seed"],
                        "selected_steps": item["training_steps"], "completed_steps": item["completed_training_steps"]})
        training_output = result_dir / "training" / stem
        training_output.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "runtime.json", "validation.csv", "training_episodes.csv"):
            if (source.parent / name).exists():
                shutil.copy2(source.parent / name, training_output / name)
    (target / "index.json").write_text(json.dumps(curated, indent=2), encoding="utf-8")
    for filename in ("episodes.csv", "curves.csv", "summary.json"):
        shutil.copy2(result_dir / filename, output / filename)
    if (result_dir / "generalization.json").exists():
        shutil.copy2(result_dir / "generalization.json", output / "generalization.json")
    if not args.skip_video:
        seed = report["seeds"][0]
        replay("frontier", None, seed, "medium", output / "replays" / "frontier-medium")
        replay("ppo", args.checkpoint, seed, "easy", output / "replays" / "ppo-easy")
        replay("ppo", args.checkpoint, seed, "medium", output / "replays" / "ppo-medium")
        fatal = next((r for r in rows if r["agent"] == "random" and r["termination"] == "monster"), None)
        replay("random", None, int(fatal["seed"]) if fatal else seed,
               fatal["preset"] if fatal else "medium", output / "replays" / "threat-case")
        demo(output, complete)
        subprocess.run([sys.executable, "scripts/verify_artifacts.py", "--submission", str(output)], cwd=ROOT, check=True)
    manifest = {"status": "full_matrix" if complete else "preliminary_pilot", "evaluation_source": str(result_dir),
                "checkpoint_source": str(checkpoint), "completed_training_runs": len(run_seeds),
                "report": str(output / "technical-report.pdf"), "demo_seconds": None if args.skip_video else 180}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.build or not args.skip_build:
        cpu = ROOT.parent / ".cpu-venv" / "Scripts" / "python.exe"
        subprocess.run([str(cpu), "scripts/build_windows.py", "--checkpoint", str(target / "demo.zip")], cwd=ROOT, check=True)
        if complete:
            subprocess.run([sys.executable, "scripts/finalize_release.py", "--commit-results"], cwd=ROOT, check=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
