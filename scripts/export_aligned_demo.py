"""Produce a four-minute aligned-policy demo without altering the frozen study.

The checkpoint is selected by validation score. Qualitative maps are declared
below; recorded benchmark means are never recomputed from these examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.memory import make_env
from exploration.rendering import font
from exploration.replay import replay

FPS = 12
SIZE = (1280, 720)
STUDY = ROOT / "results/aligned-reward-followup"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slide(title, lines, footer="", picture=None):
    canvas = Image.new("RGB", SIZE, "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "MINIGRID / ALIGNED EXPLORATION", font=font(19, True), fill="#74d4c1")
    draw.text((42, 70), title, font=font(34, True), fill="#f2f5fb")
    y = 137
    for line in lines:
        draw.text((42, y), line, font=font(24), fill="#dbe5f3")
        y += 39
    if picture is not None:
        image = Image.open(picture).convert("RGB")
        image.thumbnail((1196, 645 - y), Image.Resampling.LANCZOS)
        canvas.paste(image, ((1280 - image.width) // 2, y + 8))
    draw.text((42, 678), footer, font=font(17), fill="#93a4bd")
    return np.asarray(canvas)


def prepare(output):
    import torch

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((ROOT / "checkpoints/aligned/index.json").read_text())
    selected = max(index, key=lambda x: tuple(x["validation_score"]))
    checkpoint = ROOT / "checkpoints/aligned" / selected["checkpoint"]
    assert digest(checkpoint) == selected["sha256"]
    assert selected["seed"] == 33 and selected["memory_frame"] == "agent_aligned"
    cases = [("aligned-easy", "ppo", checkpoint, 10000, "easy"),
             ("aligned-medium", "ppo", checkpoint, 10000, "medium"),
             ("aligned-hard", "ppo", checkpoint, 10000, "hard"),
             ("frontier-medium", "frontier", None, 10000, "medium"),
             ("aligned-contact", "ppo", checkpoint, 30000, "hard"),
             ("loop-example-seed11", "ppo", ROOT / "checkpoints/aligned/main_agent_aligned-11.zip", 30029, "hard")]
    traces = {}
    for name, agent, model, seed, preset in cases:
        folder = output / "replays" / name
        trace_path = folder / "replay.json"
        expected_hash = digest(model) if model else None
        historical = ROOT / "submission/aligned/replays" / name
        if not trace_path.exists() and (historical / "replay.json").exists():
            shutil.copytree(historical, folder, dirs_exist_ok=True)
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            assert trace["checkpoint_sha256"] == expected_hash and trace["seed"] == seed and trace["preset"] == preset
            assert (folder / "episode.mp4").is_file() and (folder / "final.png").is_file()
        else:
            trace = replay(agent, str(model) if model else None, seed, preset, folder)
        traces[name] = trace

    # Procedural-generation panel: fixed maps, not cherry-picked search results.
    canvas = Image.new("RGB", SIZE, "#0b1120")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "MINIGRID / PROCEDURAL ENVIRONMENT", font=font(19, True), fill="#74d4c1")
    draw.text((42, 70), "Connected rooms, corridors, and a moving patrol", font=font(32, True), fill="#f2f5fb")
    for i, preset in enumerate(("easy", "medium", "hard")):
        env = make_env(preset)
        try:
            env.reset(seed=10000)
            image = Image.fromarray(env.unwrapped.get_frame(tile_size=20))
            image.thumbnail((365, 430), Image.Resampling.NEAREST)
            x = 35 + i * 420
            canvas.paste(image, (x + (365 - image.width) // 2, 190 + (430 - image.height) // 2))
            base = env.unwrapped
            label = f"{preset.title()}: {base.width} x {base.height}, {base.poi_count} POIs"
            draw.text((x, 145), label, font=font(23, True), fill="#dbe5f3")
            draw.text((x, 625), f"Budget {base.max_steps} | seed 10000", font=font(20), fill="#bdcbe0")
        finally:
            env.close()
    draw.text((42, 678), "Evaluator views only. All presets reuse the same wall-occluded 7 x 7 agent interface.", font=font(17), fill="#93a4bd")
    canvas.save(output / "generation.png")

    summary = json.loads((STUDY / "summary.json").read_text())
    canvas = Image.fromarray(slide("Reserved-map comparison: 5,700 episodes", [],
                                  "Equal preset weights; three trained seeds per PPO condition. See report for uncertainty."))
    draw = ImageDraw.Draw(canvas)
    columns = [42, 550, 730, 920, 1100]
    for x, label in zip(columns, ("Agent", "Coverage", "Visited", "POIs", "Survival")):
        draw.text((x, 145), label, font=font(24, True), fill="#93a4bd")
    order = [("random", "Random"), ("frontier", "Frontier"), ("no_novelty", "PPO, no bonuses (old frame)"),
             ("no_memory", "PPO, no map"), ("main", "PPO, original map"),
             ("no_novelty_agent_aligned", "PPO, no bonuses (aligned)"),
             ("main_agent_aligned", "PPO, aligned map")]
    for i, (key, label) in enumerate(order):
        y = 202 + i * 45
        if key == "main_agent_aligned":
            draw.rounded_rectangle((30, y - 6, 1246, y + 42), radius=5, fill="#173c41")
        values = summary["overall"][key]
        cells = [label] + [f"{100 * values[m]:.1f}%" for m in ("revealed_coverage", "visited_coverage", "poi_fraction", "survived")]
        for x, value in zip(columns, cells):
            draw.text((x, y), value, font=font(23, key == "main_agent_aligned"), fill="#f2f5fb")
    draw.text((42, 550), "Frontier has higher coverage AUC; aligned PPO survives more often than frontier.", font=font(23), fill="#74d4c1")
    draw.text((42, 592), "Hard: aligned coverage 81.2% vs mapless 63.5%; survival 62.7% vs 70.7%.", font=font(23), fill="#efbe76")
    canvas.save(output / "comparison.png")
    shutil.copy2(ROOT / "output/pdf/technical-report.pdf", output / "technical-report.pdf")
    metadata = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": selected["sha256"],
                "training_seed": selected["seed"], "selected_steps": selected["selected_steps"],
                "memory_frame": selected["memory_frame"], "validation_score": selected["validation_score"],
                "selection": "Highest lexicographic validation AUC/discovery/survival score among the three aligned runs",
                "qualitative_note": "Easy/Medium/Hard and frontier maps fixed at validation seed 10000. Contact map 30000 and seed-11 loop map 30029 are explicitly selected illustrations, not aggregate evidence.",
                "comparison_seed_disclosure": "Primary: predeclared 40000-40099, no checkpoint selection on confirmation. Older 20000/30000 ranges exposed during diagnosis; archived evidence and disclosure retained.",
                "study_summary_sha256": digest(STUDY / "summary.json"),
                "report_sha256": digest(output / "technical-report.pdf"),
                "case_metrics": {key: trace["metrics"] for key, trace in traces.items()}}
    (output / "demo-provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def render(output, ui_video):
    metadata = json.loads((output / "demo-provenance.json").read_text())
    writer = imageio.get_writer(output / "demo.mp4", fps=FPS, codec="libx264", quality=7,
                                macro_block_size=1, ffmpeg_log_level="error")
    storyboard = []
    elapsed = 0

    def still(name, frame, seconds):
        nonlocal elapsed
        storyboard.append({"start": elapsed, "seconds": seconds, "title": name})
        Image.fromarray(frame).save(output / "previews" / f"{elapsed:03d}.png")
        for _ in range(seconds * FPS):
            writer.append_data(frame)
        elapsed += seconds

    def clip(name, path, seconds, caption=None):
        nonlocal elapsed
        reader = imageio.get_reader(path)
        try:
            length = reader.count_frames()
            # Some Windows DPI capture paths append black right/bottom padding.
            # Trim only that unused capture border, retaining the entire real UI.
            ui_crop = None
            if Path(path).resolve() == Path(ui_video).resolve():
                first = reader.get_data(0)
                ys, xs = np.nonzero(np.max(first, axis=2) > 10)
                ui_crop = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            moving_frames = (seconds - 2) * FPS
            indices = np.concatenate((np.linspace(0, max(0, length - 1), moving_frames).astype(int),
                                      np.full(2 * FPS, max(0, length - 1), dtype=int)))
            trace_path = Path(path).with_name("replay.json")
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
            for k, index in enumerate(indices):
                frame = reader.get_data(int(index))
                if ui_crop:
                    x0, y0, x1, y1 = ui_crop
                    frame = frame[y0:y1, x0:x1]
                if caption:
                    canvas = Image.new("RGB", SIZE, "#0b1120")
                    image = Image.fromarray(frame).convert("RGB")
                    image.thumbnail((1232, 624), Image.Resampling.LANCZOS)
                    canvas.paste(image, ((1280 - image.width) // 2, (624 - image.height) // 2))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((28, 636), name, font=font(24, True), fill="#74d4c1")
                    draw.text((28, 679), caption, font=font(18), fill="#dbe5f3")
                    frame = np.asarray(canvas)
                elif trace:
                    canvas = Image.fromarray(frame).convert("RGB")
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 1280, 100), fill="#0b1120")
                    draw.text((38, 25), name, font=font(29, True), fill="#f2f5fb")
                    policy = "frontier" if trace["agent"] == "frontier" else "aligned PPO / training seed 33"
                    draw.text((38, 70), f"Map {trace['seed']} | {policy} | Whole episode compressed; final outcome held 2 seconds",
                              font=font(17), fill="#93a4bd")
                    frame = np.asarray(canvas)
                if k in (0, seconds * FPS // 2, seconds * FPS - 1):
                    Image.fromarray(frame).save(output / "previews" / f"{elapsed:03d}-{k:03d}.png")
                writer.append_data(frame)
        finally:
            reader.close()
        storyboard.append({"start": elapsed, "seconds": seconds, "title": name,
                           "source": str(Path(path).relative_to(ROOT)), "ui_capture_padding_crop": ui_crop,
                           "presentation": "Whole source clip resampled; last outcome held for two seconds"})
        elapsed += seconds

    (output / "previews").mkdir(exist_ok=True)
    try:
        still("Goal and selected policy", slide("Learning exploration under partial observability", [
            "Explore connected maps, discover green POIs, and avoid a moving red patrol.",
            "PPO + observation-derived memory; agent-centered, heading-aligned readout.",
            "Demo/build model: training seed 33, selected at 3.75 million transitions.",
            "Selected by validation AUC (0.889863), not by demo or test performance."],
            "4-minute captioned demo / CPU inference / Farama Foundation MiniGrid (Apache-2.0)"), 12)
        clip("Portable UI: seed, preset, start / pause / step / reset", ui_video, 32,
             "Actual packaged application. Ground truth is evaluator-only; PPO reads aligned memory internally.")
        still("Procedural generation", np.asarray(Image.open(output / "generation.png").convert("RGB")), 12)
        for name, seconds, label in (("aligned-easy", 24, "Easy / validation map 10000"),
                                     ("aligned-medium", 28, "Medium / validation map 10000"),
                                     ("aligned-hard", 24, "Hard / size excluded from training"),
                                     ("frontier-medium", 18, "Frontier / same Medium map and information boundary"),
                                     ("aligned-contact", 18, "Threat contact / selected Hard map 30000")):
            clip(label, output / "replays" / name / "episode.mp4", seconds)
        still("Aggregate results", np.asarray(Image.open(output / "comparison.png").convert("RGB")), 22)
        still("Reward ablation and confirmation", slide("The missing reward ablation is complete", [
            "Aligned full reward: 90.8% coverage; without seen/visit bonuses: 59.6%.",
            "Three training seeds each; matched architecture, budgets, and PPO settings.",
            "Primary maps 40000-40099 were reserved before new training and testing.",
            "Earlier seed exposure remains disclosed; all checkpoints selected on validation."],
            "Per-preset reward gains have positive paired-map intervals. Exact reward weights remain heuristic."), 14)
        clip("Failure example: a different aligned run (seed 11)",
             output / "replays/loop-example-seed11/episode.mp4", 16,
             "Hard map 30029: survives 1,536 actions, 10.0% coverage, 99.9% loops. Post-hoc extreme example.")
        still("Run and inspect", slide("Portable Windows CPU package", [
            "1. Extract the complete ZIP; keep the EXE and _internal folder together.",
            "2. Launch UI.bat opens the aligned policy; Smoke test.bat verifies loading.",
            "3. Change seed / difficulty / agent; reset, run, pause, or single-step.",
            "No Python installation, GPU, network access, or retraining required.",
            "Includes source attribution, model metadata, and third-party licenses."],
            "Code: github.com/emkienngu123/MiniGridRL | Technical report and raw evidence accompany the demo."), 20)
    finally:
        writer.close()
    assert elapsed == 240, elapsed
    movie = imageio.get_reader(output / "demo.mp4")
    try:
        count = movie.count_frames()
        fps = movie.get_meta_data()["fps"]
        assert count == 240 * FPS and count / fps == 240
        assert movie.get_data(count - 1).shape == (720, 1280, 3)
    finally:
        movie.close()
    manifest = {**metadata, "demo_seconds": 240, "fps": FPS, "frames": count,
                "video_sha256": digest(output / "demo.mp4"), "ui_source_sha256": digest(ui_video),
                "storyboard": storyboard, "audio": "Silent; explanatory on-screen captions"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "storyboard.json").write_text(json.dumps(storyboard, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(output / "demo.mp4"), "seconds": elapsed, "sha256": manifest["video_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission/final")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ui-video")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if ROOT not in output.parents or output in {ROOT / "submission", ROOT / "submission/aligned"}:
        raise ValueError("Use a new submission subdirectory; historical artifacts must remain intact")
    prepare(output)
    if not args.prepare_only:
        if not args.ui_video or not Path(args.ui_video).is_file():
            raise ValueError("Supply an actual packaged-UI recording with --ui-video")
        render(output, Path(args.ui_video).resolve())


if __name__ == "__main__":
    main()
